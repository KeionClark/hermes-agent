"""Retention runs independently of the paused kanban dispatcher."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from gateway import kanban_watchers as kww
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _aged_resolution_event(conn, task_id):
    assert kb.block_task(conn, task_id, reason="missing access", kind="capability")
    assert kb.resolve_block_and_requeue(
        conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef",
    )
    conn.execute(
        "UPDATE task_events SET created_at = ? WHERE task_id = ? AND "
        "(kind = 'block_resolved' OR (kind = 'unblocked' AND payload LIKE '%block_resolved%'))",
        (int(time.time()) - kb.BLOCK_RESOLUTION_METADATA_PURGE_AGE_SECONDS, task_id),
    )
    conn.commit()


def test_retention_watcher_sweeps_while_dispatch_is_paused(monkeypatch):
    class Runner(kww.GatewayKanbanWatchersMixin):
        _running = True

    runner = Runner()
    sweeps = []

    async def direct_thread(fn, *args):
        sweeps.append(fn(*args))
        return sweeps[-1]

    async def stop_after_one(_interval):
        runner._running = False

    monkeypatch.setattr(kww, "_to_thread_process_service", direct_thread)
    monkeypatch.setattr(runner, "_sleep_between_ticks", stop_after_one)
    monkeypatch.setattr(kww, "_purge_block_resolution_metadata", lambda _kb: 2)
    monkeypatch.setattr(kww, "_kanban_dispatch_allowed", lambda: False)

    asyncio.run(runner._kanban_block_resolution_retention_watcher())
    assert sweeps == [2]


def test_retention_watcher_emits_an_alert_when_the_sweep_fails(monkeypatch, caplog):
    class Runner(kww.GatewayKanbanWatchersMixin):
        _running = True

    runner = Runner()

    async def direct_thread(fn, *args):
        return fn(*args)

    async def stop_after_one(_interval):
        runner._running = False

    def fail(_kb):
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(kww, "_to_thread_process_service", direct_thread)
    monkeypatch.setattr(runner, "_sleep_between_ticks", stop_after_one)
    monkeypatch.setattr(kww, "_purge_block_resolution_metadata", fail)
    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        asyncio.run(runner._kanban_block_resolution_retention_watcher())
    assert "ALERT: kanban block-resolution metadata retention sweep failed" in caplog.text


def test_retention_watcher_runs_real_default_board_sweep_while_dispatch_is_paused(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="expired resolution", assignee="builder")
        _aged_resolution_event(conn, task_id)

    class Runner(kww.GatewayKanbanWatchersMixin):
        _running = True

    runner = Runner()

    async def direct_thread(fn, *args):
        return fn(*args)

    async def stop_after_one(_interval):
        runner._running = False

    monkeypatch.setattr(kww, "_to_thread_process_service", direct_thread)
    monkeypatch.setattr(runner, "_sleep_between_ticks", stop_after_one)
    monkeypatch.setattr(kww, "_kanban_dispatch_allowed", lambda: False)
    asyncio.run(runner._kanban_block_resolution_retention_watcher())

    with kbc.connect() as conn:
        assert {event.kind for event in kb.list_events(conn, task_id)} == {"created", "blocked"}


def test_retention_sweep_enumerates_archived_boards(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.create_board("archived-board")
    with kbc.connect(board="archived-board") as conn:
        task_id = kb.create_task(conn, title="archived board expiry", assignee="builder")
        _aged_resolution_event(conn, task_id)

    archived_db = Path(kb.remove_board("archived-board", archive=True)["new_path"]) / "kanban.db"
    assert archived_db.is_file()
    assert kww._purge_block_resolution_metadata(kb) == 2
    with kbc.connect(db_path=archived_db) as conn:
        assert {event.kind for event in kb.list_events(conn, task_id)} == {"created", "blocked"}

"""Board-scoped watcher activation and lock ownership."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from gateway.kanban_watchers import GatewayKanbanWatchersMixin


@pytest.mark.asyncio
async def test_bounded_watcher_consumes_blocked_event_only_when_lock_owned(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        task = kb.create_task(conn, title="synthetic", assignee="owner")
        assert kb.claim_task(conn, task)
        assert kb.block_task(conn, task, kind="needs_input", reason="synthetic")
    watcher = GatewayKanbanWatchersMixin()
    watcher._running = True
    watcher._kanban_dispatcher_lock_handle = SimpleNamespace()

    async def stop(_interval):
        watcher._running = False

    monkeypatch.setattr(watcher, "_sleep_between_ticks", stop)
    await watcher._kanban_blocked_diagnosis_watcher(kb.DEFAULT_BOARD)
    with kbc.connect() as conn:
        assert len(kb.list_comments(conn, task)) == 1
        assert kb.get_task(conn, task).status == "blocked"
        cursor = conn.execute("SELECT event_id FROM blocked_diagnosis_cursor").fetchone()[0]
        assert cursor > 0
    watcher._running = True
    watcher._kanban_dispatcher_lock_handle = None
    await watcher._kanban_blocked_diagnosis_watcher(kb.DEFAULT_BOARD)
    with kbc.connect() as conn:
        assert len(kb.list_comments(conn, task)) == 1

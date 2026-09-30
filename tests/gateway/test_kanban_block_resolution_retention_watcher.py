"""Retention runs independently of the paused kanban dispatcher."""

from __future__ import annotations

import asyncio
import logging

from gateway import kanban_watchers as kww


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

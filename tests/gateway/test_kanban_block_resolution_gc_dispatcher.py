"""Embedded enforcement for block-resolution metadata retention."""

from __future__ import annotations

from types import SimpleNamespace

from gateway import kanban_watchers_dispatcher as kwd


def _settings():
    return kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)


def test_dispatcher_runs_block_resolution_metadata_gc_daily(monkeypatch):
    class Connection:
        def close(self):
            pass

    conn = Connection()
    calls = []
    kb = SimpleNamespace(
        DEFAULT_BOARD="default",
        kanban_db_path=lambda slug: SimpleNamespace(
            expanduser=lambda: SimpleNamespace(resolve=lambda: "/tmp/board"),
            stat=lambda: SimpleNamespace(st_mtime_ns=1, st_size=1),
        ),
        gc_events=lambda got_conn, *, older_than_seconds: calls.append((got_conn, older_than_seconds)) or 2,
    )
    dispatcher = kwd._KanbanDispatcher(kb, _settings())
    monkeypatch.setattr(dispatcher, "board_db_fingerprint", lambda slug: ("/tmp/board", 1, 1))
    monkeypatch.setattr(kwd, "_kbc", lambda: SimpleNamespace(connect=lambda *, board: conn))
    monkeypatch.setattr(kwd, "_kbd", lambda: SimpleNamespace(dispatch_once=lambda *args, **kwargs: "dispatched"))

    assert dispatcher.tick_once_for_board("default") == "dispatched"
    assert dispatcher.tick_once_for_board("default") == "dispatched"
    assert calls == [(conn, 0)]

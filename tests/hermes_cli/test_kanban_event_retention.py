"""Real disposable SQLite retention, cursor and router-state regression."""
import os
import sqlite3
import subprocess
import sys

import pytest

from hermes_cli import event_bus, kanban_event_router as router
from hermes_cli.kanban_db_connect import connect, write_txn
from hermes_cli.kanban_event_retention import (
    RETENTION_SECONDS, purge_all_expired, purge_expired, require_restricted_board,
)


def _event(conn, typ="card.status_changed"):
    with write_txn(conn):
        return event_bus.publish(conn, event_type=typ, source="kanban", payload={"task_id": "t_test", "from": "todo", "to": "ready"})


def test_cutoff_cursor_restart_references_and_unrelated_history(tmp_path):
    path = tmp_path / "board.db"
    conn = connect(path)
    router.migrate_router(conn)
    old = _event(conn)
    boundary = _event(conn)
    fresh = _event(conn)
    now = fresh.created_at + 100
    cutoff = now - RETENTION_SECONDS
    conn.execute("UPDATE hermes_events SET created_at=? WHERE id=?", (cutoff - 0.01, old.id))
    conn.execute("UPDATE hermes_events SET created_at=? WHERE id=?", (cutoff, boundary.id))
    conn.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES('legacy','blocked','{}',1)")
    event_bus.subscribe(conn, "lag", event_types={"card.status_changed"}, start="beginning")
    conn.execute("INSERT INTO hermes_router_inbox VALUES(?,?,?,?)", ("lag", old.id, "card.status_changed", now))
    conn.execute("INSERT INTO hermes_router_inbox VALUES(?,?,?,?)", ("lag", boundary.id, "card.status_changed", cutoff - 0.01))
    conn.execute("INSERT INTO hermes_router_inbox VALUES(?,?,?,?)", ("lag", fresh.id, "card.status_changed", now))
    conn.execute("INSERT INTO hermes_router_failures VALUES(?,?,?,?,?)", ("lag", old.id, 1, now + 5, "callback_error"))
    conn.execute("INSERT INTO hermes_router_deadletters VALUES(?,?,?,?,?)", ("lag", old.id, 5, "callback_error", now))
    assert purge_expired(conn, now=now) == (1, 2)
    assert [e.id for e in event_bus.poll(conn, "lag")] == [boundary.id, fresh.id]
    with write_txn(conn):
        event_bus.ack(conn, "lag", boundary.id)
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_failures").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_deadletters").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == 1
    assert purge_expired(conn, now=now) == (0, 0)
    conn.close()
    conn = connect(path)
    assert [e.id for e in event_bus.poll(conn, "lag")] == [fresh.id]
    assert conn.execute("SELECT last_event_id FROM hermes_event_cursors WHERE subscriber_id='lag'").fetchone()[0] == boundary.id
    conn.close()


def test_bounded_purge_and_rollback(tmp_path):
    conn = connect(tmp_path / "board.db")
    router.migrate_router(conn)
    events = [_event(conn) for _ in range(4)]
    now = events[-1].created_at + RETENTION_SECONDS + 1
    assert purge_all_expired(conn, now=now, batch_size=2, max_batches=1) == (2, 0, True)
    assert purge_all_expired(conn, now=now, batch_size=2, max_batches=2) == (2, 0, False)
    assert purge_all_expired(conn, now=now) == (0, 0, False)
    event = _event(conn)
    conn.execute("UPDATE hermes_events SET created_at=? WHERE id=?", (now - RETENTION_SECONDS - 1, event.id))
    conn.execute("CREATE TRIGGER deny_retention BEFORE DELETE ON hermes_events BEGIN SELECT RAISE(ABORT, 'fail'); END")
    with pytest.raises(sqlite3.IntegrityError):
        purge_expired(conn, now=now)
    assert conn.execute("SELECT COUNT(*) FROM hermes_events").fetchone()[0] == 1
    conn.execute("DROP TRIGGER deny_retention")
    assert purge_expired(conn, now=now) == (1, 0)
    conn.close()


def test_old_bus_without_router_and_runner_access(tmp_path):
    path = tmp_path / "board.db"
    conn = connect(path)
    old = _event(conn)
    now = old.created_at + RETENTION_SECONDS + 1
    assert purge_expired(conn, now=now) == (1, 0)
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='hermes_router_inbox'").fetchone() is None
    conn.close()
    if not hasattr(os, "chmod") or os.name == "nt":
        pytest.skip("POSIX filesystem permissions")
    with pytest.raises(PermissionError):
        require_restricted_board(path)
    tmp_path.chmod(0o700)
    path.chmod(0o600)
    require_restricted_board(path)
    proc = subprocess.run([sys.executable, "-m", "hermes_cli.kanban_event_router", "--board-path", str(path), "--purge-only"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    conn = connect(path)
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='hermes_event_cursors'").fetchone() is not None
    assert conn.execute("SELECT COUNT(*) FROM hermes_event_cursors").fetchone()[0] == 0
    conn.close()


def test_private_cli_routing_purges_first_and_rejects_other_interests(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX filesystem permissions")
    path = tmp_path / "board.db"
    conn = connect(path)
    stale = _event(conn)
    conn.execute("UPDATE hermes_events SET created_at=1 WHERE id=?", (stale.id,))
    conn.close()
    tmp_path.chmod(0o700)
    path.chmod(0o600)
    cmd = [sys.executable, "-m", "hermes_cli.kanban_event_router", "--board-path", str(path),
           "--subscriber-id", "resolver", "--event-type", "card.status_changed"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    conn = connect(path)
    assert conn.execute("SELECT COUNT(*) FROM hermes_events").fetchone()[0] == 0
    new = _event(conn)
    conn.close()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    conn = connect(path)
    assert conn.execute("SELECT event_id FROM hermes_router_inbox").fetchone()[0] == new.id
    conn.close()
    assert subprocess.run(cmd[:-1] + ["unapproved"], capture_output=True).returncode != 0

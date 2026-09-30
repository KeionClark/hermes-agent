"""Disposable-board integration and crash/poison/concurrency contracts for opt-in router."""
import sqlite3
import subprocess
import sys
import threading

import pytest

from hermes_cli import event_bus, kanban_event_router as router
from hermes_cli.kanban_db_connect import connect, write_txn


def publish(conn, kind="a", key=None):
    with write_txn(conn):
        return event_bus.publish(conn, event_type=kind, source="router-test", payload={"status": "ready"}, idempotency_key=key)


def test_latest_multiple_filters_order_dedup_restart_and_cli(tmp_path):
    path = tmp_path / "board.db"
    conn = connect(path)
    old = publish(conn, key="old")
    router.run_once(conn, {"a": router.Subscriber(frozenset({"a"}), router.enqueue_inbox("a")),
                           "b": router.Subscriber(frozenset({"b"}), router.enqueue_inbox("b"))})
    a = publish(conn, "a", "a1")
    assert publish(conn, "a", "a1").id == a.id
    gap = publish(conn, "b")
    later = publish(conn, "a")
    # Timestamp deliberately reversed: ID is the only delivery order.
    conn.execute("UPDATE hermes_events SET created_at=? WHERE id=?", (1, later.id))
    assert router.run_once(conn, {"a": router.Subscriber(frozenset({"a"}), router.enqueue_inbox("a")),
                                  "b": router.Subscriber(frozenset({"b"}), router.enqueue_inbox("b"))}, max_events=2) == {"a": 2, "b": 1}
    assert [(r[0], r[1]) for r in conn.execute("SELECT subscriber_id,event_id FROM hermes_router_inbox ORDER BY subscriber_id,event_id")] == [("a", a.id), ("a", later.id), ("b", gap.id)]
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_inbox WHERE event_id=?", (old.id,)).fetchone()[0] == 0
    assert conn.execute("SELECT version FROM hermes_router_schema_version").fetchone()[0] == router._VERSION
    conn.close()
    proc = subprocess.run([sys.executable, "-m", "hermes_cli.kanban_event_router", "--board-path", str(path),
                           "--subscriber-id", "a", "--event-type", "a"], capture_output=True, text=True)
    # The operator-facing runner now refuses a public board and a non-card
    # event type; low-level router fixtures remain general-purpose.
    assert proc.returncode != 0
    conn = connect(path)
    assert router.run_once(conn, {"a": router.Subscriber(frozenset({"a"}), router.enqueue_inbox("a"))}) == {"a": 0}
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_inbox").fetchone()[0] == 3
    with pytest.raises(ValueError, match="interests differ"):
        router.run_once(conn, {"a": router.Subscriber(frozenset({"b"}), router.enqueue_inbox("a"))})
    conn.close()


def test_atomic_crash_before_after_ack_and_retry_deadletter(tmp_path):
    conn = connect(tmp_path / "board.db")
    effects = []
    def callback(c, event):
        c.execute("INSERT OR IGNORE INTO hermes_router_inbox VALUES(?,?,?,?)", ("s", event.id, event.event_type, 0))
        effects.append(event.id)
        if len(effects) == 1:
            raise RuntimeError("sensitive PII NEVER stored")
    sub = {"s": router.Subscriber(frozenset({"a"}), callback)}
    router.run_once(conn, sub)
    event = publish(conn)
    assert router.run_once(conn, sub, now=10) == {"s": 0}
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_inbox").fetchone()[0] == 0
    row = conn.execute("SELECT attempts,next_at,error_code FROM hermes_router_failures").fetchone()
    assert tuple(row) == (1, 11, "callback_error")
    assert router.run_once(conn, sub, now=10.5) == {"s": 0}
    assert router.run_once(conn, sub, now=11) == {"s": 1}
    assert router.run_once(conn, sub, now=12) == {"s": 0}
    assert effects == [event.id, event.id]
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_inbox").fetchone()[0] == 1
    def poison(c, e):
        raise ValueError("patient data")
    # Existing registration cannot be reset; callback implementation may change.
    publish(conn)
    for now in (20, 21, 23):
        router.run_once(conn, {"s": router.Subscriber(frozenset({"a"}), poison)}, now=now, max_attempts=3)
    row = conn.execute("SELECT attempts,error_code FROM hermes_router_deadletters").fetchone()
    assert tuple(row) == (3, "callback_data_error")
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_failures").fetchone()[0] == 0
    publish(conn)
    assert router.run_once(conn, sub, now=24) == {"s": 1}
    conn.close()


def test_concurrent_same_subscriber_once_and_legacy_rows(tmp_path):
    path = tmp_path / "legacy.db"
    conn = connect(path)
    conn.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES('old','blocked','{}',1)")
    old = conn.execute("SELECT id FROM task_events WHERE task_id='old'").fetchone()[0]
    sub = {"s": router.Subscriber(frozenset({"a"}), router.enqueue_inbox("s"))}
    router.run_once(conn, sub)
    publish(conn)
    conn.close()
    errors = []
    barrier = threading.Barrier(2)
    def worker():
        c = connect(path)
        try:
            barrier.wait()
            router.run_once(c, sub)
        except Exception as exc:
            errors.append(exc)
        finally:
            c.close()
    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    conn = connect(path)
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_inbox").fetchone()[0] == 1
    assert conn.execute("SELECT id FROM task_events WHERE task_id='old'").fetchone()[0] == old
    router.migrate_router(conn)
    assert conn.execute("SELECT version FROM hermes_router_schema_version").fetchone()[0] == router._VERSION
    conn.close()


def test_poison_a_does_not_hold_b_and_migration_is_atomic(tmp_path):
    conn = connect(tmp_path / "board.db")
    def poison(c, e):
        raise RuntimeError("sensitive")
    regs = {"a": router.Subscriber(frozenset({"a"}), poison),
            "b": router.Subscriber(frozenset({"b"}), router.enqueue_inbox("b"))}
    router.run_once(conn, regs)
    publish(conn, "a")
    event_b = publish(conn, "b")
    assert router.run_once(conn, regs, now=10) == {"a": 0, "b": 1}
    assert conn.execute("SELECT event_id FROM hermes_router_inbox").fetchone()[0] == event_b.id
    conn.close()

    # Old bus schema, no router schema: failing DDL must not stamp a version.
    conn = sqlite3.connect(tmp_path / "old.db", isolation_level=None)
    event_bus.migrate_event_bus(conn)
    conn.execute("CREATE TABLE hermes_router_inbox (wrong TEXT)")
    # IF NOT EXISTS tolerates an existing table; insert dependent operation to
    # force an actual migration error through a version-table constraint.
    conn.execute("CREATE TABLE hermes_router_schema_version (id INTEGER PRIMARY KEY, version INTEGER NOT NULL CHECK(version<2))")
    with pytest.raises(sqlite3.IntegrityError):
        router.migrate_router(conn)
    assert conn.execute("SELECT COUNT(*) FROM hermes_router_schema_version").fetchone()[0] == 0
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='hermes_router_failures'").fetchone() is None
    conn.close()


def test_cli_requires_explicit_registration(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "hermes_cli.kanban_event_router", "--board-path", str(tmp_path / "missing")], capture_output=True)
    assert proc.returncode != 0

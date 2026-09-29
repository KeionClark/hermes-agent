"""Opt-in, synchronous, bounded, same-board event router (not a worker launcher).

Callbacks execute inside BEGIN IMMEDIATE and MUST do only short, idempotent
same-connection database work. No network, subprocess, LLM, message, or nested
commit is allowed. A callback's external effects cannot be rolled back; callers
must not register one with external effects. The router is deliberately not wired
into any background service. One invocation consumes at most max_events for EACH
subscriber, alternating streams after each event. A blocking callback can still
block SQLite's single writer: callers must bound callback execution themselves.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Mapping

from hermes_cli import event_bus
from hermes_cli.kanban_db_connect import write_txn

_VERSION = 2
MAX_ATTEMPTS = 5
BASE_DELAY = 1.0
MAX_DELAY = 60.0


@dataclass(frozen=True)
class Subscriber:
    event_types: frozenset[str]
    callback: Callable[[sqlite3.Connection, event_bus.Event], None]

    def __post_init__(self):
        if not isinstance(self.event_types, frozenset) or not self.event_types or not all(
            isinstance(t, str) and t for t in self.event_types
        ):
            raise ValueError("event_types must be a nonempty frozenset of strings")
        if not callable(self.callback):
            raise ValueError("callback must be callable")


def migrate_router(conn: sqlite3.Connection) -> None:
    """Explicit, additive, atomic migration independent of event-bus version."""
    with write_txn(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS hermes_router_schema_version "
                     "(id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL)")
        row = conn.execute("SELECT version FROM hermes_router_schema_version WHERE id=1").fetchone()
        version = row[0] if row else 0
        if version > _VERSION:
            raise RuntimeError("router schema newer than this binary")
        if version < 1:
            conn.execute("CREATE TABLE IF NOT EXISTS hermes_router_failures ("
                         "subscriber_id TEXT NOT NULL, event_id INTEGER NOT NULL, "
                         "attempts INTEGER NOT NULL, next_at REAL NOT NULL, "
                         "error_code TEXT NOT NULL, PRIMARY KEY(subscriber_id,event_id))")
            conn.execute("CREATE TABLE IF NOT EXISTS hermes_router_deadletters ("
                         "subscriber_id TEXT NOT NULL, event_id INTEGER NOT NULL, "
                         "attempts INTEGER NOT NULL, error_code TEXT NOT NULL, "
                         "dead_at REAL NOT NULL, PRIMARY KEY(subscriber_id,event_id))")
        if version < 2:
            conn.execute("CREATE TABLE IF NOT EXISTS hermes_router_inbox ("
                         "subscriber_id TEXT NOT NULL, event_id INTEGER NOT NULL, "
                         "event_type TEXT NOT NULL, enqueued_at REAL NOT NULL, "
                         "PRIMARY KEY(subscriber_id,event_id))")
        if version < _VERSION:
            conn.execute("INSERT OR REPLACE INTO hermes_router_schema_version(id,version) VALUES(1,?)", (_VERSION,))


def enqueue_inbox(subscriber_id: str) -> Callable[[sqlite3.Connection, event_bus.Event], None]:
    """DB-only event-id deduplicated effect; no worker is scheduled here."""
    def callback(conn: sqlite3.Connection, event: event_bus.Event) -> None:
        conn.execute("INSERT OR IGNORE INTO hermes_router_inbox "
                     "(subscriber_id,event_id,event_type,enqueued_at) VALUES(?,?,?,?)",
                     (subscriber_id, event.id, event.event_type, time.time()))
    return callback


def _register(conn: sqlite3.Connection, registrations: Mapping[str, Subscriber]) -> None:
    # Do not silently inherit a previous registration with different interests.
    for subscriber_id, subscriber in registrations.items():
        if not isinstance(subscriber, Subscriber):
            raise TypeError("registration must be a Subscriber")
        row = conn.execute("SELECT event_types FROM hermes_event_cursors WHERE subscriber_id=?",
                           (subscriber_id,)).fetchone()
        expected = json.dumps(sorted(subscriber.event_types))
        if row is not None and row[0] != expected:
            raise ValueError("registered interests differ from persisted interests")
        event_bus.subscribe(conn, subscriber_id, event_types=subscriber.event_types, start="latest")


def _error_code(exc: Exception) -> str:
    # Classify by a closed allowlist: never persist exception messages or custom class names.
    if isinstance(exc, sqlite3.Error):
        return "sqlite_error"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "callback_data_error"
    return "callback_error"


def _record_failure(conn: sqlite3.Connection, subscriber_id: str, event_id: int,
                    now: float, error_code: str, max_attempts: int) -> bool:
    """Record failure only if the same event remains at the stream head.

    Return True if deadlettered and ACKed. Failure accounting is separate from
    the rolled-back callback transaction and serialized under another write lock.
    """
    with write_txn(conn):
        events = event_bus.poll(conn, subscriber_id, limit=1)
        if not events or events[0].id != event_id:
            return False
        row = conn.execute("SELECT attempts FROM hermes_router_failures "
                           "WHERE subscriber_id=? AND event_id=?", (subscriber_id, event_id)).fetchone()
        attempts = (row[0] if row else 0) + 1
        if attempts >= max_attempts:
            conn.execute("INSERT OR IGNORE INTO hermes_router_deadletters "
                         "(subscriber_id,event_id,attempts,error_code,dead_at) VALUES(?,?,?,?,?)",
                         (subscriber_id, event_id, attempts, error_code, now))
            event_bus.ack(conn, subscriber_id, event_id)
            conn.execute("DELETE FROM hermes_router_failures WHERE subscriber_id=? AND event_id=?",
                         (subscriber_id, event_id))
            return True
        delay = min(MAX_DELAY, BASE_DELAY * (2 ** min(attempts - 1, 16)))
        conn.execute("INSERT INTO hermes_router_failures "
                     "(subscriber_id,event_id,attempts,next_at,error_code) VALUES(?,?,?,?,?) "
                     "ON CONFLICT(subscriber_id,event_id) DO UPDATE SET "
                     "attempts=excluded.attempts,next_at=excluded.next_at,error_code=excluded.error_code",
                     (subscriber_id, event_id, attempts, now + delay, error_code))
        return False


def run_once(conn: sqlite3.Connection, registrations: Mapping[str, Subscriber], *,
             now: float | None = None, max_events: int = 100,
             max_attempts: int = MAX_ATTEMPTS) -> dict[str, int]:
    """Register at latest if new, then attempt at most max_events per stream.

    Returns successful effect counts keyed by subscriber. Backoff holds only
    that subscriber's oldest matching event. No historical backfill is implicit.
    Requires a canonical autocommit board connection, not an existing transaction.
    """
    if not isinstance(max_events, int) or not 1 <= max_events <= 1000:
        raise ValueError("max_events must be 1..1000")
    if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 100:
        raise ValueError("max_attempts must be 1..100")
    if conn.in_transaction:
        raise RuntimeError("run_once requires an autocommit connection")
    now = time.time() if now is None else now
    migrate_router(conn)
    with write_txn(conn):
        _register(conn, registrations)
    counts = {sid: 0 for sid in registrations}
    active = set(registrations)
    # Round robin; never let a poison stream consume other streams' budget.
    for _ in range(max_events):
        if not active:
            break
        for sid, subscriber in registrations.items():
            if sid not in active:
                continue
            # The read outside the transaction is only advisory; re-poll after
            # BEGIN IMMEDIATE, including the retry check, to serialize contenders.
            if not event_bus.poll(conn, sid, limit=1):
                active.remove(sid)
                continue
            failed: tuple[int, str] | None = None
            attempted = False
            try:
                with write_txn(conn):
                    events = event_bus.poll(conn, sid, limit=1)
                    if events:
                        event = events[0]
                        retry = conn.execute("SELECT next_at FROM hermes_router_failures "
                                             "WHERE subscriber_id=? AND event_id=?",
                                             (sid, event.id)).fetchone()
                        if retry is not None and retry[0] > now:
                            active.remove(sid)
                        else:
                            attempted = True
                            try:
                                subscriber.callback(conn, event)
                            except Exception as exc:
                                failed = (event.id, _error_code(exc))
                                raise _CallbackFailed from None
                            event_bus.ack(conn, sid, event.id)
                            conn.execute("DELETE FROM hermes_router_failures WHERE subscriber_id=? AND event_id=?",
                                         (sid, event.id))
            except _CallbackFailed:
                assert failed is not None
                dead = _record_failure(conn, sid, failed[0], now, failed[1], max_attempts)
                if not dead:
                    active.remove(sid)
            if attempted and failed is None:
                counts[sid] += 1
            elif not attempted and sid in active:
                active.remove(sid)
    return counts


class _CallbackFailed(Exception):
    """Discard callback error text before crossing transaction boundary."""


def main(argv: list[str] | None = None) -> int:
    """Explicit, finite opt-in runner for a selected disposable/approved board.

    Each invocation has a finite number of ticks; never defaults to the live
    board or an all-events subscription. Operator owns approval of board path.
    """
    import argparse
    from pathlib import Path
    from hermes_cli.kanban_db_connect import connect

    parser = argparse.ArgumentParser(description="Opt-in DB-only kanban event inbox")
    parser.add_argument("--board-path", type=Path, required=True)
    parser.add_argument("--subscriber-id", required=True)
    parser.add_argument("--event-type", action="append", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--max-events", type=int, default=100)
    args = parser.parse_args(argv)
    if not args.board_path.is_file() or args.iterations < 1 or args.iterations > 10000 or args.interval < 0:
        parser.error("board must exist; iterations must be 1..10000; interval must be nonnegative")
    sub = Subscriber(frozenset(args.event_type), enqueue_inbox(args.subscriber_id))
    conn = connect(args.board_path)
    try:
        for iteration in range(args.iterations):
            run_once(conn, {args.subscriber_id: sub}, max_events=args.max_events)
            if iteration + 1 < args.iterations:
                time.sleep(args.interval)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

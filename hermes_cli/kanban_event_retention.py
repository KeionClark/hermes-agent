"""Explicit 30-day privacy retention for the opt-in Kanban bus and router inbox.

Invoke from the finite router runner or a separately scheduled, trusted Hermes
maintenance invocation. A successful bounded pass is NOT a claim that an entire
backlog was removed: repeat until both returned counts are zero. No automatic
migration, subscriber registration, or worker creation occurs here.
"""
from __future__ import annotations

import os
import sqlite3
import stat
import time
from pathlib import Path

from hermes_cli.kanban_db_connect import write_txn

RETENTION_SECONDS = 30 * 24 * 60 * 60
DEFAULT_BATCH = 500


def require_restricted_board(path: Path) -> None:
    """Refuse a runner target readable by other local users or reached via symlink.

    SQLite WAL/SHM and backups must be protected by the private parent directory.
    This is an operator preflight, not a replacement for filesystem ACL review.
    """
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise PermissionError("router board and parent must not be symlinks")
    for target in (path.parent, path):
        info = target.stat()
        if not (stat.S_ISDIR(info.st_mode) if target == path.parent else stat.S_ISREG(info.st_mode)):
            raise PermissionError("router board must be a regular file in a private directory")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise PermissionError("router board and directory must be owned by the runner")
        if info.st_mode & 0o077:
            raise PermissionError("router board and directory require owner-only permissions")


def purge_expired(conn: sqlite3.Connection, *, now: float | None = None,
                  batch_size: int = DEFAULT_BATCH) -> tuple[int, int]:
    """Delete at most one batch from each table, atomically, strictly before cutoff.

    Old event IDs are never reused (AUTOINCREMENT). A lagging cursor remains
    monotonic; poll skips expired holes and ACK checks the next *surviving*
    event. Retry and deadletter references for purged events are removed in the
    same transaction, so restart cannot retain references to deleted events.
    A recent inbox row referencing an expired event is also erased: event age,
    not enqueue time, wins the privacy limit. A standalone stale inbox row is
    erased even if its event is new or already missing.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be 1..1000")
    if conn.in_transaction:
        raise RuntimeError("purge_expired requires an autocommit board connection")
    now = time.time() if now is None else now
    cutoff = now - RETENTION_SECONDS
    with write_txn(conn):
        # Router migration is explicit; a bus-only board still needs its events
        # expired without creating router tables or changing unrelated schemas.
        has_inbox = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='hermes_router_inbox'").fetchone() is not None
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM hermes_events WHERE created_at < ? ORDER BY created_at,id LIMIT ?",
            (cutoff, batch_size))]
        inbox_deleted = 0
        if has_inbox:
            inbox_deleted = conn.execute(
                "DELETE FROM hermes_router_inbox WHERE rowid IN ("
                "SELECT rowid FROM hermes_router_inbox WHERE enqueued_at < ? "
                "ORDER BY enqueued_at, rowid LIMIT ?)", (cutoff, batch_size)).rowcount
        if ids:
            placeholders = ",".join("?" for _ in ids)
            if has_inbox:
                inbox_deleted += conn.execute(
                    f"DELETE FROM hermes_router_inbox WHERE event_id IN ({placeholders})", ids).rowcount
            for table in ("hermes_router_failures", "hermes_router_deadletters"):
                if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                    conn.execute(f"DELETE FROM {table} WHERE event_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM hermes_events WHERE id IN ({placeholders})", ids)
        return len(ids), inbox_deleted


def purge_all_expired(conn: sqlite3.Connection, *, now: float | None = None,
                      batch_size: int = DEFAULT_BATCH, max_batches: int = 100) -> tuple[int, int, bool]:
    """Bounded drain. Boolean indicates whether another pass may be needed."""
    if not isinstance(max_batches, int) or isinstance(max_batches, bool) or not 1 <= max_batches <= 10000:
        raise ValueError("max_batches must be 1..10000")
    now = time.time() if now is None else now
    events = inbox = 0
    for _ in range(max_batches):
        e, i = purge_expired(conn, now=now, batch_size=batch_size)
        events += e
        inbox += i
        if e < batch_size and i < batch_size:
            return events, inbox, False
    return events, inbox, True

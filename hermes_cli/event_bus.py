"""
Durable SQLite event substrate with per-subscriber ACK cursors.

P0a phase: standalone, dark by default — no router, no publishers, no live
activation.  Importable by later P0a/P0b modules once integration is gated.

Design contract: t_a5105ab5 / P0A_EVENT_BUS_CONTRACT.md.

Public API
----------
- ``publish(conn, *, event_type, source, payload, idempotency_key=None) -> Event``
- ``subscribe(conn, subscriber_id, *, event_types=None, start='latest') -> int``
- ``poll(conn, subscriber_id, *, limit=100) -> list[Event]``
- ``ack(conn, subscriber_id, event_id) -> None``
- ``migrate_event_bus(conn) -> None``

Ownership: sole owner of this module, ``kanban_db_connect.py`` migration hook,
``tests/hermes_cli/test_event_bus.py``, and the benchmark script.  Does not
touch router, dispatcher, publishers, or the live board.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Collection, Literal, Mapping, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Event:
    """Immutable event record returned by ``publish`` and ``poll``."""

    id: int
    event_type: str
    source: str
    payload: dict[str, Any]
    created_at: float
    idempotency_key: Optional[str]


# ---------------------------------------------------------------------------
# Payload allowlist — fail-closed
# ---------------------------------------------------------------------------

# Flat set of allowed top-level and nested field names.  Any key not in this
# set is rejected, regardless of nesting depth.  Explicitly forbidden markers
# are also checked (defense-in-depth) but the allowlist is the gate.
_ALLOWED_PAYLOAD_KEYS: frozenset[str] = frozenset({
    # Task lifecycle
    "task_id", "from", "to", "run_id", "kind",
    # Block / triage
    "reason_code", "block_kind",
    # Metrics / goals
    "metric_id", "goal_id", "value", "threshold", "direction",
    # Provenance
    "source_timestamp", "payload_hash",
    # Identifiers
    "subscriber_id", "event_id", "event_type",
    # Generic safe scalars
    "status", "outcome", "profile", "tenant", "board",
    "count", "duration_ms", "attempt", "version",
    "idempotency_key", "parent_id", "child_id",
    "step_key", "workflow_template_id",
})

# Explicitly forbidden at ANY nesting depth — even if the allowlist were
# accidentally expanded, these hard-block.
_FORBIDDEN_PAYLOAD_KEYS: frozenset[str] = frozenset({
    # PII / PHI
    "content", "body", "text", "message",
    "patient", "contact", "name", "email", "phone", "address",
    # Credentials
    "credentials", "token", "auth", "password", "secret", "api_key",
    # Verbose / unsafe
    "traceback", "raw", "metadata",
    "title", "summary", "description", "comment",
    "webhook_body", "chat_id", "user_id",
})

_MAX_PAYLOAD_SIZE = 8192  # bytes of canonical JSON
_MAX_FIELD_LEN = 256      # event_type, source string length


def _validate_payload(payload: Mapping[str, Any]) -> str:
    """Validate and canonically serialize a payload mapping.

    Returns the canonical JSON string.  Raises ``ValueError`` on any
    violation.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a JSON object (Mapping)")

    def _check_value(v: Any, path: str) -> Any:
        """Recursively validate values; return the cleaned value."""
        if v is None or isinstance(v, (bool, int, float, str)):
            if isinstance(v, str) and len(v) > _MAX_FIELD_LEN:
                raise ValueError(
                    f"payload string value at '{path}' exceeds "
                    f"{_MAX_FIELD_LEN} chars"
                )
            return v
        if isinstance(v, (list, tuple)):
            return [_check_value(item, f"{path}[]") for item in v]
        if isinstance(v, Mapping):
            return _check_obj(v, path)
        raise ValueError(
            f"payload value at '{path}' has unsupported type "
            f"{type(v).__name__}; only JSON scalars, lists, and "
            f"objects are allowed"
        )

    def _check_obj(obj: Mapping[str, Any], prefix: str) -> dict[str, Any]:
        """Validate an object's keys and values recursively."""
        result: dict[str, Any] = {}
        for key, val in obj.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"payload key at '{prefix}' must be a string, "
                    f"got {type(key).__name__}"
                )
            full_path = f"{prefix}.{key}" if prefix else key

            # Forbidden check first (hard block)
            if key in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(
                    f"payload key '{key}' at '{full_path}' is forbidden; "
                    f"normalize to IDs, enumerated codes, or hashes"
                )
            # Allowlist check
            if key not in _ALLOWED_PAYLOAD_KEYS:
                raise ValueError(
                    f"payload key '{key}' at '{full_path}' is not in the "
                    f"event bus allowlist; add it to _ALLOWED_PAYLOAD_KEYS "
                    f"if this is a known safe field"
                )
            result[key] = _check_value(val, full_path)
        return result

    cleaned = _check_obj(payload, "")
    canonical = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    if len(canonical.encode("utf-8")) > _MAX_PAYLOAD_SIZE:
        raise ValueError(
            f"payload exceeds {_MAX_PAYLOAD_SIZE} bytes after serialization"
        )
    return canonical


# ---------------------------------------------------------------------------
# Schema & migration
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1

_SCHEMA_V1 = """\
CREATE TABLE IF NOT EXISTS hermes_event_schema_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS hermes_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT NOT NULL,
    source          TEXT NOT NULL,
    payload         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    idempotency_key TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_events_idempotency
    ON hermes_events(source, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_events_type_id
    ON hermes_events(event_type, id);

CREATE TABLE IF NOT EXISTS hermes_event_cursors (
    subscriber_id  TEXT PRIMARY KEY,
    event_types    TEXT NOT NULL,
    last_event_id  INTEGER NOT NULL DEFAULT 0 CHECK (last_event_id >= 0),
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
"""


def migrate_event_bus(conn: sqlite3.Connection) -> None:
    """Idempotent migration to the current event-bus schema version.

    Called from ``kanban_db_connect.connect`` after the core schema and
    column migrations.  Uses ``hermes_event_schema_version`` (singleton
    integer) — independent of ``PRAGMA user_version`` used by other
    subsystems.

    The canonical connection uses ``isolation_level=None`` (autocommit), so
    bare ``conn.execute`` of DDL auto-commits each statement independently.
    To guarantee atomicity (all-or-nothing: version table + bus DDL + stamp),
    this function wraps every mutating step in an explicit ``BEGIN IMMEDIATE``
    / ``COMMIT`` pair and rolls back on any failure.

    Uses ``conn.execute`` — NOT ``conn.executescript`` which auto-commits
    any open transaction.
    """
    # Probe version outside a write transaction (read-only).  The version
    # table itself may not exist yet; we handle that below inside the txn.
    try:
        row = conn.execute(
            "SELECT version FROM hermes_event_schema_version WHERE id = 1"
        ).fetchone()
        current = row[0] if row else 0
    except sqlite3.OperationalError:
        # Table does not exist yet — first run.
        current = 0

    if current > _SCHEMA_VERSION:
        raise RuntimeError(
            f"hermes_event_schema_version {current} > {_SCHEMA_VERSION}; "
            f"this binary is too old to read this event bus schema"
        )

    if current >= _SCHEMA_VERSION:
        return  # Already at target version.

    # --- Atomic write block: all DDL + stamp in one transaction. ----------
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Re-check version under the write lock (another process may have
        # migrated between our read above and acquiring the lock).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS hermes_event_schema_version "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
        )
        row = conn.execute(
            "SELECT version FROM hermes_event_schema_version WHERE id = 1"
        ).fetchone()
        current = row[0] if row else 0

        if current >= _SCHEMA_VERSION:
            conn.execute("COMMIT")
            return  # Another writer got here first.

        # Version 0 → 1: create all tables and indexes.
        if current < 1:
            for stmt in _SCHEMA_V1.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

            # Stamp version.
            conn.execute(
                "INSERT OR REPLACE INTO hermes_event_schema_version "
                "(id, version) VALUES (1, ?)",
                (_SCHEMA_VERSION,),
            )

        conn.execute("COMMIT")
    except BaseException:
        with _suppress_rollback():
            conn.execute("ROLLBACK")
        raise


def _suppress_rollback():
    """Context manager that suppresses errors during ROLLBACK.

    A ROLLBACK after a failed DDL should not mask the original exception.
    """
    import contextlib
    return contextlib.suppress(Exception)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def publish(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    source: str,
    payload: Mapping[str, Any],
    idempotency_key: Optional[str] = None,
) -> Event:
    """Publish an event into the bus under the caller's active transaction.

    The caller MUST hold an active write transaction (``conn.in_transaction``
    is True) — typically via ``kanban_db_connect.write_txn(conn)``.  This
    couples the event atomically with whatever board mutation the caller is
    performing.

    With a non-None ``idempotency_key``, a duplicate ``(source, idempotency_key)``
    returns the original ``Event`` unchanged (even if ``payload`` differs).
    """
    # --- Guards ---
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "publish() requires an active write transaction; "
            "use kanban_db_connect.write_txn(conn)"
        )
    if not event_type or not isinstance(event_type, str):
        raise ValueError("event_type must be a non-empty string")
    if len(event_type) > _MAX_FIELD_LEN:
        raise ValueError(f"event_type exceeds {_MAX_FIELD_LEN} chars")
    if not source or not isinstance(source, str):
        raise ValueError("source must be a non-empty string")
    if len(source) > _MAX_FIELD_LEN:
        raise ValueError(f"source exceeds {_MAX_FIELD_LEN} chars")

    canonical_payload = _validate_payload(payload)
    now = time.time()

    if idempotency_key is not None:
        # INSERT OR IGNORE + read-back.  The partial unique index on
        # (source, idempotency_key) WHERE idempotency_key IS NOT NULL
        # handles dedup; OR IGNORE works with partial indexes where
        # ON CONFLICT(col, col) does not.
        conn.execute(
            "INSERT OR IGNORE INTO hermes_events "
            "(event_type, source, payload, created_at, idempotency_key) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_type, source, canonical_payload, now, idempotency_key),
        )
        row = conn.execute(
            "SELECT id, event_type, source, payload, created_at, "
            "idempotency_key FROM hermes_events "
            "WHERE source = ? AND idempotency_key = ?",
            (source, idempotency_key),
        ).fetchone()
    else:
        cur = conn.execute(
            "INSERT INTO hermes_events "
            "(event_type, source, payload, created_at, idempotency_key) "
            "VALUES (?, ?, ?, ?, NULL)",
            (event_type, source, canonical_payload, now),
        )
        row = conn.execute(
            "SELECT id, event_type, source, payload, created_at, "
            "idempotency_key FROM hermes_events WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()

    return Event(
        id=row[0],
        event_type=row[1],
        source=row[2],
        payload=json.loads(row[3]),
        created_at=row[4],
        idempotency_key=row[5],
    )


def subscribe(
    conn: sqlite3.Connection,
    subscriber_id: str,
    *,
    event_types: Collection[str] | None = None,
    start: Literal["latest", "beginning"] = "latest",
) -> int:
    """Register a durable subscriber with immutable event-type interests.

    ``event_types=None`` means "all events".  An explicit collection
    restricts ``poll`` to only those types; the interests are immutable
    after first registration.

    ``start='latest'`` (default) sets the initial cursor to the current
    ``MAX(id)`` in ``hermes_events``, so only future events are polled.
    ``start='beginning'`` sets the cursor to 0, enabling full replay.

    Repeated registration with the same ``subscriber_id`` returns the
    existing cursor without modification — it MUST NOT reset the cursor
    or change the interests.

    Returns the current cursor position (``last_event_id``).
    """
    if not subscriber_id or not isinstance(subscriber_id, str):
        raise ValueError("subscriber_id must be a non-empty string")
    if len(subscriber_id) > _MAX_FIELD_LEN:
        raise ValueError(f"subscriber_id exceeds {_MAX_FIELD_LEN} chars")

    # Canonical interests: sorted JSON array or null.
    if event_types is not None:
        types_list = sorted(set(event_types))
        if not types_list:
            raise ValueError("event_types must be None or a non-empty collection")
        interests_json = json.dumps(types_list)
    else:
        interests_json = "null"

    # Check for existing registration.
    existing = conn.execute(
        "SELECT last_event_id, event_types FROM hermes_event_cursors "
        "WHERE subscriber_id = ?",
        (subscriber_id,),
    ).fetchone()

    if existing is not None:
        # Return existing cursor; do NOT reset or change interests.
        return existing[0]

    # New subscriber — determine initial cursor.
    if start == "latest":
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM hermes_events"
        ).fetchone()
        initial_cursor = row[0]
    else:
        initial_cursor = 0

    now = time.time()
    conn.execute(
        "INSERT INTO hermes_event_cursors "
        "(subscriber_id, event_types, last_event_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (subscriber_id, interests_json, initial_cursor, now, now),
    )
    return initial_cursor


def poll(
    conn: sqlite3.Connection,
    subscriber_id: str,
    *,
    limit: int = 100,
) -> list[Event]:
    """Poll for new events matching the subscriber's registered interests.

    Returns events with ``id > cursor`` in ascending id order, bounded by
    ``limit`` (1..1000).  Does NOT advance the cursor — call ``ack`` after
    successful processing.

    The ``event_types`` filter comes from the subscriber's registration
    (immutable); ``poll`` does not accept a separate filter argument.
    """
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    # Fetch subscriber state.
    row = conn.execute(
        "SELECT last_event_id, event_types FROM hermes_event_cursors "
        "WHERE subscriber_id = ?",
        (subscriber_id,),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"subscriber '{subscriber_id}' is not registered; "
            f"call subscribe() first"
        )

    cursor = row[0]
    interests_raw = row[1]
    interests = json.loads(interests_raw) if interests_raw != "null" else None

    if interests is not None:
        # Filter by event_type.
        placeholders = ",".join("?" * len(interests))
        rows = conn.execute(
            f"SELECT id, event_type, source, payload, created_at, "
            f"idempotency_key FROM hermes_events "
            f"WHERE id > ? AND event_type IN ({placeholders}) "
            f"ORDER BY id ASC LIMIT ?",
            (cursor, *interests, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, event_type, source, payload, created_at, "
            "idempotency_key FROM hermes_events "
            "WHERE id > ? ORDER BY id ASC LIMIT ?",
            (cursor, limit),
        ).fetchall()

    return [
        Event(
            id=r[0],
            event_type=r[1],
            source=r[2],
            payload=json.loads(r[3]),
            created_at=r[4],
            idempotency_key=r[5],
        )
        for r in rows
    ]


def ack(
    conn: sqlite3.Connection,
    subscriber_id: str,
    event_id: int,
) -> None:
    """Advance the subscriber's cursor to ``event_id``.

    Must run under an active write transaction.  Enforces:

    1. ``event_id`` must reference an existing event.
    2. ``event_id`` must be > current cursor (monotonic advance).
    3. ``event_id`` must be the first unacked event matching the
       subscriber's interests (no gap-skipping of matching events).
    4. Duplicate ACK of the current cursor value is harmless (no-op).

    Two processes on the same subscriber_id may both observe one event
    (at-least-once); downstream must use event-id idempotency.
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "ack() requires an active write transaction; "
            "use kanban_db_connect.write_txn(conn)"
        )

    # Fetch subscriber state.
    row = conn.execute(
        "SELECT last_event_id, event_types FROM hermes_event_cursors "
        "WHERE subscriber_id = ?",
        (subscriber_id,),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"subscriber '{subscriber_id}' is not registered; "
            f"call subscribe() first"
        )

    cursor = row[0]
    interests_raw = row[1]
    interests = json.loads(interests_raw) if interests_raw != "null" else None

    # Duplicate ACK of current cursor is harmless.
    if event_id == cursor:
        return

    # Monotonic advance: event_id must be > cursor.
    if event_id <= cursor:
        raise ValueError(
            f"event_id {event_id} <= current cursor {cursor}; "
            f"ACK must advance monotonically"
        )

    # Verify event exists.
    evt_row = conn.execute(
        "SELECT id FROM hermes_events WHERE id = ?", (event_id,)
    ).fetchone()
    if evt_row is None:
        raise ValueError(f"event_id {event_id} does not exist")

    # Verify this is the FIRST unacked matching event (no gap-skipping).
    if interests is not None:
        placeholders = ",".join("?" * len(interests))
        next_row = conn.execute(
            f"SELECT id FROM hermes_events "
            f"WHERE id > ? AND event_type IN ({placeholders}) "
            f"ORDER BY id ASC LIMIT 1",
            (cursor, *interests),
        ).fetchone()
    else:
        next_row = conn.execute(
            "SELECT id FROM hermes_events "
            "WHERE id > ? ORDER BY id ASC LIMIT 1",
            (cursor,),
        ).fetchone()

    if next_row is None:
        raise ValueError(
            f"no unacked events after cursor {cursor} matching interests"
        )
    if next_row[0] != event_id:
        raise ValueError(
            f"event_id {event_id} is not the first unacked matching event "
            f"(expected {next_row[0]}); cannot skip events"
        )

    # Advance cursor.
    now = time.time()
    conn.execute(
        "UPDATE hermes_event_cursors "
        "SET last_event_id = ?, updated_at = ? "
        "WHERE subscriber_id = ? AND last_event_id = ?",
        (event_id, now, subscriber_id, cursor),
    )

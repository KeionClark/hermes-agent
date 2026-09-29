"""
Provenance-backed board goals, measurements and gap-edge events (dark).

P0b phase: goal hierarchy with live metrics stored in the board SQLite DB,
append-only measurements with provenance, signed-gap evaluation with
hysteresis edge detection, and a read-only Kanban adapter for board counts.

Design contract: t_2269abda / architecture §4, §§11-12.

Public API
----------
- ``migrate_goal_schema(conn) -> None``
- ``create_goal(...) -> Goal``
- ``activate_goal(conn, goal_id, approval_evidence_ref, approval_ts) -> Goal``
- ``pause_goal(conn, goal_id) -> Goal``
- ``get_goal(conn, goal_id) -> Goal | None``
- ``list_goals(conn, *, state=None) -> list[Goal]``
- ``record_measurement(conn, *, goal_id, metric_id, value, source_id,
      source_timestamp, observed_at, source_ref, payload_hash,
      idempotency_key) -> MeasurementResult``
- ``snapshot_board_counts(conn) -> BoardSnapshot``
- ``record_board_snapshot(target_conn, snapshot, goal_id) -> MeasurementResult``

Ownership: sole owner of this module, ``kanban_db_connect.py`` goal-schema
migration hook, ``tests/hermes_cli/test_goal_metrics.py``, and
``docs/goal_metrics.md``.  Does not touch event_bus.py, router, dispatcher,
publishers, or the live board.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOAL_SCHEMA_VERSION = 1
_MAX_CLOCK_SKEW_SECONDS = 300  # 5 minutes future tolerance
_MAX_TITLE_LEN = 512  # bounded free text for internal use only

GoalDirection = Literal["higher_is_better", "lower_is_better"]
GoalState = Literal["draft", "active", "paused"]
EdgeState = Literal["unknown", "healthy", "breached"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Goal:
    """Immutable goal record."""

    id: str
    parent_id: Optional[str]
    metric_id: str
    source_id: str
    direction: GoalDirection
    target_value: Optional[float]
    tolerance: float
    max_age_seconds: int
    state: GoalState
    title: Optional[str]
    approval_evidence_ref: Optional[str]
    approval_timestamp: Optional[float]
    created_at: float
    metric_version: int


@dataclass(frozen=True, slots=True)
class Measurement:
    """Immutable measurement record."""

    id: int
    goal_id: str
    metric_id: str
    value: float
    source_id: str
    source_timestamp: float
    observed_at: float
    source_ref: Optional[str]
    payload_hash: Optional[str]
    idempotency_key: str
    edge_state: EdgeState
    gap_value: Optional[float]
    created_at: float


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    """Result of a record_measurement call."""

    measurement: Measurement
    events: list[dict[str, Any]]
    is_replay: bool
    rejection_reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BoardSnapshot:
    """Read-only snapshot of board counts."""

    blocked_count: int
    non_archived_count: int
    source_timestamp: float
    board_id: str
    snapshot_id: str


# ---------------------------------------------------------------------------
# Schema & migration
# ---------------------------------------------------------------------------

_GOAL_SCHEMA_V1 = """\
CREATE TABLE IF NOT EXISTS goal_schema_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS goals (
    id                     TEXT PRIMARY KEY,
    parent_id              TEXT REFERENCES goals(id),
    metric_id              TEXT NOT NULL,
    source_id              TEXT NOT NULL,
    direction              TEXT NOT NULL CHECK (direction IN ('higher_is_better', 'lower_is_better')),
    target_value           REAL,
    tolerance              REAL NOT NULL DEFAULT 0 CHECK (tolerance >= 0),
    max_age_seconds        INTEGER NOT NULL CHECK (max_age_seconds > 0),
    state                  TEXT NOT NULL DEFAULT 'draft' CHECK (state IN ('draft', 'active', 'paused')),
    title                  TEXT,
    approval_evidence_ref  TEXT,
    approval_timestamp     REAL,
    created_at             REAL NOT NULL,
    metric_version         INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT chk_title_len CHECK (title IS NULL OR length(title) <= 512),
    CONSTRAINT chk_active_requires_approval CHECK (
        state != 'active' OR (approval_evidence_ref IS NOT NULL AND approval_timestamp IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_goals_state ON goals(state);
CREATE INDEX IF NOT EXISTS idx_goals_metric ON goals(metric_id);

CREATE TABLE IF NOT EXISTS goal_measurements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id           TEXT NOT NULL REFERENCES goals(id),
    metric_id         TEXT NOT NULL,
    value             REAL NOT NULL,
    source_id         TEXT NOT NULL,
    source_timestamp  REAL NOT NULL,
    observed_at       REAL NOT NULL,
    source_ref        TEXT,
    payload_hash      TEXT,
    idempotency_key   TEXT NOT NULL,
    edge_state        TEXT NOT NULL DEFAULT 'unknown' CHECK (edge_state IN ('unknown', 'healthy', 'breached')),
    gap_value         REAL,
    created_at        REAL NOT NULL,
    CONSTRAINT chk_unique_idem UNIQUE (goal_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_measurements_goal_ts
    ON goal_measurements(goal_id, source_timestamp);
CREATE INDEX IF NOT EXISTS idx_measurements_goal_id
    ON goal_measurements(goal_id, id);
"""


def migrate_goal_schema(conn: sqlite3.Connection) -> None:
    """Idempotent migration to the current goal-metrics schema version.

    Called from ``kanban_db_connect.connect`` after the core schema and
    P0a bus migration.  Uses ``goal_schema_version`` (singleton integer) —
    independent of both ``PRAGMA user_version`` and
    ``hermes_event_schema_version``.

    Wraps DDL in an explicit ``BEGIN IMMEDIATE`` / ``COMMIT`` pair for
    atomicity under the canonical ``isolation_level=None`` connection.
    """
    # Probe version outside write txn (read-only).
    try:
        row = conn.execute(
            "SELECT version FROM goal_schema_version WHERE id = 1"
        ).fetchone()
        current = row[0] if row else 0
    except sqlite3.OperationalError:
        current = 0

    if current > _GOAL_SCHEMA_VERSION:
        raise RuntimeError(
            f"goal_schema_version {current} > {_GOAL_SCHEMA_VERSION}; "
            f"this binary is too old to read this goal-metrics schema"
        )

    if current >= _GOAL_SCHEMA_VERSION:
        return  # Already at target version.

    # --- Atomic write block ---
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Re-check under write lock.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS goal_schema_version "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
        )
        row = conn.execute(
            "SELECT version FROM goal_schema_version WHERE id = 1"
        ).fetchone()
        current = row[0] if row else 0

        if current >= _GOAL_SCHEMA_VERSION:
            conn.execute("COMMIT")
            return

        if current < 1:
            for stmt in _GOAL_SCHEMA_V1.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

            conn.execute(
                "INSERT OR REPLACE INTO goal_schema_version "
                "(id, version) VALUES (1, ?)",
                (_GOAL_SCHEMA_VERSION,),
            )

        conn.execute("COMMIT")
    except BaseException:
        import contextlib
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Goal CRUD
# ---------------------------------------------------------------------------


def create_goal(
    conn: sqlite3.Connection,
    *,
    goal_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    metric_id: str,
    source_id: str,
    direction: GoalDirection = "higher_is_better",
    target_value: Optional[float] = None,
    tolerance: float = 0.0,
    max_age_seconds: int = 86400,
    title: Optional[str] = None,
) -> Goal:
    """Create a new goal in 'draft' state.

    Must be called under an active write transaction.
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "create_goal() requires an active write transaction"
        )

    if goal_id is None:
        goal_id = str(uuid.uuid4())

    # Validate inputs.
    if not metric_id or not isinstance(metric_id, str):
        raise ValueError("metric_id must be a non-empty string")
    if not source_id or not isinstance(source_id, str):
        raise ValueError("source_id must be a non-empty string")
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(f"invalid direction: {direction}")
    if target_value is not None and (
        not isinstance(target_value, (int, float))
        or math.isnan(target_value)
        or math.isinf(target_value)
    ):
        raise ValueError("target_value must be a finite number or None")
    if not isinstance(tolerance, (int, float)) or tolerance < 0 or math.isnan(tolerance) or math.isinf(tolerance):
        raise ValueError("tolerance must be a non-negative finite number")
    if not isinstance(max_age_seconds, int) or max_age_seconds <= 0:
        raise ValueError("max_age_seconds must be a positive integer")
    if title is not None and len(title) > _MAX_TITLE_LEN:
        raise ValueError(f"title exceeds {_MAX_TITLE_LEN} chars")

    # Validate parent exists.
    if parent_id is not None:
        parent = conn.execute(
            "SELECT id FROM goals WHERE id = ?", (parent_id,)
        ).fetchone()
        if parent is None:
            raise ValueError(f"parent goal '{parent_id}' does not exist")

    now = time.time()

    conn.execute(
        "INSERT INTO goals "
        "(id, parent_id, metric_id, source_id, direction, target_value, "
        "tolerance, max_age_seconds, state, title, "
        "approval_evidence_ref, approval_timestamp, created_at, metric_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, NULL, NULL, ?, 1)",
        (goal_id, parent_id, metric_id, source_id, direction,
         target_value, tolerance, max_age_seconds, title, now),
    )

    return Goal(
        id=goal_id,
        parent_id=parent_id,
        metric_id=metric_id,
        source_id=source_id,
        direction=direction,
        target_value=target_value,
        tolerance=tolerance,
        max_age_seconds=max_age_seconds,
        state="draft",
        title=title,
        approval_evidence_ref=None,
        approval_timestamp=None,
        created_at=now,
        metric_version=1,
    )


def activate_goal(
    conn: sqlite3.Connection,
    goal_id: str,
    approval_evidence_ref: str,
    approval_timestamp: float,
) -> Goal:
    """Activate a goal with explicit approval evidence.

    Must be called under an active write transaction.
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "activate_goal() requires an active write transaction"
        )
    if not approval_evidence_ref:
        raise ValueError("approval_evidence_ref is required for activation")
    if not isinstance(approval_timestamp, (int, float)):
        raise ValueError("approval_timestamp must be numeric")

    row = conn.execute(
        "SELECT * FROM goals WHERE id = ?", (goal_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"goal '{goal_id}' does not exist")

    conn.execute(
        "UPDATE goals SET state = 'active', "
        "approval_evidence_ref = ?, approval_timestamp = ? "
        "WHERE id = ?",
        (approval_evidence_ref, approval_timestamp, goal_id),
    )

    return _row_to_goal(conn.execute(
        "SELECT * FROM goals WHERE id = ?", (goal_id,)
    ).fetchone())


def pause_goal(conn: sqlite3.Connection, goal_id: str) -> Goal:
    """Pause a goal.

    Must be called under an active write transaction.
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError("pause_goal() requires an active write transaction")

    row = conn.execute(
        "SELECT * FROM goals WHERE id = ?", (goal_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"goal '{goal_id}' does not exist")

    conn.execute(
        "UPDATE goals SET state = 'paused' WHERE id = ?", (goal_id,)
    )

    return _row_to_goal(conn.execute(
        "SELECT * FROM goals WHERE id = ?", (goal_id,)
    ).fetchone())


def get_goal(conn: sqlite3.Connection, goal_id: str) -> Optional[Goal]:
    """Read a goal by ID. Returns None if not found."""
    row = conn.execute(
        "SELECT * FROM goals WHERE id = ?", (goal_id,)
    ).fetchone()
    return _row_to_goal(row) if row else None


def list_goals(
    conn: sqlite3.Connection,
    *,
    state: Optional[GoalState] = None,
) -> list[Goal]:
    """List goals, optionally filtered by state."""
    if state is not None:
        rows = conn.execute(
            "SELECT * FROM goals WHERE state = ? ORDER BY created_at",
            (state,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM goals ORDER BY created_at"
        ).fetchall()
    return [_row_to_goal(r) for r in rows]


def _row_to_goal(row: sqlite3.Row) -> Goal:
    return Goal(
        id=row["id"],
        parent_id=row["parent_id"],
        metric_id=row["metric_id"],
        source_id=row["source_id"],
        direction=row["direction"],
        target_value=row["target_value"],
        tolerance=row["tolerance"],
        max_age_seconds=row["max_age_seconds"],
        state=row["state"],
        title=row["title"],
        approval_evidence_ref=row["approval_evidence_ref"],
        approval_timestamp=row["approval_timestamp"],
        created_at=row["created_at"],
        metric_version=row["metric_version"],
    )


# ---------------------------------------------------------------------------
# Measurement recording with gap evaluation
# ---------------------------------------------------------------------------


def _compute_gap(
    value: float,
    target: float,
    direction: GoalDirection,
) -> float:
    """Compute signed gap: positive means target not met.

    higher_is_better: gap = target - value  (positive when value < target)
    lower_is_better:  gap = value - target  (positive when value > target)
    """
    if direction == "higher_is_better":
        return target - value
    else:
        return value - target


def _evaluate_edge(
    gap: float,
    tolerance: float,
    prev_edge: EdgeState,
) -> tuple[EdgeState, float]:
    """Evaluate edge state from gap.

    Returns (new_edge_state, gap_magnitude).
    gap_magnitude = max(0, gap - tolerance).
    Within tolerance: gap <= tolerance → healthy.
    Exact boundary (gap == tolerance) is healthy.
    """
    # gap <= tolerance means healthy (inclusive boundary)
    if gap <= tolerance:
        new_state: EdgeState = "healthy"
        magnitude = 0.0
    else:
        new_state = "breached"
        magnitude = gap - tolerance

    return new_state, magnitude


def _make_event_idempotency_key(
    goal_id: str,
    measurement_id: int,
    event_type: str,
) -> str:
    """Deterministic idempotency key for edge events."""
    return f"goal_edge:{goal_id}:{measurement_id}:{event_type}"


def record_measurement(
    conn: sqlite3.Connection,
    *,
    goal_id: str,
    metric_id: str,
    value: float,
    source_id: str,
    source_timestamp: float,
    observed_at: float,
    source_ref: Optional[str] = None,
    payload_hash: Optional[str] = None,
    idempotency_key: str,
) -> MeasurementResult:
    """Record a measurement against a goal, evaluate gap, emit edge events.

    Must be called under an active write transaction (same txn used for
    event_bus.publish).

    Validation:
    - Goal must exist and be active
    - metric_id must match goal's metric_id
    - source_id must match goal's source_id
    - value must be finite
    - source_timestamp must not be in the future (bounded clock skew)
    - source_timestamp must be monotonically >= previous valid measurement
    - Stale measurements (age > max_age_seconds at observed_at) are rejected
    - Missing provenance (no source_ref and no payload_hash) is rejected

    Idempotency:
    - Same (goal_id, idempotency_key) returns prior measurement + events
    - Same key with conflicting payload_hash fails closed
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "record_measurement() requires an active write transaction"
        )

    # --- Input validation ---
    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        raise ValueError("value must be a finite number")
    if not idempotency_key or not isinstance(idempotency_key, str):
        raise ValueError("idempotency_key must be a non-empty string")
    if source_ref is None and payload_hash is None:
        raise ValueError(
            "at least one of source_ref or payload_hash is required "
            "for provenance"
        )

    # --- Goal lookup and validation ---
    goal_row = conn.execute(
        "SELECT * FROM goals WHERE id = ?", (goal_id,)
    ).fetchone()
    if goal_row is None:
        raise ValueError(f"goal '{goal_id}' does not exist")

    goal = _row_to_goal(goal_row)

    if goal.state != "active":
        raise ValueError(
            f"goal '{goal_id}' is in state '{goal.state}', not 'active'"
        )
    if metric_id != goal.metric_id:
        raise ValueError(
            f"metric_id '{metric_id}' does not match goal's "
            f"metric_id '{goal.metric_id}'"
        )
    if source_id != goal.source_id:
        raise ValueError(
            f"source_id '{source_id}' does not match goal's "
            f"source_id '{goal.source_id}'"
        )

    # --- Timestamp validation ---
    now = time.time()
    if source_timestamp > now + _MAX_CLOCK_SKEW_SECONDS:
        raise ValueError(
            f"source_timestamp {source_timestamp} is more than "
            f"{_MAX_CLOCK_SKEW_SECONDS}s in the future"
        )

    # --- Staleness check ---
    age_at_observation = observed_at - source_timestamp
    if age_at_observation > goal.max_age_seconds:
        raise ValueError(
            f"measurement is stale: age {age_at_observation:.0f}s exceeds "
            f"goal max_age_seconds {goal.max_age_seconds}"
        )

    # --- Idempotency check ---
    existing = conn.execute(
        "SELECT * FROM goal_measurements "
        "WHERE goal_id = ? AND idempotency_key = ?",
        (goal_id, idempotency_key),
    ).fetchone()

    if existing is not None:
        # Replay: same key. Check for conflicting payload.
        if payload_hash is not None and existing["payload_hash"] is not None:
            if payload_hash != existing["payload_hash"]:
                raise ValueError(
                    f"idempotency_key '{idempotency_key}' already exists "
                    f"with a different payload_hash; conflicting replay "
                    f"fails closed"
                )

        measurement = _row_to_measurement(existing)

        # Retrieve previously emitted events for this measurement.
        events = _get_events_for_measurement(conn, goal_id, measurement.id)

        return MeasurementResult(
            measurement=measurement,
            events=events,
            is_replay=True,
        )

    # --- Monotonic source_timestamp check ---
    prev_row = conn.execute(
        "SELECT * FROM goal_measurements "
        "WHERE goal_id = ? ORDER BY source_timestamp DESC, id DESC LIMIT 1",
        (goal_id,),
    ).fetchone()

    if prev_row is not None:
        if source_timestamp < prev_row["source_timestamp"]:
            raise ValueError(
                f"source_timestamp {source_timestamp} is earlier than "
                f"previous measurement's source_timestamp "
                f"{prev_row['source_timestamp']}; monotonic order required"
            )

    # --- Gap evaluation ---
    events: list[dict[str, Any]] = []
    edge_state: EdgeState = "unknown"
    gap_value: Optional[float] = None

    if goal.target_value is not None:
        gap = _compute_gap(value, goal.target_value, goal.direction)
        prev_edge: EdgeState = "unknown"

        # Get previous VALID edge state (from accepted measurements only).
        if prev_row is not None:
            prev_edge = prev_row["edge_state"]

        new_edge, gap_magnitude = _evaluate_edge(gap, goal.tolerance, prev_edge)
        edge_state = new_edge
        gap_value = gap

        # --- Edge transition event emission ---
        # We need a placeholder measurement_id for the idempotency key.
        # We'll insert the measurement first, then emit events, then
        # update the measurement ID in events. But since events use
        # the bus, we do everything in one pass.

    # --- Insert measurement ---
    m_now = time.time()
    cur = conn.execute(
        "INSERT INTO goal_measurements "
        "(goal_id, metric_id, value, source_id, source_timestamp, "
        "observed_at, source_ref, payload_hash, idempotency_key, "
        "edge_state, gap_value, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (goal_id, metric_id, value, source_id, source_timestamp,
         observed_at, source_ref, payload_hash, idempotency_key,
         edge_state, gap_value, m_now),
    )
    assert cur.lastrowid is not None, "INSERT must return a rowid"
    measurement_id: int = cur.lastrowid

    # --- Emit edge events if target is set ---
    if goal.target_value is not None:
        prev_edge_state: EdgeState = "unknown"
        if prev_row is not None:
            prev_edge_state = prev_row["edge_state"]

        events = _emit_edge_events(
            conn,
            goal=goal,
            measurement_id=measurement_id,
            value=value,
            gap_value=gap_value,
            edge_state=edge_state,
            prev_edge=prev_edge_state,
            source_timestamp=source_timestamp,
        )

    measurement = Measurement(
        id=measurement_id,
        goal_id=goal_id,
        metric_id=metric_id,
        value=value,
        source_id=source_id,
        source_timestamp=source_timestamp,
        observed_at=observed_at,
        source_ref=source_ref,
        payload_hash=payload_hash,
        idempotency_key=idempotency_key,
        edge_state=edge_state,
        gap_value=gap_value,
        created_at=m_now,
    )

    return MeasurementResult(
        measurement=measurement,
        events=events,
        is_replay=False,
    )


def _emit_edge_events(
    conn: sqlite3.Connection,
    *,
    goal: Goal,
    measurement_id: int,
    value: float,
    gap_value: Optional[float],
    edge_state: EdgeState,
    prev_edge: EdgeState,
    source_timestamp: float,
) -> list[dict[str, Any]]:
    """Emit events for edge-state transitions.

    Transition rules:
    - First reading (prev=unknown):
      - If breached: emit goal.gap_detected AND metric.threshold_crossed
      - If healthy: no event (documented: no fabricated recovery for first healthy)
    - healthy→breached: emit goal.gap_detected AND metric.threshold_crossed
    - breached→healthy: emit goal.achieved (recovery, re-arms)
    - Same state: no event (no repeated crossings)
    """
    from hermes_cli.event_bus import publish

    events: list[dict[str, Any]] = []
    assert goal.target_value is not None

    # Compute threshold: target ± tolerance, direction consistent
    if goal.direction == "higher_is_better":
        threshold = goal.target_value - goal.tolerance
    else:
        threshold = goal.target_value + goal.tolerance

    def _publish_event(event_type: str, extra_payload: dict) -> dict:
        base_payload = {
            "goal_id": goal.id,
            "metric_id": goal.metric_id,
            "value": value,
            "threshold": threshold,
            "direction": goal.direction,
            "source_timestamp": source_timestamp,
        }
        if gap_value is not None:
            base_payload["status"] = edge_state
        payload = {**base_payload, **extra_payload}
        idem_key = _make_event_idempotency_key(
            goal.id, measurement_id, event_type
        )
        evt = publish(
            conn,
            event_type=event_type,
            source="goal_metrics",
            payload=payload,
            idempotency_key=idem_key,
        )
        return {
            "event_id": evt.id,
            "event_type": event_type,
            "idempotency_key": idem_key,
        }

    should_emit_breach = False
    should_emit_recovery = False

    if prev_edge == "unknown":
        # First reading.
        if edge_state == "breached":
            should_emit_breach = True
        # First healthy: no event (documented)
    elif prev_edge == "healthy" and edge_state == "breached":
        should_emit_breach = True
    elif prev_edge == "breached" and edge_state == "healthy":
        should_emit_recovery = True
    # Same state: no event.

    if should_emit_breach:
        events.append(_publish_event("goal.gap_detected", {}))
        events.append(_publish_event("metric.threshold_crossed", {}))

    if should_emit_recovery:
        events.append(_publish_event("goal.achieved", {}))

    return events


def _get_events_for_measurement(
    conn: sqlite3.Connection,
    goal_id: str,
    measurement_id: int,
) -> list[dict[str, Any]]:
    """Retrieve events associated with a measurement (for replay)."""
    events = []
    for event_type in ("goal.gap_detected", "metric.threshold_crossed", "goal.achieved"):
        idem_key = _make_event_idempotency_key(
            goal_id, measurement_id, event_type
        )
        row = conn.execute(
            "SELECT id, event_type, idempotency_key FROM hermes_events "
            "WHERE source = 'goal_metrics' AND idempotency_key = ?",
            (idem_key,),
        ).fetchone()
        if row is not None:
            events.append({
                "event_id": row["id"] if hasattr(row, "keys") else row[0],
                "event_type": row["event_type"] if hasattr(row, "keys") else row[1],
                "idempotency_key": row["idempotency_key"] if hasattr(row, "keys") else row[2],
            })
    return events


def _row_to_measurement(row: sqlite3.Row) -> Measurement:
    return Measurement(
        id=row["id"],
        goal_id=row["goal_id"],
        metric_id=row["metric_id"],
        value=row["value"],
        source_id=row["source_id"],
        source_timestamp=row["source_timestamp"],
        observed_at=row["observed_at"],
        source_ref=row["source_ref"],
        payload_hash=row["payload_hash"],
        idempotency_key=row["idempotency_key"],
        edge_state=row["edge_state"],
        gap_value=row["gap_value"],
        created_at=row["created_at"],
    )


# ---------------------------------------------------------------------------
# Read-only Kanban adapter
# ---------------------------------------------------------------------------


def snapshot_board_counts(conn: sqlite3.Connection) -> BoardSnapshot:
    """Take a read-only snapshot of board counts from the SAME board.

    Counts rows exactly as board_stats(conn): non-archived tasks.
    blocked_count: tasks with status='blocked' (non-archived).
    non_archived_count: all tasks where status != 'archived'.

    Atomicity: both counts are computed in a single SQL statement so they
    read from one consistent snapshot, even on an autocommit (canonical
    ``isolation_level=None``) connection under WAL-mode concurrent writes.

    Timestamp semantics: ``source_timestamp`` is taken immediately before
    the aggregate query executes.  The snapshot reflects the board state
    visible to the connection at (or fractionally after) that instant.

    Caller-owned transaction: if the caller already holds a transaction,
    the query participates in that transaction's read view — no nested
    ``BEGIN`` is issued and no ``COMMIT``/``ROLLBACK`` is emitted.

    No writes to the source board.
    """
    now = time.time()
    snapshot_id = str(uuid.uuid4())

    # Determine board identity from the connection's file path.
    db_path = None
    try:
        for _seq, name, file in conn.execute("PRAGMA database_list"):
            if name == "main":
                db_path = file
                break
    except (sqlite3.Error, TypeError, ValueError):
        pass
    board_id = db_path or "unknown"

    # Single-statement aggregate: both counts come from one read snapshot.
    # On autocommit the implicit transaction wraps exactly this statement;
    # under an existing transaction it shares the caller's read view.
    row = conn.execute(
        "SELECT"
        "  COUNT(*) AS non_archived,"
        "  SUM(CASE WHEN status = 'blocked' THEN 1 ELSE 0 END) AS blocked"
        " FROM tasks"
        " WHERE status != 'archived'"
    ).fetchone()

    non_archived_count = row[0] if row and row[0] is not None else 0
    blocked_count = row[1] if row and row[1] is not None else 0

    return BoardSnapshot(
        blocked_count=blocked_count,
        non_archived_count=non_archived_count,
        source_timestamp=now,
        board_id=board_id,
        snapshot_id=snapshot_id,
    )


def record_board_snapshot(
    target_conn: sqlite3.Connection,
    snapshot: BoardSnapshot,
    goal_id: str,
    *,
    metric_id_blocked: str = "board.blocked_card_count",
    metric_id_non_archived: str = "board.non_archived_card_count",
    source_id: str = "kanban_board_adapter",
) -> list[MeasurementResult]:
    """Record board snapshot counts as measurements against a goal.

    target_conn is the connection where measurements are recorded.
    If source board is the same DB, reads + writes are coordinated
    under the caller's existing write transaction.

    Returns a list of MeasurementResult (one per metric recorded).
    """
    if not getattr(target_conn, "in_transaction", False):
        raise RuntimeError(
            "record_board_snapshot() requires an active write transaction"
        )

    results = []

    # Look up the goal to determine which metric to record.
    goal = get_goal(target_conn, goal_id)
    if goal is None:
        raise ValueError(f"goal '{goal_id}' does not exist")

    # Determine which count to record based on the goal's metric_id.
    if goal.metric_id == metric_id_blocked:
        count_value = float(snapshot.blocked_count)
    elif goal.metric_id == metric_id_non_archived:
        count_value = float(snapshot.non_archived_count)
    else:
        raise ValueError(
            f"goal metric_id '{goal.metric_id}' is not a known board "
            f"count metric ({metric_id_blocked}, {metric_id_non_archived})"
        )

    # Build provenance hash from snapshot evidence.
    evidence = json.dumps({
        "snapshot_id": snapshot.snapshot_id,
        "board_id": snapshot.board_id,
        "blocked_count": snapshot.blocked_count,
        "non_archived_count": snapshot.non_archived_count,
        "source_timestamp": snapshot.source_timestamp,
    }, sort_keys=True)
    payload_hash = hashlib.sha256(evidence.encode()).hexdigest()[:32]

    idem_key = f"board_snapshot:{snapshot.snapshot_id}:{goal.metric_id}"

    result = record_measurement(
        target_conn,
        goal_id=goal_id,
        metric_id=goal.metric_id,
        value=count_value,
        source_id=source_id,
        source_timestamp=snapshot.source_timestamp,
        observed_at=snapshot.source_timestamp,
        source_ref=snapshot.snapshot_id,
        payload_hash=payload_hash,
        idempotency_key=idem_key,
    )
    results.append(result)

    return results

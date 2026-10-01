"""
Provenance-safe outcome ledger for evaluated action results (dark).

P2 phase: tracks outcomes of actions taken against goals, evaluates whether
measurements confirm the expected directional change, and persists the
verdict with before/after provenance.  All operations run against an
opt-in disposable board — never auto-migrated on ``connect()``.

Design contract: t_7da651df / t_fe4d8221.

Public API
----------
- ``migrate_outcome_schema(conn) -> None``
- ``create_outcome(conn, *, ...) -> Outcome``
- ``complete_action(conn, outcome_id, *, ...) -> Outcome``
- ``evaluate_due_outcome(conn, outcome_id, *, now) -> Outcome | None``
- ``revert_outcome(conn, outcome_id, *, reason, now) -> Outcome``

Ownership: sole owner of this module, ``tests/hermes_cli/test_outcome_ledger.py``,
and optionally ``docs/outcome_ledger.md``.  Does not touch goal_metrics.py,
event_bus.py, ooda_controller.py, or proposal_queue.py.

metric_version handling
~~~~~~~~~~~~~~~~~~~~~~~
``goal_measurements`` does not carry a ``metric_version`` column (see
goal_metrics.py schema lines 155-175).  We therefore cannot know an old
measurement's version after the fact.  Instead, the outcome row snapshots
the goal's ``metric_version`` at creation time (``goal_metric_version``),
and ``evaluate_due_outcome`` fails closed if the goal's live
``metric_version`` has changed since creation — the before/after comparison
is invalid when the metric definition changed under it.  Adding
``metric_version`` to ``goal_measurements`` is a provenance extension owned
by the integration layer, not this module.

Event payload
~~~~~~~~~~~~~
The event bus allowlist (event_bus.py line 56-72) does not include
``outcome_id``.  Events emitted by this module carry only the safe
allowlisted fields ``{goal_id, task_id, metric_id, status}``.
The event's ``idempotency_key`` is ``outcome_eval:<outcome_id>``, so
traceability back to the outcome row is preserved via the key, not the
payload.  Adding ``outcome_id`` to the allowlist requires explicit
coordination with the integration owner.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTCOME_SCHEMA_VERSION = 1

OutcomeStatus = Literal[
    "pending",              # action created, not yet completed
    "awaiting_remeasure",   # action completed, waiting for post-action measurement
    "confirmed",            # after measurement matches expected direction
    "falsified",            # after measurement contradicts expected direction
    "reverted",             # manually reverted
]

ExpectedDirection = Literal["higher", "lower"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeasurementSnapshot:
    """Immutable copy of provenance fields from a goal_measurements row.

    Does NOT include metric_version — that column does not exist on
    goal_measurements.  The goal's metric_version is snapshotted
    separately in the outcome row at creation time.
    """

    measurement_id: int
    value: float
    source_id: str
    source_timestamp: float
    observed_at: float
    source_ref: Optional[str]
    metric_id: str


@dataclass(frozen=True, slots=True)
class Outcome:
    """Immutable outcome record."""

    id: str
    task_id: str
    goal_id: str
    hypothesis: str
    action_type: str
    action_revision: str
    expected_direction: ExpectedDirection
    goal_metric_version: int
    before_measurement_id: Optional[int]
    before_value: Optional[float]
    before_source_id: Optional[str]
    before_source_timestamp: Optional[float]
    before_observed_at: Optional[float]
    before_source_ref: Optional[str]
    before_metric_id: Optional[str]
    after_measurement_id: Optional[int]
    after_value: Optional[float]
    after_source_id: Optional[str]
    after_source_timestamp: Optional[float]
    after_observed_at: Optional[float]
    after_source_ref: Optional[str]
    after_metric_id: Optional[str]
    settling_seconds: int
    started_at: float
    completed_at: Optional[float]
    change_summary: Optional[str]
    status: OutcomeStatus
    effect: Optional[float]
    confidence_delta: Optional[float]
    evaluated_at: Optional[float]
    reverted_at: Optional[float]
    revert_reason: Optional[str]
    idempotency_key: str
    created_at: float


# ---------------------------------------------------------------------------
# Schema & migration
# ---------------------------------------------------------------------------

_OUTCOME_SCHEMA_V1 = """\
CREATE TABLE IF NOT EXISTS outcome_schema_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    id                      TEXT PRIMARY KEY,
    task_id                 TEXT NOT NULL,
    goal_id                 TEXT NOT NULL,
    hypothesis              TEXT NOT NULL,
    action_type             TEXT NOT NULL,
    action_revision         TEXT NOT NULL,
    expected_direction      TEXT NOT NULL CHECK (expected_direction IN ('higher', 'lower')),
    goal_metric_version     INTEGER NOT NULL,
    before_measurement_id   INTEGER,
    before_value            REAL,
    before_source_id        TEXT,
    before_source_timestamp REAL,
    before_observed_at      REAL,
    before_source_ref       TEXT,
    before_metric_id        TEXT,
    after_measurement_id    INTEGER,
    after_value             REAL,
    after_source_id         TEXT,
    after_source_timestamp  REAL,
    after_observed_at       REAL,
    after_source_ref        TEXT,
    after_metric_id         TEXT,
    settling_seconds        INTEGER NOT NULL CHECK (settling_seconds >= 0),
    started_at              REAL NOT NULL,
    completed_at            REAL,
    change_summary          TEXT,
    status                  TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'awaiting_remeasure',
                                              'confirmed', 'falsified', 'reverted')),
    effect                  REAL,
    confidence_delta        REAL,
    evaluated_at            REAL,
    reverted_at             REAL,
    revert_reason           TEXT,
    idempotency_key         TEXT NOT NULL UNIQUE,
    created_at              REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcomes_goal ON outcomes(goal_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_status ON outcomes(status);
CREATE INDEX IF NOT EXISTS idx_outcomes_task ON outcomes(task_id);
"""


class OutcomeSchemaNotInitializedError(RuntimeError):
    """Raised when outcome CRUD is attempted on a board without outcome tables.

    The canonical board does not migrate outcome tables on ``connect()``;
    they are created only on fresh/disposable boards via explicit
    ``migrate_outcome_schema()``.  This error is a fail-closed guard.
    """


def _require_outcome_schema(conn: sqlite3.Connection) -> None:
    """Fail closed if the outcome schema has not been migrated."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'"
    ).fetchone()
    if row is None:
        raise OutcomeSchemaNotInitializedError(
            "Outcome tables have not been created on this board. "
            "Call migrate_outcome_schema(conn) on a fresh/disposable board "
            "before using outcome CRUD functions."
        )


def migrate_outcome_schema(conn: sqlite3.Connection) -> None:
    """Idempotent migration to the current outcome schema version.

    **Not called from ``connect()``** — callers must explicitly invoke this
    on disposable boards that opt into the outcome system.  Uses
    ``outcome_schema_version`` (singleton integer) — independent of both
    ``PRAGMA user_version`` and other schema version tables.

    Wraps DDL in an explicit ``BEGIN IMMEDIATE`` / ``COMMIT`` pair for
    atomicity under the canonical ``isolation_level=None`` connection.
    """
    # Probe version outside write txn (read-only).
    try:
        row = conn.execute(
            "SELECT version FROM outcome_schema_version WHERE id = 1"
        ).fetchone()
        current = row[0] if row else 0
    except sqlite3.OperationalError:
        current = 0

    if current > _OUTCOME_SCHEMA_VERSION:
        raise RuntimeError(
            f"outcome_schema_version {current} > {_OUTCOME_SCHEMA_VERSION}; "
            f"this binary is too old to read this outcome schema"
        )

    if current >= _OUTCOME_SCHEMA_VERSION:
        return  # Already at target version.

    # --- Atomic write block ---
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Re-check under write lock.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS outcome_schema_version "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
        )
        row = conn.execute(
            "SELECT version FROM outcome_schema_version WHERE id = 1"
        ).fetchone()
        current = row[0] if row else 0

        if current >= _OUTCOME_SCHEMA_VERSION:
            conn.execute("COMMIT")
            return

        if current < 1:
            for stmt in _OUTCOME_SCHEMA_V1.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

            conn.execute(
                "INSERT OR REPLACE INTO outcome_schema_version "
                "(id, version) VALUES (1, ?)",
                (_OUTCOME_SCHEMA_VERSION,),
            )

        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_outcome(row: sqlite3.Row) -> Outcome:
    """Convert a database row to an Outcome dataclass."""
    return Outcome(
        id=row["id"],
        task_id=row["task_id"],
        goal_id=row["goal_id"],
        hypothesis=row["hypothesis"],
        action_type=row["action_type"],
        action_revision=row["action_revision"],
        expected_direction=row["expected_direction"],
        goal_metric_version=row["goal_metric_version"],
        before_measurement_id=row["before_measurement_id"],
        before_value=row["before_value"],
        before_source_id=row["before_source_id"],
        before_source_timestamp=row["before_source_timestamp"],
        before_observed_at=row["before_observed_at"],
        before_source_ref=row["before_source_ref"],
        before_metric_id=row["before_metric_id"],
        after_measurement_id=row["after_measurement_id"],
        after_value=row["after_value"],
        after_source_id=row["after_source_id"],
        after_source_timestamp=row["after_source_timestamp"],
        after_observed_at=row["after_observed_at"],
        after_source_ref=row["after_source_ref"],
        after_metric_id=row["after_metric_id"],
        settling_seconds=row["settling_seconds"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        change_summary=row["change_summary"],
        status=row["status"],
        effect=row["effect"],
        confidence_delta=row["confidence_delta"],
        evaluated_at=row["evaluated_at"],
        reverted_at=row["reverted_at"],
        revert_reason=row["revert_reason"],
        idempotency_key=row["idempotency_key"],
        created_at=row["created_at"],
    )


def _snapshot_measurement(
    conn: sqlite3.Connection, measurement_id: int
) -> Optional[MeasurementSnapshot]:
    """Read a measurement by ID and return a snapshot of provenance fields.

    Does not read metric_version — that column does not exist on
    goal_measurements.  The goal's metric_version is captured separately
    by the caller.
    """
    row = conn.execute(
        "SELECT id, value, source_id, source_timestamp, observed_at, "
        "source_ref, metric_id FROM goal_measurements WHERE id = ?",
        (measurement_id,),
    ).fetchone()
    if row is None:
        return None
    return MeasurementSnapshot(
        measurement_id=row["id"],
        value=row["value"],
        source_id=row["source_id"],
        source_timestamp=row["source_timestamp"],
        observed_at=row["observed_at"],
        source_ref=row["source_ref"],
        metric_id=row["metric_id"],
    )


def _read_outcome(conn: sqlite3.Connection, outcome_id: str) -> Outcome:
    """Read an outcome by ID; raise ValueError if not found."""
    row = conn.execute(
        "SELECT * FROM outcomes WHERE id = ?", (outcome_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"outcome '{outcome_id}' does not exist")
    return _row_to_outcome(row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_outcome(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    goal_id: str,
    hypothesis: str,
    action_type: str,
    action_revision: str,
    expected_direction: ExpectedDirection,
    before_measurement_id: Optional[int],
    started_at: float,
    idempotency_key: str,
    settling_seconds: int = 0,
) -> Outcome:
    """Create a new outcome in 'pending' status.

    Must be called under an active write transaction.

    ``before_measurement_id`` references an existing ``goal_measurements``
    row.  Its provenance fields (value, source_id, source_timestamp,
    observed_at, source_ref, metric_id) are copied into the outcome record.
    If None (no prior measurement exists), before provenance is all NULL.

    ``before_measurement_id``'s ``source_timestamp`` must be <= ``started_at``.

    The goal's current ``metric_version`` is snapshotted into
    ``goal_metric_version`` at creation time.  ``evaluate_due_outcome``
    fails closed if the goal's live metric_version has since changed.

    Idempotency: same ``idempotency_key`` with identical payload returns the
    existing outcome.  Same key with different payload raises ValueError
    (fail closed, no silent mutation).
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "create_outcome() requires an active write transaction"
        )
    _require_outcome_schema(conn)

    # --- Input validation ---
    if not task_id or not isinstance(task_id, str):
        raise ValueError("task_id must be a non-empty string")
    if not goal_id or not isinstance(goal_id, str):
        raise ValueError("goal_id must be a non-empty string")
    if not hypothesis or not isinstance(hypothesis, str):
        raise ValueError("hypothesis must be a non-empty string")
    if not action_type or not isinstance(action_type, str):
        raise ValueError("action_type must be a non-empty string")
    if not action_revision or not isinstance(action_revision, str):
        raise ValueError("action_revision must be a non-empty string")
    if expected_direction not in ("higher", "lower"):
        raise ValueError(f"invalid expected_direction: {expected_direction}")
    if not idempotency_key or not isinstance(idempotency_key, str):
        raise ValueError("idempotency_key must be a non-empty string")
    if not isinstance(settling_seconds, int) or settling_seconds < 0:
        raise ValueError("settling_seconds must be a non-negative integer")
    if not isinstance(started_at, (int, float)):
        raise ValueError("started_at must be numeric")

    # --- Goal existence + metric_version snapshot ---
    goal_row = conn.execute(
        "SELECT id, metric_id, metric_version FROM goals WHERE id = ?",
        (goal_id,),
    ).fetchone()
    if goal_row is None:
        raise ValueError(f"goal '{goal_id}' does not exist")
    goal_metric_version: int = goal_row["metric_version"]

    # --- Before measurement snapshot ---
    before_snap: Optional[MeasurementSnapshot] = None
    if before_measurement_id is not None:
        before_snap = _snapshot_measurement(conn, before_measurement_id)
        if before_snap is None:
            raise ValueError(
                f"before_measurement_id {before_measurement_id} does not exist"
            )
        # Validate source_timestamp <= started_at
        if before_snap.source_timestamp > started_at:
            raise ValueError(
                f"before measurement source_timestamp "
                f"{before_snap.source_timestamp} > started_at {started_at}"
            )

    # --- Idempotency check ---
    existing = conn.execute(
        "SELECT * FROM outcomes WHERE idempotency_key = ?",
        (idempotency_key,),
    ).fetchone()

    if existing is not None:
        existing_outcome = _row_to_outcome(existing)
        # Check for conflicting payload.
        if (
            existing_outcome.task_id != task_id
            or existing_outcome.goal_id != goal_id
            or existing_outcome.action_type != action_type
            or existing_outcome.action_revision != action_revision
            or existing_outcome.expected_direction != expected_direction
        ):
            raise ValueError(
                f"idempotency_key '{idempotency_key}' already exists "
                f"with different payload; conflicting replay fails closed"
            )
        return existing_outcome

    # --- Insert ---
    outcome_id = str(uuid.uuid4())
    now = time.time()

    conn.execute(
        "INSERT INTO outcomes "
        "(id, task_id, goal_id, hypothesis, action_type, action_revision, "
        "expected_direction, goal_metric_version, "
        "before_measurement_id, before_value, "
        "before_source_id, before_source_timestamp, before_observed_at, "
        "before_source_ref, before_metric_id, "
        "after_measurement_id, after_value, after_source_id, "
        "after_source_timestamp, after_observed_at, after_source_ref, "
        "after_metric_id, "
        "settling_seconds, started_at, completed_at, change_summary, "
        "status, effect, confidence_delta, evaluated_at, "
        "reverted_at, revert_reason, idempotency_key, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "NULL, NULL, NULL, NULL, NULL, NULL, NULL, "
        "?, ?, NULL, NULL, 'pending', NULL, NULL, NULL, NULL, NULL, ?, ?)",
        (
            outcome_id, task_id, goal_id, hypothesis,
            action_type, action_revision, expected_direction,
            goal_metric_version,
            before_measurement_id,
            before_snap.value if before_snap else None,
            before_snap.source_id if before_snap else None,
            before_snap.source_timestamp if before_snap else None,
            before_snap.observed_at if before_snap else None,
            before_snap.source_ref if before_snap else None,
            before_snap.metric_id if before_snap else None,
            settling_seconds, started_at,
            idempotency_key, now,
        ),
    )

    return _read_outcome(conn, outcome_id)


def complete_action(
    conn: sqlite3.Connection,
    outcome_id: str,
    *,
    action_revision: str,
    completed_at: float,
    change_summary: Optional[str] = None,
) -> Outcome:
    """Mark the action as completed, transitioning to 'awaiting_remeasure'.

    Must be called under an active write transaction.

    ``action_revision`` is the final revision of the action (e.g. commit SHA).
    ``completed_at`` is the timestamp when the action finished.
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "complete_action() requires an active write transaction"
        )
    _require_outcome_schema(conn)

    outcome = _read_outcome(conn, outcome_id)

    if outcome.status != "pending":
        raise ValueError(
            f"outcome '{outcome_id}' is in status '{outcome.status}', "
            f"expected 'pending'"
        )

    if not action_revision or not isinstance(action_revision, str):
        raise ValueError("action_revision must be a non-empty string")
    if not isinstance(completed_at, (int, float)):
        raise ValueError("completed_at must be numeric")
    if completed_at < outcome.started_at:
        raise ValueError(
            f"completed_at {completed_at} < started_at {outcome.started_at}"
        )

    conn.execute(
        "UPDATE outcomes SET "
        "action_revision = ?, completed_at = ?, change_summary = ?, "
        "status = 'awaiting_remeasure' "
        "WHERE id = ?",
        (action_revision, completed_at, change_summary, outcome_id),
    )

    return _read_outcome(conn, outcome_id)


def evaluate_due_outcome(
    conn: sqlite3.Connection,
    outcome_id: str,
    *,
    now: float,
) -> Optional[Outcome]:
    """Evaluate an outcome if a valid after-measurement is available.

    Must be called under an active write transaction.

    Returns the evaluated Outcome if evaluation succeeded, or None if
    the outcome is not yet evaluable (no valid after measurement).

    Evaluation rules:
    - Outcome must be in 'awaiting_remeasure' status.
    - The goal's current metric_version must match the snapshotted
      goal_metric_version from creation time.  If it changed, fail closed
      with ValueError — the before/after comparison is invalid.
    - Find the latest measurement for the same goal where:
      - same metric_id, source_id as the goal
      - source_timestamp > completed_at
      - source_timestamp <= now
      - source_timestamp >= completed_at + settling_seconds
      - fresh: (now - source_timestamp) <= goal.max_age_seconds
    - If no valid after measurement exists: return None (pending).
    - If before is absent: fail closed -> falsified.
    - Compare before vs after values per expected_direction:
      - 'higher': after > before -> confirmed, else falsified
      - 'lower': after < before -> confirmed, else falsified
      - Equal is falsified, not unknown.
    - Persist effect (after - before) and confidence_delta only once.

    Idempotence: already-evaluated outcomes return their existing record.
    Already-reverted outcomes return their existing record.

    Note: ``goal_measurements`` does not carry metric_version.  We cannot
    know an old measurement's version after the fact.  The fail-closed
    metric_version check on the goal itself catches definition changes.
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "evaluate_due_outcome() requires an active write transaction"
        )
    _require_outcome_schema(conn)

    outcome = _read_outcome(conn, outcome_id)

    # Idempotence: already terminal.
    if outcome.status in ("confirmed", "falsified", "reverted"):
        return outcome

    if outcome.status == "pending":
        raise ValueError(
            f"outcome '{outcome_id}' is still 'pending' — "
            f"call complete_action() first"
        )

    assert outcome.status == "awaiting_remeasure"
    assert outcome.completed_at is not None

    # --- Goal lookup for matching criteria ---
    goal_row = conn.execute(
        "SELECT id, metric_id, source_id, metric_version, max_age_seconds "
        "FROM goals WHERE id = ?",
        (outcome.goal_id,),
    ).fetchone()
    if goal_row is None:
        raise ValueError(f"goal '{outcome.goal_id}' does not exist")

    goal_metric_id = goal_row["metric_id"]
    goal_source_id = goal_row["source_id"]
    goal_metric_version = goal_row["metric_version"]
    goal_max_age = goal_row["max_age_seconds"]

    # --- Fail closed on metric_version drift ---
    if goal_metric_version != outcome.goal_metric_version:
        raise ValueError(
            f"goal '{outcome.goal_id}' metric_version changed from "
            f"{outcome.goal_metric_version} to {goal_metric_version} "
            f"since outcome creation; before/after comparison is invalid. "
            f"Revert or discard this outcome."
        )

    # --- Find valid after measurement ---
    # source_timestamp > completed_at, <= now, >= completed_at + settling_seconds.
    # Fresh: (now - source_timestamp) <= max_age_seconds.
    min_source_ts = outcome.completed_at + outcome.settling_seconds
    freshness_cutoff = now - goal_max_age

    after_row = conn.execute(
        "SELECT gm.id, gm.value, gm.source_id, gm.source_timestamp, "
        "gm.observed_at, gm.source_ref, gm.metric_id "
        "FROM goal_measurements gm "
        "WHERE gm.goal_id = ? "
        "AND gm.metric_id = ? "
        "AND gm.source_id = ? "
        "AND gm.source_timestamp > ? "
        "AND gm.source_timestamp <= ? "
        "AND gm.source_timestamp >= ? "
        "ORDER BY gm.source_timestamp DESC, gm.id DESC "
        "LIMIT 1",
        (
            outcome.goal_id,
            goal_metric_id,
            goal_source_id,
            outcome.completed_at,
            now,
            max(min_source_ts, freshness_cutoff),
        ),
    ).fetchone()

    if after_row is None:
        # No valid after measurement yet — still pending.
        return None

    after_snap = MeasurementSnapshot(
        measurement_id=after_row["id"],
        value=after_row["value"],
        source_id=after_row["source_id"],
        source_timestamp=after_row["source_timestamp"],
        observed_at=after_row["observed_at"],
        source_ref=after_row["source_ref"],
        metric_id=after_row["metric_id"],
    )

    # --- Determine verdict ---
    if outcome.before_measurement_id is None:
        # Absent before: fail closed -> falsified.
        verdict: OutcomeStatus = "falsified"
        effect: Optional[float] = None
        confidence_delta: Optional[float] = None
    else:
        assert outcome.before_value is not None
        effect = after_snap.value - outcome.before_value
        confidence_delta = abs(effect)

        if outcome.expected_direction == "higher":
            verdict = "confirmed" if after_snap.value > outcome.before_value else "falsified"
        else:  # "lower"
            verdict = "confirmed" if after_snap.value < outcome.before_value else "falsified"

    # --- Persist ---
    conn.execute(
        "UPDATE outcomes SET "
        "after_measurement_id = ?, after_value = ?, "
        "after_source_id = ?, after_source_timestamp = ?, "
        "after_observed_at = ?, after_source_ref = ?, "
        "after_metric_id = ?, "
        "status = ?, effect = ?, confidence_delta = ?, "
        "evaluated_at = ? "
        "WHERE id = ?",
        (
            after_snap.measurement_id, after_snap.value,
            after_snap.source_id, after_snap.source_timestamp,
            after_snap.observed_at, after_snap.source_ref,
            after_snap.metric_id,
            verdict, effect, confidence_delta,
            now,
            outcome_id,
        ),
    )

    result = _read_outcome(conn, outcome_id)

    # --- Emit event (safe allowlisted payload only) ---
    if verdict in ("confirmed", "falsified"):
        _try_emit_outcome_event(conn, result)

    return result


def _try_emit_outcome_event(
    conn: sqlite3.Connection, outcome: Outcome
) -> None:
    """Emit outcome.evaluated via event_bus.publish with safe payload.

    Payload is restricted to the event bus allowlist (event_bus.py):
    ``{goal_id, task_id, metric_id, status}``.  No ``outcome_id`` in
    payload — the event's ``idempotency_key`` (``outcome_eval:<id>``)
    provides traceability back to the outcome row.  No patient text.

    Only emits if metric_id can be normalized.  Swallows errors silently —
    event emission must never break the evaluation path.
    """
    try:
        from hermes_cli.event_bus import publish

        # Build safe allowlisted payload: only IDs and status.
        payload: dict[str, Any] = {
            "goal_id": outcome.goal_id,
            "task_id": outcome.task_id,
            "status": outcome.status,
        }

        # metric_id: use after_metric_id if available, else before_metric_id.
        metric_id = outcome.after_metric_id or outcome.before_metric_id
        if metric_id is not None and isinstance(metric_id, str) and metric_id:
            payload["metric_id"] = metric_id
        else:
            # Cannot normalize metric_id — skip event entirely.
            return

        publish(
            conn,
            event_type="outcome.evaluated",
            source="outcome_ledger",
            payload=payload,
            idempotency_key=f"outcome_eval:{outcome.id}",
        )
    except Exception:
        # Best-effort: never let event emission break the evaluation path.
        _log.debug(
            "Failed to emit outcome.evaluated event for %s", outcome.id,
            exc_info=True,
        )


def revert_outcome(
    conn: sqlite3.Connection,
    outcome_id: str,
    *,
    reason: str,
    now: float,
) -> Outcome:
    """Revert an outcome, marking it as reverted with a reason.

    Must be called under an active write transaction.

    Can revert from any non-terminal status except 'reverted' itself.
    Idempotence: already-reverted outcomes return their existing record.
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "revert_outcome() requires an active write transaction"
        )
    _require_outcome_schema(conn)

    if not reason or not isinstance(reason, str):
        raise ValueError("reason must be a non-empty string")
    if not isinstance(now, (int, float)):
        raise ValueError("now must be numeric")

    outcome = _read_outcome(conn, outcome_id)

    # Idempotence: already reverted.
    if outcome.status == "reverted":
        return outcome

    conn.execute(
        "UPDATE outcomes SET "
        "status = 'reverted', reverted_at = ?, revert_reason = ? "
        "WHERE id = ?",
        (now, reason, outcome_id),
    )

    return _read_outcome(conn, outcome_id)

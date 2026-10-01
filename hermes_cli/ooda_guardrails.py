"""
Isolated SQLite guardrail engine for OODA Act spend/drift/trip-wire control (dark).

P3a phase: atomic four-scope spend reservation at the OODA Decide→Act boundary,
evaluated-outcome drift assessment, independent trip-wire detection, inert
rollback proposal, and audited manual reset.  All operations run against an
opt-in disposable board — never auto-migrated on ``connect()``.

Design contract: t_884be9d4 / architecture §9.

Public API
----------
- ``migrate_guardrail_schema(conn) -> None``
- ``reserve_act(conn, *, ...) -> ReservationResult``
- ``commit_act(conn, reservation_key, actual_tokens, now) -> None``
- ``release_act(conn, reservation_key, now) -> None``
- ``evaluate_drift(conn, agent_id, now, min_samples=10) -> DriftResult``
- ``evaluate_trip_wire(conn, ...) -> TripWireResult``
- ``propose_rollback(conn, trigger, outcome_ids) -> RollbackProposal``
- ``reset_guardrail(conn, scope, actor, reason, approval_ref, now) -> None``

Ownership: sole owner of this module and
``tests/hermes_cli/test_ooda_guardrails.py``.  Does not touch
outcome_ledger.py, goal_metrics.py, ooda_controller.py, or the canonical
board schema.

Dollar handling
~~~~~~~~~~~~~~~
All dollar amounts are stored as exact integer micro-USD (1 USD = 1_000_000).
Input accepts ``Decimal`` or ``str``; conversion rounds UP (ceil) to the next
micro-USD.  No floating-point dollars touch the ledger.

Transaction contract
~~~~~~~~~~~~~~~~~~~~
``reserve_act`` requires an active write transaction (``BEGIN IMMEDIATE``
already executed by the caller).  It participates in the caller's transaction
and does NOT issue its own ``BEGIN``.  ``commit_act`` and ``release_act``
manage their own transactions.
"""

from __future__ import annotations

import contextlib
import logging
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any, Literal, Mapping, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GUARDRAIL_SCHEMA_VERSION = 1
_MICRO_USD = 1_000_000  # 1 USD = 1_000_000 micro-USD

ReservationStatus = Literal["outstanding", "committed", "released"]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReservationResult:
    """Result of a reserve_act call."""

    allowed: bool
    reason: str
    reservation_id: Optional[str]
    remaining: Optional[dict[str, int]]  # scope -> remaining micro-USD or tokens


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Result of an evaluate_drift call."""

    agent_id: str
    total_evaluated: int
    wrong_direction_count: int
    drift_rate: Optional[float]  # None when insufficient evidence
    sufficient_evidence: bool
    window_start: float
    window_end: float


@dataclass(frozen=True, slots=True)
class TripWireResult:
    """Result of an evaluate_trip_wire call."""

    tripped: Optional[bool]  # True=tripped, False=safe, None=unknown
    reason: str
    primary_outcome_id: Optional[str]
    secondary_ratio: Optional[float]
    expected_min_ratio: Optional[float]
    expected_max_ratio: Optional[float]


@dataclass(frozen=True, slots=True)
class RollbackProposal:
    """Inert rollback proposal — requires explicit release/QA approval."""

    proposal_id: str
    trigger: str
    outcome_ids: list[str]
    revision: str
    evidence: str
    created_at: float
    requires_approval: bool  # always True


# ---------------------------------------------------------------------------
# Schema & migration
# ---------------------------------------------------------------------------

_GUARDRAIL_SCHEMA_V1 = """\
CREATE TABLE IF NOT EXISTS guardrail_schema_version (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS guardrail_reservations (
    id                  TEXT PRIMARY KEY,
    task_id             TEXT NOT NULL,
    agent_id            TEXT NOT NULL,
    lane_id             TEXT NOT NULL,
    reservation_key     TEXT NOT NULL UNIQUE,
    estimated_tokens    INTEGER NOT NULL CHECK (estimated_tokens > 0),
    price_micro_usd     INTEGER NOT NULL CHECK (price_micro_usd > 0),
    estimated_cost_micro_usd INTEGER NOT NULL CHECK (estimated_cost_micro_usd > 0),
    actual_tokens       INTEGER,
    actual_cost_micro_usd INTEGER,
    status              TEXT NOT NULL DEFAULT 'outstanding'
                        CHECK (status IN ('outstanding', 'committed', 'released')),
    approval_ref        TEXT,
    reserved_at         REAL NOT NULL,
    committed_at        REAL,
    released_at         REAL,
    created_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reservations_key ON guardrail_reservations(reservation_key);
CREATE INDEX IF NOT EXISTS idx_reservations_task ON guardrail_reservations(task_id);
CREATE INDEX IF NOT EXISTS idx_reservations_agent ON guardrail_reservations(agent_id);
CREATE INDEX IF NOT EXISTS idx_reservations_lane ON guardrail_reservations(lane_id);
CREATE INDEX IF NOT EXISTS idx_reservations_status ON guardrail_reservations(status);

CREATE TABLE IF NOT EXISTS guardrail_halts (
    id          TEXT PRIMARY KEY,
    scope       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    halted_at   REAL NOT NULL,
    resolved_at REAL,
    resolved_by TEXT,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_halts_scope ON guardrail_halts(scope);

CREATE TABLE IF NOT EXISTS guardrail_resets (
    id            TEXT PRIMARY KEY,
    scope         TEXT NOT NULL,
    actor         TEXT NOT NULL,
    reason        TEXT NOT NULL,
    approval_ref  TEXT NOT NULL,
    reset_at      REAL NOT NULL,
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_resets_scope ON guardrail_resets(scope);
"""


class GuardrailSchemaNotInitializedError(RuntimeError):
    """Raised when guardrail operations are attempted without schema."""


def _require_guardrail_schema(conn: sqlite3.Connection) -> None:
    """Fail closed if the guardrail schema has not been migrated."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='guardrail_reservations'"
    ).fetchone()
    if row is None:
        raise GuardrailSchemaNotInitializedError(
            "Guardrail tables have not been created on this board. "
            "Call migrate_guardrail_schema(conn) on a fresh/disposable board "
            "before using guardrail functions."
        )


def migrate_guardrail_schema(conn: sqlite3.Connection) -> None:
    """Idempotent migration to the current guardrail schema version.

    **Not called from ``connect()``** — callers must explicitly invoke this
    on disposable boards that opt into the guardrail system.  Uses
    ``guardrail_schema_version`` (singleton integer) — independent of both
    ``PRAGMA user_version`` and other schema version tables.

    Wraps DDL in an explicit ``BEGIN IMMEDIATE`` / ``COMMIT`` pair for
    atomicity under the canonical ``isolation_level=None`` connection.
    """
    # Probe version outside write txn (read-only).
    try:
        row = conn.execute(
            "SELECT version FROM guardrail_schema_version WHERE id = 1"
        ).fetchone()
        current = row[0] if row else 0
    except sqlite3.OperationalError:
        current = 0

    if current > _GUARDRAIL_SCHEMA_VERSION:
        raise RuntimeError(
            f"guardrail_schema_version {current} > {_GUARDRAIL_SCHEMA_VERSION}; "
            f"this binary is too old to read this guardrail schema"
        )

    if current >= _GUARDRAIL_SCHEMA_VERSION:
        return  # Already at target version.

    # --- Atomic write block ---
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Re-check under write lock.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS guardrail_schema_version "
            "(id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)"
        )
        row = conn.execute(
            "SELECT version FROM guardrail_schema_version WHERE id = 1"
        ).fetchone()
        current = row[0] if row else 0

        if current >= _GUARDRAIL_SCHEMA_VERSION:
            conn.execute("COMMIT")
            return

        if current < 1:
            for stmt in _GUARDRAIL_SCHEMA_V1.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

            conn.execute(
                "INSERT OR REPLACE INTO guardrail_schema_version "
                "(id, version) VALUES (1, ?)",
                (1,),
            )

        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_micro_usd(value: Decimal | str) -> int:
    """Convert a Decimal or string dollar amount to integer micro-USD.

    Rounds UP (ceiling) to the next micro-USD.  Rejects non-positive,
    non-finite, and invalid inputs.
    """
    if isinstance(value, str):
        try:
            value = Decimal(value)
        except InvalidOperation:
            raise ValueError(f"invalid dollar amount: {value!r}")

    if not isinstance(value, Decimal):
        raise TypeError(f"expected Decimal or str, got {type(value).__name__}")

    if not value.is_finite():
        raise ValueError(f"dollar amount must be finite, got {value}")
    if value <= 0:
        raise ValueError(f"dollar amount must be positive, got {value}")

    micro = value * _MICRO_USD
    # Round UP (ceiling) to integer micro-USD
    micro_int = int(micro.to_integral_value(rounding=ROUND_CEILING))
    if micro_int <= 0:
        raise ValueError(f"dollar amount too small: {value}")
    return micro_int


def _estimate_cost_micro_usd(
    estimated_tokens: int, price_micro_usd: int
) -> int:
    """Compute estimated cost in micro-USD, rounding UP.

    cost = ceil(estimated_tokens * price_micro_usd / 1_000_000)
    Uses Decimal to avoid floating-point error.
    """
    d_tokens = Decimal(estimated_tokens)
    d_price = Decimal(price_micro_usd)
    d_million = Decimal(_MICRO_USD)
    raw = d_tokens * d_price / d_million
    result = int(raw.to_integral_value(rounding=ROUND_CEILING))
    return max(result, 1)  # at least 1 micro-USD


def _utc_day_start(ts: float) -> float:
    """Return the UTC midnight (start of day) for a timestamp."""
    import calendar
    import time as _time

    gm = _time.gmtime(ts)
    return float(calendar.timegm((gm.tm_year, gm.tm_mon, gm.tm_mday, 0, 0, 0, 0, 0, 0)))


def _utc_week_start(ts: float) -> float:
    """Return the UTC Monday 00:00:00 for the week containing the timestamp."""
    import calendar
    import time as _time

    gm = _time.gmtime(ts)
    # Monday is weekday 0
    days_since_monday = gm.tm_wday
    monday = calendar.timegm((gm.tm_year, gm.tm_mon, gm.tm_mday, 0, 0, 0, 0, 0, 0))
    monday -= days_since_monday * 86400
    return float(monday)


def _check_active_halt(conn: sqlite3.Connection, scope: str) -> Optional[str]:
    """Check if there is an active (unresolved) halt for the given scope.

    Returns the halt reason if halted, None otherwise.
    """
    row = conn.execute(
        "SELECT reason FROM guardrail_halts "
        "WHERE scope = ? AND resolved_at IS NULL "
        "ORDER BY halted_at DESC LIMIT 1",
        (scope,),
    ).fetchone()
    return row[0] if row else None


def _record_halt(conn: sqlite3.Connection, scope: str, reason: str, now: float) -> str:
    """Record a new halt (idempotent for same scope while unresolved)."""
    # Check if already halted for this scope
    existing = conn.execute(
        "SELECT id FROM guardrail_halts "
        "WHERE scope = ? AND resolved_at IS NULL",
        (scope,),
    ).fetchone()
    if existing:
        return existing[0]

    halt_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO guardrail_halts (id, scope, reason, halted_at, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (halt_id, scope, reason, now, now),
    )
    return halt_id


# ---------------------------------------------------------------------------
# Public API: reserve_act
# ---------------------------------------------------------------------------


def reserve_act(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    agent_id: str,
    lane_id: str,
    reservation_key: str,
    estimated_tokens: int,
    price_per_million_usd: Decimal | str,
    caps: Mapping[str, Decimal | str],
    now: float,
    approval_ref: str | None = None,
) -> ReservationResult:
    """Reserve spend budget before an OODA Act.

    Must be called under an active write transaction (caller owns BEGIN
    IMMEDIATE).  Participates in the caller's transaction — does NOT
    issue its own BEGIN.

    Four cap scopes checked atomically:
    - ``task_token``: max tokens for this task
    - ``agent_day``: max micro-USD for this agent today (UTC day)
    - ``lane_week``: max micro-USD for this lane this week (UTC week)
    - ``global_day``: max micro-USD across all agents today (UTC day)

    Each cap is optional; only provided caps are enforced.  Missing caps
    in the ``caps`` mapping are skipped, but at least one must be present
    when the function is called (fail closed on empty caps).

    Outstanding (not yet committed/released) reservations count toward
    caps alongside settled (committed) spend.

    Returns a ReservationResult with allowed/denied, reason, and remaining
    budget by scope.
    """
    if not getattr(conn, "in_transaction", False):
        raise RuntimeError(
            "reserve_act() requires an active write transaction"
        )
    _require_guardrail_schema(conn)

    # --- Input validation (fail closed) ---
    if not task_id or not isinstance(task_id, str):
        raise ValueError("task_id must be a non-empty string")
    if not agent_id or not isinstance(agent_id, str):
        raise ValueError("agent_id must be a non-empty string")
    if not lane_id or not isinstance(lane_id, str):
        raise ValueError("lane_id must be a non-empty string")
    if not reservation_key or not isinstance(reservation_key, str):
        raise ValueError("reservation_key must be a non-empty string")
    if not isinstance(estimated_tokens, int) or estimated_tokens <= 0:
        raise ValueError("estimated_tokens must be a positive integer")
    if not isinstance(now, (int, float)) or math.isnan(now) or math.isinf(now):
        raise ValueError("now must be a finite number")

    # Validate price
    price_micro = _to_micro_usd(price_per_million_usd)

    # Validate caps: must be a mapping, at least one valid scope
    if not isinstance(caps, Mapping):
        raise TypeError("caps must be a Mapping")

    valid_scopes = {"task_token", "agent_day", "lane_week", "global_day"}
    parsed_caps: dict[str, int] = {}
    for scope_name, cap_value in caps.items():
        if scope_name not in valid_scopes:
            raise ValueError(f"unknown cap scope: {scope_name}")
        if scope_name == "task_token":
            # task_token cap is in tokens, not USD
            if isinstance(cap_value, (Decimal, str)):
                try:
                    d = Decimal(str(cap_value))
                except InvalidOperation:
                    raise ValueError(f"invalid task_token cap: {cap_value!r}")
                if not d.is_finite() or d <= 0:
                    raise ValueError(f"task_token cap must be positive finite, got {cap_value}")
                parsed_caps[scope_name] = int(d.to_integral_value(rounding=ROUND_CEILING))
            elif isinstance(cap_value, int) and cap_value > 0:
                parsed_caps[scope_name] = cap_value
            else:
                raise ValueError(f"task_token cap must be a positive integer or Decimal, got {cap_value!r}")
        else:
            # USD scopes: convert to micro-USD
            parsed_caps[scope_name] = _to_micro_usd(cap_value)

    if not parsed_caps:
        raise ValueError("caps must contain at least one valid scope")

    # --- Check for active halts across all relevant scopes ---
    halt_scopes = [
        f"task:{task_id}",
        f"agent:{agent_id}",
        f"lane:{lane_id}",
        "global",
    ]
    for hs in halt_scopes:
        halt_reason = _check_active_halt(conn, hs)
        if halt_reason is not None:
            return ReservationResult(
                allowed=False,
                reason=f"safe halt active on {hs}: {halt_reason}",
                reservation_id=None,
                remaining=None,
            )

    # --- Idempotency / replay check ---
    existing = conn.execute(
        "SELECT * FROM guardrail_reservations WHERE reservation_key = ?",
        (reservation_key,),
    ).fetchone()

    if existing is not None:
        # Same key exists — check for conflicting payload
        if (
            existing["task_id"] != task_id
            or existing["agent_id"] != agent_id
            or existing["lane_id"] != lane_id
        ):
            raise ValueError(
                f"reservation_key '{reservation_key}' already exists "
                f"with different payload; conflicting replay fails closed"
            )
        # Idempotent replay: return the original result
        return ReservationResult(
            allowed=existing["status"] != "released",
            reason="replay" if existing["status"] != "released" else "replayed released reservation",
            reservation_id=existing["id"],
            remaining=None,
        )

    # --- Compute estimated cost ---
    estimated_cost = _estimate_cost_micro_usd(estimated_tokens, price_micro)

    # --- Check each cap scope ---
    remaining: dict[str, int] = {}
    day_start = _utc_day_start(now)
    day_end = day_start + 86400.0
    week_start = _utc_week_start(now)
    week_end = week_start + 7 * 86400.0

    # task_token: total tokens (outstanding + committed) for this task
    if "task_token" in parsed_caps:
        row = conn.execute(
            "SELECT COALESCE(SUM("
            "  CASE WHEN status = 'committed' THEN actual_tokens "
            "  ELSE estimated_tokens END"
            "), 0) AS total "
            "FROM guardrail_reservations "
            "WHERE task_id = ? AND status IN ('outstanding', 'committed')",
            (task_id,),
        ).fetchone()
        used_tokens = row[0]
        cap_tokens = parsed_caps["task_token"]
        remaining_tokens = cap_tokens - used_tokens
        remaining["task_token"] = remaining_tokens
        if used_tokens + estimated_tokens > cap_tokens:
            return ReservationResult(
                allowed=False,
                reason=f"task_token cap exceeded: {used_tokens}+{estimated_tokens} > {cap_tokens}",
                reservation_id=None,
                remaining=remaining,
            )

    # agent_day: total micro-USD for this agent today
    if "agent_day" in parsed_caps:
        row = conn.execute(
            "SELECT COALESCE(SUM("
            "  CASE WHEN status = 'committed' THEN actual_cost_micro_usd "
            "  ELSE estimated_cost_micro_usd END"
            "), 0) AS total "
            "FROM guardrail_reservations "
            "WHERE agent_id = ? AND status IN ('outstanding', 'committed') "
            "AND reserved_at >= ? AND reserved_at < ?",
            (agent_id, day_start, day_end),
        ).fetchone()
        used_usd = row[0]
        cap_usd = parsed_caps["agent_day"]
        remaining_usd = cap_usd - used_usd
        remaining["agent_day"] = remaining_usd
        if used_usd + estimated_cost > cap_usd:
            # Record halt on cap breach
            _record_halt(
                conn, f"agent:{agent_id}",
                f"agent_day cap exceeded: used={used_usd}, est={estimated_cost}, cap={cap_usd}",
                now,
            )
            return ReservationResult(
                allowed=False,
                reason=f"agent_day cap exceeded: {used_usd}+{estimated_cost} > {cap_usd}",
                reservation_id=None,
                remaining=remaining,
            )

    # lane_week: total micro-USD for this lane this week
    if "lane_week" in parsed_caps:
        row = conn.execute(
            "SELECT COALESCE(SUM("
            "  CASE WHEN status = 'committed' THEN actual_cost_micro_usd "
            "  ELSE estimated_cost_micro_usd END"
            "), 0) AS total "
            "FROM guardrail_reservations "
            "WHERE lane_id = ? AND status IN ('outstanding', 'committed') "
            "AND reserved_at >= ? AND reserved_at < ?",
            (lane_id, week_start, week_end),
        ).fetchone()
        used_usd = row[0]
        cap_usd = parsed_caps["lane_week"]
        remaining_usd = cap_usd - used_usd
        remaining["lane_week"] = remaining_usd
        if used_usd + estimated_cost > cap_usd:
            _record_halt(
                conn, f"lane:{lane_id}",
                f"lane_week cap exceeded: used={used_usd}, est={estimated_cost}, cap={cap_usd}",
                now,
            )
            return ReservationResult(
                allowed=False,
                reason=f"lane_week cap exceeded: {used_usd}+{estimated_cost} > {cap_usd}",
                reservation_id=None,
                remaining=remaining,
            )

    # global_day: total micro-USD across all agents today
    if "global_day" in parsed_caps:
        row = conn.execute(
            "SELECT COALESCE(SUM("
            "  CASE WHEN status = 'committed' THEN actual_cost_micro_usd "
            "  ELSE estimated_cost_micro_usd END"
            "), 0) AS total "
            "FROM guardrail_reservations "
            "WHERE status IN ('outstanding', 'committed') "
            "AND reserved_at >= ? AND reserved_at < ?",
            (day_start, day_end),
        ).fetchone()
        used_usd = row[0]
        cap_usd = parsed_caps["global_day"]
        remaining_usd = cap_usd - used_usd
        remaining["global_day"] = remaining_usd
        if used_usd + estimated_cost > cap_usd:
            _record_halt(
                conn, "global",
                f"global_day cap exceeded: used={used_usd}, est={estimated_cost}, cap={cap_usd}",
                now,
            )
            return ReservationResult(
                allowed=False,
                reason=f"global_day cap exceeded: {used_usd}+{estimated_cost} > {cap_usd}",
                reservation_id=None,
                remaining=remaining,
            )

    # --- All caps pass — insert reservation ---
    reservation_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO guardrail_reservations "
        "(id, task_id, agent_id, lane_id, reservation_key, "
        "estimated_tokens, price_micro_usd, estimated_cost_micro_usd, "
        "status, approval_ref, reserved_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'outstanding', ?, ?, ?)",
        (
            reservation_id, task_id, agent_id, lane_id, reservation_key,
            estimated_tokens, price_micro, estimated_cost,
            approval_ref, now, now,
        ),
    )

    return ReservationResult(
        allowed=True,
        reason="reserved",
        reservation_id=reservation_id,
        remaining=remaining,
    )


# ---------------------------------------------------------------------------
# Public API: commit_act / release_act
# ---------------------------------------------------------------------------


def commit_act(
    conn: sqlite3.Connection,
    reservation_key: str,
    actual_tokens: int,
    now: float,
) -> None:
    """Commit a reservation with actual token usage.

    Manages its own transaction.  Idempotent: committing an already-committed
    reservation is a no-op.  Committing a released reservation raises
    ValueError (irreversible: the cost was already released).

    ``actual_tokens`` must be a positive integer.  The actual cost is computed
    from the reservation's price_micro_usd.
    """
    _require_guardrail_schema(conn)

    if not reservation_key or not isinstance(reservation_key, str):
        raise ValueError("reservation_key must be a non-empty string")
    if not isinstance(actual_tokens, int) or actual_tokens <= 0:
        raise ValueError("actual_tokens must be a positive integer")
    if not isinstance(now, (int, float)) or math.isnan(now) or math.isinf(now):
        raise ValueError("now must be a finite number")

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM guardrail_reservations WHERE reservation_key = ?",
            (reservation_key,),
        ).fetchone()

        if row is None:
            conn.execute("ROLLBACK")
            raise ValueError(f"reservation_key '{reservation_key}' not found")

        status = row["status"]

        if status == "committed":
            # Idempotent: already committed
            conn.execute("COMMIT")
            return

        if status == "released":
            conn.execute("ROLLBACK")
            raise ValueError(
                f"reservation_key '{reservation_key}' was already released; "
                f"cannot commit a released reservation (irreversible)"
            )

        assert status == "outstanding"

        # Compute actual cost
        price_micro = row["price_micro_usd"]
        actual_cost = _estimate_cost_micro_usd(actual_tokens, price_micro)

        conn.execute(
            "UPDATE guardrail_reservations SET "
            "actual_tokens = ?, actual_cost_micro_usd = ?, "
            "status = 'committed', committed_at = ? "
            "WHERE reservation_key = ?",
            (actual_tokens, actual_cost, now, reservation_key),
        )
        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise


def release_act(
    conn: sqlite3.Connection,
    reservation_key: str,
    now: float,
) -> None:
    """Release a reservation (action not taken / rolled back).

    Manages its own transaction.  Idempotent: releasing an already-released
    reservation is a no-op.  Releasing a committed reservation raises
    ValueError (irreversible: the cost was already charged).
    """
    _require_guardrail_schema(conn)

    if not reservation_key or not isinstance(reservation_key, str):
        raise ValueError("reservation_key must be a non-empty string")
    if not isinstance(now, (int, float)) or math.isnan(now) or math.isinf(now):
        raise ValueError("now must be a finite number")

    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT * FROM guardrail_reservations WHERE reservation_key = ?",
            (reservation_key,),
        ).fetchone()

        if row is None:
            conn.execute("ROLLBACK")
            raise ValueError(f"reservation_key '{reservation_key}' not found")

        status = row["status"]

        if status == "released":
            # Idempotent: already released
            conn.execute("COMMIT")
            return

        if status == "committed":
            conn.execute("ROLLBACK")
            raise ValueError(
                f"reservation_key '{reservation_key}' was already committed; "
                f"cannot release a committed reservation (irreversible cost)"
            )

        assert status == "outstanding"

        conn.execute(
            "UPDATE guardrail_reservations SET "
            "status = 'released', released_at = ? "
            "WHERE reservation_key = ?",
            (now, reservation_key),
        )
        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Public API: evaluate_drift
# ---------------------------------------------------------------------------


def evaluate_drift(
    conn: sqlite3.Connection,
    agent_id: str,
    now: float,
    min_samples: int = 10,
) -> DriftResult:
    """Read-only 7-day rolling drift assessment from evaluated outcomes.

    Counts outcomes with status in ('confirmed', 'falsified') where
    ``evaluated_at`` is finite and within the 7-day window.  Does NOT
    count 'pending', 'awaiting_remeasure', or 'reverted' outcomes.

    Wrong-direction: signed ``effect`` contradicts ``expected_direction``:
    - expected_direction='higher' and effect <= 0
    - expected_direction='lower' and effect >= 0

    Returns ``sufficient_evidence=False`` and ``drift_rate=None`` when
    the denominator (total evaluated) < ``min_samples``.  Never claims
    zero risk on low sample.
    """
    if not agent_id or not isinstance(agent_id, str):
        raise ValueError("agent_id must be a non-empty string")
    if not isinstance(now, (int, float)) or math.isnan(now) or math.isinf(now):
        raise ValueError("now must be a finite number")
    if not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")

    # Check outcome schema exists — fail closed
    outcome_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'"
    ).fetchone()
    if outcome_exists is None:
        raise GuardrailSchemaNotInitializedError(
            "Outcome tables not present; cannot evaluate drift"
        )

    window_start = now - 7 * 86400.0
    window_end = now

    # Read only evaluated outcomes (confirmed/falsified) in the window.
    # We need to join with task_id to filter by agent.  However, outcomes
    # don't have agent_id — we use the guardrail_reservations table to
    # map task_id → agent_id, OR we accept agent_id is carried in the
    # outcomes table indirectly.  Per the spec, agent_id is passed to us
    # as a filter.  We search outcomes via task_id → reservations link,
    # but the simpler approach: the caller identifies agent outcomes.
    #
    # Actually, outcomes don't carry agent_id.  We need to look up which
    # tasks belong to this agent via the reservations table.
    # Simpler: query outcomes directly and let the caller filter.
    # The spec says "read-only 7-day rolling evaluated outcomes only" —
    # for an agent.  We'll filter outcomes by joining with reservations.
    #
    # But reservations may not exist for all outcomes.  The safest approach:
    # query outcomes whose task_id appears in reservations for this agent.
    # If no reservations → no data → insufficient.

    rows = conn.execute(
        "SELECT o.effect, o.expected_direction "
        "FROM outcomes o "
        "WHERE o.status IN ('confirmed', 'falsified') "
        "AND o.evaluated_at IS NOT NULL "
        "AND o.evaluated_at >= ? "
        "AND o.evaluated_at <= ? "
        "AND o.task_id IN ("
        "  SELECT DISTINCT task_id FROM guardrail_reservations WHERE agent_id = ?"
        ")",
        (window_start, window_end, agent_id),
    ).fetchall()

    total_evaluated = len(rows)

    if total_evaluated < min_samples:
        return DriftResult(
            agent_id=agent_id,
            total_evaluated=total_evaluated,
            wrong_direction_count=0,
            drift_rate=None,
            sufficient_evidence=False,
            window_start=window_start,
            window_end=window_end,
        )

    wrong_count = 0
    for row in rows:
        effect = row[0]
        direction = row[1]
        if effect is None:
            continue
        if direction == "higher" and effect <= 0:
            wrong_count += 1
        elif direction == "lower" and effect >= 0:
            wrong_count += 1

    drift_rate = wrong_count / total_evaluated if total_evaluated > 0 else None

    return DriftResult(
        agent_id=agent_id,
        total_evaluated=total_evaluated,
        wrong_direction_count=wrong_count,
        drift_rate=drift_rate,
        sufficient_evidence=True,
        window_start=window_start,
        window_end=window_end,
    )


# ---------------------------------------------------------------------------
# Public API: evaluate_trip_wire
# ---------------------------------------------------------------------------


def evaluate_trip_wire(
    conn: sqlite3.Connection,
    goal_id: str,
    primary_outcome_id: str,
    secondary_goal_id: str,
    baseline_measurement_id: int,
    latest_measurement_id: int,
    expected_min_ratio: float,
    expected_max_ratio: float,
    now: float,
) -> TripWireResult:
    """Read-only trip-wire: detect secondary degradation with primary improvement.

    Verified independent provenance: the secondary measurements must come
    from a different ``source_id`` than the primary outcome's goal.  Same
    source is rejected.  Stale, unknown, version-mismatch, and invalid
    baseline measurements are rejected.

    Trip condition: independently measured secondary degrades outside
    ``[expected_min_ratio, expected_max_ratio]`` AND primary improves.
    Otherwise returns ``tripped=None`` (unknown).
    """
    if not goal_id or not isinstance(goal_id, str):
        raise ValueError("goal_id must be a non-empty string")
    if not primary_outcome_id or not isinstance(primary_outcome_id, str):
        raise ValueError("primary_outcome_id must be a non-empty string")
    if not secondary_goal_id or not isinstance(secondary_goal_id, str):
        raise ValueError("secondary_goal_id must be a non-empty string")
    if not isinstance(baseline_measurement_id, int) or baseline_measurement_id <= 0:
        raise ValueError("baseline_measurement_id must be a positive integer")
    if not isinstance(latest_measurement_id, int) or latest_measurement_id <= 0:
        raise ValueError("latest_measurement_id must be a positive integer")
    if not isinstance(expected_min_ratio, (int, float)) or math.isnan(expected_min_ratio):
        raise ValueError("expected_min_ratio must be a finite number")
    if not isinstance(expected_max_ratio, (int, float)) or math.isnan(expected_max_ratio):
        raise ValueError("expected_max_ratio must be a finite number")
    if expected_min_ratio > expected_max_ratio:
        raise ValueError("expected_min_ratio must be <= expected_max_ratio")
    if not isinstance(now, (int, float)) or math.isnan(now) or math.isinf(now):
        raise ValueError("now must be a finite number")

    # Check required tables exist
    for table in ("outcomes", "goals", "goal_measurements"):
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if row is None:
            raise GuardrailSchemaNotInitializedError(
                f"Required table '{table}' not present; cannot evaluate trip wire"
            )

    # --- Primary outcome lookup ---
    primary_outcome = conn.execute(
        "SELECT * FROM outcomes WHERE id = ?", (primary_outcome_id,)
    ).fetchone()
    if primary_outcome is None:
        return TripWireResult(
            tripped=None,
            reason=f"primary outcome '{primary_outcome_id}' not found",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # Primary must be evaluated (confirmed) to be "improving"
    if primary_outcome["status"] != "confirmed":
        return TripWireResult(
            tripped=None,
            reason=f"primary outcome not confirmed (status={primary_outcome['status']})",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # --- Primary goal's source_id (for independence check) ---
    primary_goal = conn.execute(
        "SELECT * FROM goals WHERE id = ?", (goal_id,)
    ).fetchone()
    if primary_goal is None:
        return TripWireResult(
            tripped=None,
            reason=f"primary goal '{goal_id}' not found",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )
    primary_source_id = primary_goal["source_id"]

    # --- Secondary goal lookup ---
    secondary_goal = conn.execute(
        "SELECT * FROM goals WHERE id = ?", (secondary_goal_id,)
    ).fetchone()
    if secondary_goal is None:
        return TripWireResult(
            tripped=None,
            reason=f"secondary goal '{secondary_goal_id}' not found",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # --- Independence check: secondary source_id must differ from primary ---
    secondary_source_id = secondary_goal["source_id"]
    if secondary_source_id == primary_source_id:
        return TripWireResult(
            tripped=None,
            reason=(
                f"secondary source_id '{secondary_source_id}' is the same as "
                f"primary source_id; independent provenance required"
            ),
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # --- Baseline measurement ---
    baseline = conn.execute(
        "SELECT * FROM goal_measurements WHERE id = ?",
        (baseline_measurement_id,),
    ).fetchone()
    if baseline is None:
        return TripWireResult(
            tripped=None,
            reason=f"baseline measurement {baseline_measurement_id} not found",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # Baseline must belong to the secondary goal
    if baseline["goal_id"] != secondary_goal_id:
        return TripWireResult(
            tripped=None,
            reason="baseline measurement does not belong to secondary goal",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # Baseline value must be valid (non-zero for ratio)
    baseline_value = baseline["value"]
    if baseline_value is None or baseline_value == 0:
        return TripWireResult(
            tripped=None,
            reason="invalid baseline: value is zero or None",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # --- Latest measurement ---
    latest = conn.execute(
        "SELECT * FROM goal_measurements WHERE id = ?",
        (latest_measurement_id,),
    ).fetchone()
    if latest is None:
        return TripWireResult(
            tripped=None,
            reason=f"latest measurement {latest_measurement_id} not found",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # Latest must belong to the secondary goal
    if latest["goal_id"] != secondary_goal_id:
        return TripWireResult(
            tripped=None,
            reason="latest measurement does not belong to secondary goal",
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # Version consistency check
    baseline_version = baseline["metric_version"] if "metric_version" in baseline.keys() else None
    latest_version = latest["metric_version"] if "metric_version" in latest.keys() else None
    if baseline_version is not None and latest_version is not None:
        if baseline_version != latest_version:
            return TripWireResult(
                tripped=None,
                reason=(
                    f"metric_version mismatch: baseline={baseline_version}, "
                    f"latest={latest_version}"
                ),
                primary_outcome_id=primary_outcome_id,
                secondary_ratio=None,
                expected_min_ratio=expected_min_ratio,
                expected_max_ratio=expected_max_ratio,
            )

    # Staleness check: latest must not be stale relative to now
    secondary_max_age = secondary_goal["max_age_seconds"]
    latest_age = now - latest["source_timestamp"]
    if latest_age > secondary_max_age:
        return TripWireResult(
            tripped=None,
            reason=(
                f"latest measurement is stale: age={latest_age:.0f}s > "
                f"max_age={secondary_max_age}s"
            ),
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=None,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    # --- Compute ratio and evaluate ---
    latest_value = latest["value"]
    ratio = latest_value / baseline_value

    # Trip condition: secondary degrades outside expected range
    # AND primary improves (already confirmed above)
    if ratio < expected_min_ratio or ratio > expected_max_ratio:
        return TripWireResult(
            tripped=True,
            reason=(
                f"secondary ratio {ratio:.4f} outside "
                f"[{expected_min_ratio}, {expected_max_ratio}] "
                f"while primary improved"
            ),
            primary_outcome_id=primary_outcome_id,
            secondary_ratio=ratio,
            expected_min_ratio=expected_min_ratio,
            expected_max_ratio=expected_max_ratio,
        )

    return TripWireResult(
        tripped=False,
        reason="secondary within expected range",
        primary_outcome_id=primary_outcome_id,
        secondary_ratio=ratio,
        expected_min_ratio=expected_min_ratio,
        expected_max_ratio=expected_max_ratio,
    )


# ---------------------------------------------------------------------------
# Public API: propose_rollback
# ---------------------------------------------------------------------------


def propose_rollback(
    conn: sqlite3.Connection,
    trigger: str,
    outcome_ids: list[str],
) -> RollbackProposal:
    """Create an inert rollback proposal.

    Does NOT execute any git revert, deploy, card archive, or practice/patient
    mutation.  The proposal is a structured record requiring explicit
    release/QA approval before any action is taken.

    ``trigger`` describes what caused the proposal (e.g. "drift_rate_0.7",
    "trip_wire_secondary_degraded").
    ``outcome_ids`` lists the outcomes that triggered the proposal.
    """
    if not trigger or not isinstance(trigger, str):
        raise ValueError("trigger must be a non-empty string")
    if not isinstance(outcome_ids, list) or not outcome_ids:
        raise ValueError("outcome_ids must be a non-empty list")
    for oid in outcome_ids:
        if not oid or not isinstance(oid, str):
            raise ValueError("each outcome_id must be a non-empty string")

    # Check outcome schema for evidence collection
    outcome_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='outcomes'"
    ).fetchone()

    evidence_parts = []
    revision = "unknown"

    if outcome_exists:
        for oid in outcome_ids:
            row = conn.execute(
                "SELECT id, action_revision, status, effect, expected_direction, "
                "goal_id, hypothesis FROM outcomes WHERE id = ?",
                (oid,),
            ).fetchone()
            if row:
                evidence_parts.append(
                    f"outcome={row['id']}: status={row['status']}, "
                    f"effect={row['effect']}, direction={row['expected_direction']}, "
                    f"goal={row['goal_id']}, hypothesis={row['hypothesis']}"
                )
                if row["action_revision"] and row["action_revision"] != "unknown":
                    revision = row["action_revision"]

    evidence = "; ".join(evidence_parts) if evidence_parts else "no outcome evidence found"
    now = time.time()

    return RollbackProposal(
        proposal_id=str(uuid.uuid4()),
        trigger=trigger,
        outcome_ids=outcome_ids,
        revision=revision,
        evidence=evidence,
        created_at=now,
        requires_approval=True,
    )


# ---------------------------------------------------------------------------
# Public API: reset_guardrail
# ---------------------------------------------------------------------------


def reset_guardrail(
    conn: sqlite3.Connection,
    scope: str,
    actor: str,
    reason: str,
    approval_ref: str,
    now: float,
) -> None:
    """Manually reset a guardrail halt.

    Requires explicit actor and policy approval reference.  Audited
    append-only: prior spend records are preserved (never erased) and
    caps are not bypassed without new signed policy.

    Resolves the active halt for the given scope and records the reset
    in the audit log.
    """
    _require_guardrail_schema(conn)

    if not scope or not isinstance(scope, str):
        raise ValueError("scope must be a non-empty string")
    if not actor or not isinstance(actor, str):
        raise ValueError("actor must be a non-empty string")
    if not reason or not isinstance(reason, str):
        raise ValueError("reason must be a non-empty string")
    if not approval_ref or not isinstance(approval_ref, str):
        raise ValueError("approval_ref must be a non-empty string")
    if not isinstance(now, (int, float)) or math.isnan(now) or math.isinf(now):
        raise ValueError("now must be a finite number")

    conn.execute("BEGIN IMMEDIATE")
    try:
        # Resolve any active halts for this scope
        conn.execute(
            "UPDATE guardrail_halts SET "
            "resolved_at = ?, resolved_by = ? "
            "WHERE scope = ? AND resolved_at IS NULL",
            (now, actor, scope),
        )

        # Append audit record
        reset_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO guardrail_resets "
            "(id, scope, actor, reason, approval_ref, reset_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (reset_id, scope, actor, reason, approval_ref, now, now),
        )

        conn.execute("COMMIT")
    except BaseException:
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise

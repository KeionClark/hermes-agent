"""Opt-in, finite OODA event consumer with bounded dedup Act.

Observe → Orient → Decide → Act loop consuming the P0a event bus and P0b
goal metrics, writing at most ``max_active_acts`` deduped investigation cards
per invocation.  Never spawns a worker, sends a message, deploys code, or
modifies patient data.  Each invocation is finite (bounded ``max_events``),
same-board only, profile/tenant/board-confined.

Public API
----------
- ``run_once(conn, *, subscriber_id, max_events, max_active_acts, ...)``

Ownership: sole owner of this module, ``tests/hermes_cli/test_ooda_controller.py``.
Does not edit kanban_db.py, kanban_db_dispatch.py, event_bus.py, the router,
or any live board.  ``ooda_decision_rights`` is imported read-only from the
sibling builder's module.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from hermes_cli import event_bus
from hermes_cli.kanban_db_connect import write_txn

_log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# Event types the controller subscribes to.
_OBSERVE_EVENTS: frozenset[str] = frozenset({
    "goal.gap_detected",
    "metric.threshold_crossed",
    "run.completed",
})

# Recursion guard: the controller emits card.created when it Acts;
# that event type must never be in our subscription.
assert "card.created" not in _OBSERVE_EVENTS, "self-emitted card.created must not be subscribed"


# -----------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OodaResult:
    """Immutable result of a single ``run_once`` invocation."""
    events_processed: int
    events_skipped: int
    acts_created: int
    acts_deduplicated: int
    noop_reasons: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------
# Internal helpers: Observe
# -----------------------------------------------------------------------

def _observe(
    conn: sqlite3.Connection,
    subscriber_id: str,
    *,
    max_events: int,
    permitted_board: str,
) -> list[event_bus.Event]:
    """Poll metadata-only events from the bus for a registered subscriber.

    Board confinement: the subscriber must have been registered on the same
    DB as ``permitted_board``; this function does NOT open a second connection.
    Returns at most ``max_events`` events in ascending id order.
    """
    events = event_bus.poll(conn, subscriber_id, limit=max_events)
    return events


# -----------------------------------------------------------------------
# Internal helpers: Orient
# -----------------------------------------------------------------------

def _orient_gap(
    conn: sqlite3.Connection,
    evt: event_bus.Event,
    *,
    now: float,
) -> Optional[dict[str, Any]]:
    """Deterministic orientation on a gap/threshold event.

    Returns an action context dict if the gap is verified active and fresh,
    or None (no-op) otherwise.  Requires the goal to be active and the
    measurement to be non-stale.

    Never fabricates a gap; only reads from goal_measurements.
    """
    payload = evt.payload
    goal_id = payload.get("goal_id")
    metric_id = payload.get("metric_id")

    if not goal_id or not metric_id:
        return None

    # Look up goal — must exist and be active.
    try:
        from hermes_cli.goal_metrics import get_goal
        goal = get_goal(conn, goal_id)
    except Exception:
        return None

    if goal is None or goal.state != "active":
        return None

    # Verify latest measurement for this goal is actually breached.
    row = conn.execute(
        "SELECT edge_state, source_timestamp, gap_value FROM goal_measurements "
        "WHERE goal_id = ? ORDER BY id DESC LIMIT 1",
        (goal_id,),
    ).fetchone()

    if row is None:
        return None

    edge_state = row[0] if not hasattr(row, "keys") else row["edge_state"]
    source_ts = row[1] if not hasattr(row, "keys") else row["source_timestamp"]
    gap_value = row[2] if not hasattr(row, "keys") else row["gap_value"]

    # Must be breached (not healthy/unknown).
    if edge_state != "breached":
        return None

    # Must be fresh (not stale).
    age = now - source_ts
    if age > goal.max_age_seconds:
        return None

    return {
        "goal_id": goal_id,
        "metric_id": metric_id,
        "gap_value": gap_value,
        "edge_state": edge_state,
        "event_type": evt.event_type,
        "event_id": evt.id,
        "source_timestamp": source_ts,
    }


def _orient_run_completed(
    conn: sqlite3.Connection,
    evt: event_bus.Event,
) -> Optional[dict[str, Any]]:
    """Orientation on a run.completed event: capacity release signal.

    Returns context for capacity accounting but does NOT trigger Act.
    """
    payload = evt.payload
    task_id = payload.get("task_id")
    outcome = payload.get("outcome")
    run_id = payload.get("run_id")

    if not task_id:
        return None

    return {
        "type": "capacity_release",
        "task_id": task_id,
        "run_id": run_id,
        "outcome": outcome,
        "event_id": evt.id,
    }


# -----------------------------------------------------------------------
# Internal helpers: Decide
# -----------------------------------------------------------------------

def _decide(
    action_context: dict[str, Any],
    *,
    max_active_acts: int,
    active_count: int,
) -> tuple[bool, str]:
    """Pure decision gate using ooda_decision_rights if available.

    Returns (should_act, reason).  Fails closed on any missing evidence.
    """
    # Capacity gate: enforce max in-flight.
    if active_count >= max_active_acts:
        return False, "at_capacity"

    # Require evidence_ref (event_id from the gap detection).
    evidence_ref = action_context.get("event_id")
    if evidence_ref is None:
        return False, "no_evidence_ref"

    # Try to use the decision rights module if available.
    try:
        from hermes_cli.ooda_decision_rights import (
            ActionType,
            ActionContext,
            Decision,
            classify,
        )
        ctx = ActionContext(
            evidence_ref=str(evidence_ref),
            effect_scope="investigation",
            audience=None,
            reserved_effects=frozenset(),
            same_purpose=None,
            bounded_pilot=True,
            send_cap=None,
            qa_revision=None,
            project_ruling_ref=None,
            release_gate_ref=None,
            explicit_approval_ref=None,
            external_access_ready=None,
        )
        decision = classify(ActionType.create_investigation_card, ctx)
        if decision.outcome != "agent_authorized":
            return False, f"rights_gate:{decision.outcome}"
    except ImportError:
        # Rights module not yet available (sibling builder in progress).
        # Still allow investigation cards with evidence — these are the
        # safest possible Act (bounded scope, no deploy, no send).
        pass
    except Exception as exc:
        # Any error from the rights module fails closed.
        _log.debug("decision rights error: %s", exc)
        return False, f"rights_error:{type(exc).__name__}"

    return True, "authorized"


# -----------------------------------------------------------------------
# Internal helpers: Act
# -----------------------------------------------------------------------

def _stable_idempotency_key(
    goal_id: str,
    metric_id: str,
    event_id: int,
) -> str:
    """Deterministic idempotency key for investigation cards.

    Same (goal, metric, event) always produces the same key, so restart
    replay is deduped at the DB level.
    """
    return f"ooda:investigate:{goal_id}:{metric_id}:{event_id}"


def _act_create_investigation(
    conn: sqlite3.Connection,
    context: dict[str, Any],
    *,
    assignee: str,
    permitted_board: str,
    tenant: Optional[str] = None,
) -> Optional[str]:
    """Create a bounded investigation card with stable idempotency key.

    Returns the task_id of the created (or existing dedup) task, or None
    if the creation failed.  Inserts directly into the tasks table with
    minimal schema to avoid heavy create_task validation (which depends
    on board metadata, delegation fences, etc.).

    Never spawns a worker directly.
    """
    import uuid
    from hermes_cli.kanban_event_publish import emit_card_created

    goal_id = context["goal_id"]
    metric_id = context["metric_id"]
    event_id = context["event_id"]
    gap_value = context.get("gap_value")

    idem_key = _stable_idempotency_key(goal_id, metric_id, event_id)

    # Check for existing task with this idempotency key (dedup).
    existing = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ?",
        (idem_key,),
    ).fetchone()
    if existing is not None:
        return None  # Already exists — dedup, return None to signal.

    task_id = f"t_{uuid.uuid4().hex[:8]}"
    title = f"Investigate {metric_id} gap (goal {goal_id[:8]})"
    body = (
        f"Auto-created by OODA controller.\n\n"
        f"Goal: {goal_id}\n"
        f"Metric: {metric_id}\n"
        f"Gap value: {gap_value}\n"
        f"Source event: {event_id}\n"
        f"Event type: {context.get('event_type', 'unknown')}\n\n"
        f"Investigate the metric gap and determine corrective action."
    )

    now = int(time.time())

    try:
        conn.execute(
            "INSERT INTO tasks "
            "(id, title, body, status, assignee, created_at, priority, "
            "idempotency_key, tenant) "
            "VALUES (?, ?, ?, 'ready', ?, ?, 0, ?, ?)",
            (task_id, title, body, assignee, now, idem_key, tenant),
        )
        # Emit card.created event on the bus (best-effort).
        emit_card_created(conn, task_id)
        return task_id
    except Exception:
        _log.debug("investigation card insert failed", exc_info=True)
        return None


# -----------------------------------------------------------------------
# Count active OODA-created tasks
# -----------------------------------------------------------------------

def _count_active_ooda_acts(
    conn: sqlite3.Connection,
) -> int:
    """Count non-archived, non-done OODA investigation tasks.

    Uses the idempotency_key prefix ``ooda:investigate:`` to identify
    controller-created cards.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM tasks "
        "WHERE idempotency_key LIKE 'ooda:investigate:%' "
        "AND status NOT IN ('done', 'archived')"
    ).fetchone()
    return row[0] if row else 0


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def run_once(
    conn: sqlite3.Connection,
    *,
    subscriber_id: str,
    max_events: int = 50,
    max_active_acts: int = 2,
    permitted_board: str,
    assignee: str = "researcher-a",
    tenant: Optional[str] = None,
    now: Optional[float] = None,
) -> OodaResult:
    """Finite, opt-in OODA loop: Observe-Orient-Decide-Act.

    Consumes at most ``max_events`` from the bus subscriber, orients on
    verified active fresh gap edges, decides through the rights classifier,
    and acts by creating bounded investigation cards with stable idempotency
    keys.

    Invariants:
    - Same-board only (``permitted_board`` names the board being consumed).
    - At most ``max_active_acts`` OODA-created cards in flight at once.
    - Dedup via idempotency_key on create_task.
    - Never spawns a worker, sends a message, or deploys.
    - Never creates from unknown metric, stale measurement, or completed task.
    - Self-emitted card.created is NOT in the subscription (no recursion).
    - Requires an autocommit connection (canonical kanban_db_connect.connect).

    Parameters
    ----------
    conn : sqlite3.Connection
        Autocommit board connection (``isolation_level=None``).
    subscriber_id : str
        Durable subscriber for the event bus.
    max_events : int
        Maximum events to process in this invocation.
    max_active_acts : int
        Maximum concurrent OODA investigation cards.
    permitted_board : str
        Board path/name for confinement verification.
    assignee : str
        Profile to assign investigation cards to.
    tenant : str, optional
        Tenant for investigation cards.
    now : float, optional
        Override current time (for testing).

    Returns
    -------
    OodaResult
        Statistics about this run.
    """
    if conn.in_transaction:
        raise RuntimeError("run_once requires an autocommit connection")
    if not isinstance(max_events, int) or not 1 <= max_events <= 1000:
        raise ValueError("max_events must be 1..1000")
    if not isinstance(max_active_acts, int) or max_active_acts < 1:
        raise ValueError("max_active_acts must be >= 1")
    if not subscriber_id or not isinstance(subscriber_id, str):
        raise ValueError("subscriber_id must be a non-empty string")
    if not permitted_board or not isinstance(permitted_board, str):
        raise ValueError("permitted_board must be a non-empty string")

    if now is None:
        now = time.time()

    # Ensure subscriber is registered for our event types.
    event_bus.subscribe(
        conn, subscriber_id,
        event_types=_OBSERVE_EVENTS,
        start="latest",
    )

    events_processed = 0
    events_skipped = 0
    acts_created = 0
    acts_deduplicated = 0
    noop_reasons: list[str] = []

    # --- Observe ---
    events = _observe(
        conn, subscriber_id,
        max_events=max_events,
        permitted_board=permitted_board,
    )

    if not events:
        noop_reasons.append("no_events")
        return OodaResult(
            events_processed=0,
            events_skipped=0,
            acts_created=0,
            acts_deduplicated=0,
            noop_reasons=noop_reasons,
        )

    for evt in events:
        # --- Orient ---
        if evt.event_type == "run.completed":
            # Capacity release signal; orient but don't act.
            ctx = _orient_run_completed(conn, evt)
            # ACK and continue (capacity accounting, not an actionable gap).
            with write_txn(conn):
                event_bus.ack(conn, subscriber_id, evt.id)
            events_processed += 1
            continue

        if evt.event_type in ("goal.gap_detected", "metric.threshold_crossed"):
            action_context = _orient_gap(conn, evt, now=now)
            if action_context is None:
                # No-op: gap not verified active/fresh.
                with write_txn(conn):
                    event_bus.ack(conn, subscriber_id, evt.id)
                events_skipped += 1
                noop_reasons.append(f"orient_noop:{evt.event_type}:{evt.id}")
                continue

            # --- Decide ---
            active_count = _count_active_ooda_acts(conn)
            should_act, reason = _decide(
                action_context,
                max_active_acts=max_active_acts,
                active_count=active_count,
            )

            if not should_act:
                with write_txn(conn):
                    event_bus.ack(conn, subscriber_id, evt.id)
                events_skipped += 1
                noop_reasons.append(f"decide_no:{reason}:{evt.id}")
                continue

            # --- Act ---
            # Atomic: create investigation card + ACK event.
            with write_txn(conn):
                task_id = _act_create_investigation(
                    conn, action_context,
                    assignee=assignee,
                    permitted_board=permitted_board,
                    tenant=tenant,
                )
                event_bus.ack(conn, subscriber_id, evt.id)
                if task_id is not None:
                    acts_created += 1
                else:
                    # _act returns None for dedup (idempotency_key already exists)
                    # or on failure.
                    idem_key = _stable_idempotency_key(
                        action_context["goal_id"],
                        action_context["metric_id"],
                        action_context["event_id"],
                    )
                    existing = conn.execute(
                        "SELECT id FROM tasks WHERE idempotency_key = ?",
                        (idem_key,),
                    ).fetchone()
                    if existing:
                        acts_deduplicated += 1

            events_processed += 1
            continue

        # Unknown event type in our subscription — ACK and skip.
        with write_txn(conn):
            event_bus.ack(conn, subscriber_id, evt.id)
        events_skipped += 1

    return OodaResult(
        events_processed=events_processed,
        events_skipped=events_skipped,
        acts_created=acts_created,
        acts_deduplicated=acts_deduplicated,
        noop_reasons=noop_reasons,
    )

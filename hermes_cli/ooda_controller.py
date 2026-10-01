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
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_cli import event_bus, ooda_guardrails
from hermes_cli.kanban_db_connect import write_txn

_log = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

# Event types the controller subscribes to.
_OBSERVE_EVENTS: frozenset[str] = frozenset({
    "goal.gap_detected",
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
    rollback_proposals: list[ooda_guardrails.RollbackProposal] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GuardrailPolicy:
    """Explicit, externally verified budget and provenance for dark Act admission.

    Values here are supplied by a trusted operator; this class does not grant
    spending approval. The controller never guesses model price or actual usage.
    """
    agent_id: str
    lane_id: str
    estimated_tokens: int
    price_per_million_usd: Decimal | str
    caps: Mapping[str, Decimal | str]
    approval_ref: str
    drift_max_ratio: float
    drift_min_samples: int = 10
    # Optional independent secondary signal: (primary outcome, secondary goal,
    # baseline measurement, min ratio, max ratio). Latest is looked up here.
    trip_wire: Optional[tuple[str, str, int, float, float]] = None


def _halt(conn: sqlite3.Connection, scope: str, reason: str, now: float) -> None:
    """Persist a deduplicated halt, without ACKing the triggering event."""
    with write_txn(conn):
        ooda_guardrails._record_halt(conn, scope, reason, now)


def _check_safety(conn: sqlite3.Connection, policy: GuardrailPolicy, goal_id: str,
                  now: float) -> tuple[Optional[str], Optional[ooda_guardrails.RollbackProposal]]:
    drift = ooda_guardrails.evaluate_drift(
        conn, policy.agent_id, now, min_samples=policy.drift_min_samples,
    )
    if drift.sufficient_evidence and drift.drift_rate is not None and drift.drift_rate > policy.drift_max_ratio:
        outcomes = conn.execute(
            "SELECT o.id FROM outcomes o WHERE o.status IN ('confirmed', 'falsified') "
            "AND o.evaluated_at BETWEEN ? AND ? AND o.task_id IN "
            "(SELECT task_id FROM guardrail_reservations WHERE agent_id = ?) "
            "ORDER BY o.evaluated_at DESC LIMIT 20",
            (drift.window_start, now, policy.agent_id),
        ).fetchall()
        proposal = ooda_guardrails.propose_rollback(
            conn, 'evaluated_outcome_drift', [r[0] for r in outcomes],
        )
        return "drift:threshold_exceeded", proposal
    if policy.trip_wire is not None:
        outcome_id, secondary_goal_id, baseline_id, minimum, maximum = policy.trip_wire
        outcome = conn.execute(
            "SELECT goal_id FROM outcomes WHERE id = ?", (outcome_id,),
        ).fetchone()
        if outcome is None or outcome[0] != goal_id:
            return "trip_wire:primary_outcome_goal_mismatch", None
        latest = conn.execute(
            "SELECT id FROM goal_measurements WHERE goal_id = ? ORDER BY id DESC LIMIT 1",
            (secondary_goal_id,),
        ).fetchone()
        if latest is None:
            return "trip_wire:secondary_measurement_missing", None
        provenance = conn.execute(
            "SELECT g.source_id, b.source_id, l.source_id, b.metric_version, l.metric_version "
            "FROM goals g JOIN goal_measurements b ON b.id = ? "
            "JOIN goal_measurements l ON l.id = ? WHERE g.id = ?",
            (baseline_id, latest[0], goal_id),
        ).fetchone()
        if (provenance is None or not provenance[1] or not provenance[2]
                or provenance[0] in (provenance[1], provenance[2])
                or provenance[1] != provenance[2]
                or provenance[3] is None or provenance[3] != provenance[4]):
            return "trip_wire:unverified_independent_measurements", None
        trip = ooda_guardrails.evaluate_trip_wire(
            conn, goal_id, outcome_id, secondary_goal_id, baseline_id,
            latest[0], minimum, maximum, now,
        )
        if trip.tripped is None:
            return f"trip_wire:unknown:{trip.reason}", None
        if trip.tripped:
            proposal = ooda_guardrails.propose_rollback(conn, 'independent_trip_wire', [outcome_id])
            return "trip_wire:tripped", proposal
    return None, None


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

    The caller verifies the connection's board before subscribing; this
    function does not open a second connection.
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

    if not isinstance(task_id, str) or not isinstance(run_id, int):
        return None
    row = conn.execute(
        "SELECT 1 FROM tasks t JOIN task_runs r ON r.task_id = t.id "
        "WHERE t.id = ? AND r.id = ? AND r.ended_at IS NOT NULL "
        "AND t.idempotency_key LIKE 'ooda:investigate:%'",
        (task_id, run_id),
    ).fetchone()
    if row is None:
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
            audience="ooda-controller",
            reserved_effects=frozenset(),
            same_purpose=True,
            bounded_pilot=True,
            send_cap=None,
            qa_revision=None,
            project_ruling_ref=None,
            release_gate_ref=None,
            explicit_approval_ref=None,
            external_access_ready=None,
            bounded_safe_scope=True,
            dependency_ready=True,
        )
        decision = classify(ActionType.CREATE_INVESTIGATION_CARD, ctx)
        if decision.outcome != "agent_authorized":
            return False, f"rights_gate:{decision.outcome}"
    except ImportError:
        # Partial rollouts must not silently bypass the policy classifier.
        return False, "rights_unavailable"
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

    task_id = context.get("reserved_task_id") or f"t_{uuid.uuid4().hex[:8]}"
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
    assignee: str = "default",
    tenant: Optional[str] = None,
    now: Optional[float] = None,
    guardrail_policy: Optional[GuardrailPolicy] = None,
    guardrails_enabled: bool = True,
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
    if type(max_active_acts) is not int or not 1 <= max_active_acts <= 2:
        raise ValueError("max_active_acts must be 1..2")
    if not subscriber_id or not isinstance(subscriber_id, str):
        raise ValueError("subscriber_id must be a non-empty string")
    if not permitted_board or not isinstance(permitted_board, str):
        raise ValueError("permitted_board must be a non-empty string")
    # Verify the actual opened database, not a caller-supplied label.  Both
    # reads and writes stay on this very connection; no implicit board switch.
    dbs = conn.execute("PRAGMA database_list").fetchall()
    main_files = [row[2] for row in dbs if row[1] == "main"]
    if len(main_files) != 1 or not main_files[0] or not Path(permitted_board).is_file():
        raise ValueError("permitted_board must name the opened file board")
    if Path(main_files[0]).resolve() != Path(permitted_board).resolve():
        raise ValueError("permitted_board does not match connection")
    # A ready card for an unserved profile is a stranded card.  Use the same
    # profile liveness predicate as the dispatcher, before subscription/ACK.
    from hermes_cli.profiles import profile_exists
    if not isinstance(assignee, str) or not profile_exists(assignee):
        raise ValueError("assignee must be an existing live profile")

    policy = guardrail_policy
    if guardrails_enabled:
        if not isinstance(guardrail_policy, GuardrailPolicy):
            raise ValueError("explicit guardrail policy and approved caps required before Act")
        policy = guardrail_policy
        if (not all(isinstance(v, str) and v.strip() for v in
                    (policy.agent_id, policy.lane_id, policy.approval_ref))
                or type(policy.estimated_tokens) is not int or policy.estimated_tokens <= 0
                or not isinstance(policy.caps, Mapping)
                or set(policy.caps) != {"task_token", "agent_day", "lane_week", "global_day"}
                or not isinstance(policy.drift_max_ratio, (int, float))
                or not math.isfinite(policy.drift_max_ratio)
                or not 0 <= policy.drift_max_ratio <= 1
                or type(policy.drift_min_samples) is not int or policy.drift_min_samples < 1):
            raise ValueError("guardrail policy requires verified price, approval, four caps and drift threshold")
        # No implicit schema migration, including on an empty event bus.
        ooda_guardrails._require_guardrail_schema(conn)
        # Reserve validates price and cap types even when there is no event;
        # no speculative reservation is created here.
        ooda_guardrails._to_micro_usd(policy.price_per_million_usd)
    elif guardrail_policy is not None:
        raise ValueError("guardrail policy cannot be bypassed by disabling guardrails")

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
    rollback_proposals: list[ooda_guardrails.RollbackProposal] = []

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

        if evt.event_type == "goal.gap_detected":
            action_context = _orient_gap(conn, evt, now=now)
            if action_context is None:
                # No-op: gap not verified active/fresh.
                with write_txn(conn):
                    event_bus.ack(conn, subscriber_id, evt.id)
                events_skipped += 1
                noop_reasons.append(f"orient_noop:{evt.event_type}:{evt.id}")
                continue

            # --- Decide ---
            # Capacity, rights, card insertion and ACK share one IMMEDIATE write
            # transaction. A second controller cannot race the capacity read.
            deferred_for_capacity = False
            denied_reason = None
            created_this_event = False
            try:
              with write_txn(conn):
                idem_key = _stable_idempotency_key(
                    action_context["goal_id"],
                    action_context["metric_id"],
                    action_context["event_id"],
                )
                existing = conn.execute(
                    "SELECT id FROM tasks WHERE idempotency_key = ?", (idem_key,),
                ).fetchone()
                if existing:
                    acts_deduplicated += 1
                else:
                    active_count = _count_active_ooda_acts(conn)
                    should_act, reason = _decide(
                        action_context,
                        max_active_acts=max_active_acts,
                        active_count=active_count,
                    )
                    if should_act:
                        if guardrails_enabled:
                            assert policy is not None
                            denied_reason, proposal = _check_safety(
                                conn, policy, action_context["goal_id"], now,
                            )
                            if proposal is not None:
                                rollback_proposals.append(proposal)
                            if denied_reason is None:
                                reserved_task_id = f"t_{uuid.uuid4().hex[:8]}"
                                reservation = ooda_guardrails.reserve_act(
                                    conn, task_id=reserved_task_id, agent_id=policy.agent_id,
                                    lane_id=policy.lane_id, reservation_key=idem_key,
                                    estimated_tokens=policy.estimated_tokens,
                                    price_per_million_usd=policy.price_per_million_usd,
                                    caps=policy.caps, now=now, approval_ref=policy.approval_ref,
                                )
                                if not reservation.allowed:
                                    denied_reason = reservation.reason
                                else:
                                    action_context["reserved_task_id"] = reserved_task_id
                        if denied_reason is not None:
                            deferred_for_capacity = True
                        else:
                            task_id = _act_create_investigation(
                                conn, action_context,
                                assignee=assignee,
                                permitted_board=permitted_board,
                                tenant=tenant,
                            )
                            if task_id is None:
                                raise RuntimeError("investigation card not persisted; keep event pending")
                            created_this_event = True
                    else:
                        events_skipped += 1
                        noop_reasons.append(f"decide_no:{reason}:{evt.id}")
                        deferred_for_capacity = reason == "at_capacity"
                if not deferred_for_capacity:
                    event_bus.ack(conn, subscriber_id, evt.id)
            except Exception as exc:
                if not guardrails_enabled:
                    raise
                denied_reason = f"act_failure:{type(exc).__name__}"
            if denied_reason is not None:
                # Any reservation and card from a failed transaction rolled back.
                # Cap-denial halt inside reserve remains if the transaction committed.
                assert policy is not None
                _halt(conn, f"agent:{policy.agent_id}", denied_reason, now)
                noop_reasons.append(f"guardrail_halt:{denied_reason}:{evt.id}")
                break
            if created_this_event:
                acts_created += 1
            if deferred_for_capacity:
                # Keep the first unserved edge pending, in order, until a
                # terminal run frees a slot. ACK would lose it forever.
                break
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
        rollback_proposals=rollback_proposals,
    )


def propose_gap_once(
    conn: sqlite3.Connection, *, subscriber_id: str, candidates: list[dict[str, Any]],
    max_events: int = 20, now: Optional[float] = None,
) -> int:
    """Explicit opt-in gap-event producer. Proposals are inert, never Act cards.

    Candidates are supplied by the caller, not synthesized from a metric or
    routed to a worker. A retry sees the same event identity and deduplicates.
    No scheduler or canonical-board subscription is installed here.
    """
    from hermes_cli.proposal_queue import propose_for_gap

    if conn.in_transaction:
        raise RuntimeError("proposal producer needs an autocommit connection")
    if not 1 <= max_events <= 100:
        raise ValueError("max_events must be 1..100")
    now = time.time() if now is None else now
    event_bus.subscribe(conn, subscriber_id, event_types={"goal.gap_detected"}, start="beginning")
    produced = 0
    for evt in event_bus.poll(conn, subscriber_id, limit=max_events):
        goal_id = evt.payload.get("goal_id")
        metric_id = evt.payload.get("metric_id")
        if not isinstance(goal_id, str) or not isinstance(metric_id, str):
            with write_txn(conn):
                event_bus.ack(conn, subscriber_id, evt.id)
            continue
        # A later healthy observation invalidates an old gap event.
        row = conn.execute(
            "SELECT id, source_timestamp, edge_state FROM goal_measurements "
            "WHERE goal_id = ? ORDER BY id DESC LIMIT 1", (goal_id,),
        ).fetchone()
        if row and row["edge_state"] == "breached" and row["source_timestamp"] == evt.payload.get("source_timestamp"):
            for candidate in candidates:
                if candidate["goal_id"] != goal_id or candidate["metric_id"] != metric_id:
                    continue
                propose_for_gap(
                    conn, goal_id=goal_id, metric_id=metric_id, gap_measurement_id=row["id"],
                    action_type=candidate["action_type"], hypothesis=candidate["hypothesis"],
                    proposed_action=candidate["proposed_action"], impact=candidate["impact"],
                    confidence=candidate["confidence"], cost=candidate["cost"],
                    decision_tier=candidate["decision_tier"],
                    idempotency_key=f"gap_proposal:{evt.id}:{candidate['idempotency_key']}", now=now,
                )
                produced += 1
        with write_txn(conn):
            event_bus.ack(conn, subscriber_id, evt.id)
    return produced

"""Typed publish helpers for Kanban lifecycle events on the event bus.

P0a bridge: every Kanban status mutation emits a corresponding event on the
durable event bus (``hermes_cli.event_bus``) within the same write transaction.

Event types
-----------
- ``card.created``       — once on successful INSERT (create_task and decomposer child).
- ``card.blocked``       — once on actual entry to ``blocked`` status (incl.
  ``initial_status='blocked'`` and dispatch failure breaker).  NOT emitted for
  in-place reclassification, dependency ``todo``, or loop ``triage``.
- ``card.completed``     — once on successful transition to ``done``
  (complete_task, swarm inline, dashboard direct done).
- ``run.completed``      — once on successful terminal close of an actual or
  explicitly synthesized Kanban task run (including failures/reclaims).
- ``card.status_changed``— every actual status transition: dispatcher
  claim/reclaim/timeout/crash, link/unlink/promote/demote, review/request_changes,
  unblock/reopen/triage/archive, direct dashboard and swarm.

Payload contract (privacy-safe, no PII/PHI):
    ``card.status_changed``: ``{task_id, from, to, run_id}``
    ``card.created``:        ``{task_id, run_id}``
    ``card.blocked``:        ``{task_id, run_id, block_kind}``
    ``card.completed``:      ``{task_id, run_id}``
    ``run.completed``:       ``{task_id, run_id, outcome}`` (no summary/error/metadata)

No idempotency key: each guarded CAS mutation fires once; router dedupes by
``event.id``.

Exceptions (documented, never published):
    - ``kanban_transfer.py`` ``_scrub_local_state``: bulk cleanup on export/import.
    - ``kanban_db_connect.py`` schema migrations.
    - ``gc_events`` / ``delete_archived_task`` / ``delete_task``: cleanup paths.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

_log = logging.getLogger(__name__)


def _bus_available(conn: sqlite3.Connection) -> bool:
    """Check if the event bus tables exist on this connection.

    Legacy DBs initialized before the P0a migration may not have
    ``hermes_events``; never crash a board mutation because the bus is absent.
    """
    try:
        conn.execute(
            "SELECT 1 FROM hermes_event_schema_version WHERE id = 1"
        ).fetchone()
        return True
    except sqlite3.OperationalError:
        return False


# Cache per connection id so we probe once per connection lifetime.
_bus_probed: dict[int, bool] = {}


def _has_bus(conn: sqlite3.Connection) -> bool:
    cid = id(conn)
    cached = _bus_probed.get(cid)
    if cached is not None:
        return cached
    ok = _bus_available(conn)
    _bus_probed[cid] = ok
    return ok


def _publish_safe(
    conn: sqlite3.Connection,
    *,
    event_type: str,
    payload: dict,
) -> None:
    """Best-effort publish into the event bus; never raises.

    Caller MUST hold an active write transaction.
    """
    if not _has_bus(conn):
        return
    try:
        from hermes_cli.event_bus import publish

        publish(
            conn,
            event_type=event_type,
            source="kanban",
            payload=payload,
        )
    except Exception:
        # Never break a board mutation because the bus is unavailable.
        _log.debug("kanban event publish failed for %s", event_type, exc_info=True)


# ---- Public typed helpers ---------------------------------------------------

def emit_card_created(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: Optional[int] = None,
) -> None:
    """Emit ``card.created`` once on successful task INSERT."""
    _publish_safe(
        conn,
        event_type="card.created",
        payload={"task_id": task_id, "run_id": run_id},
    )


def emit_card_blocked(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: Optional[int] = None,
    block_kind: Optional[str] = None,
) -> None:
    """Emit ``card.blocked`` on actual entry to blocked status."""
    payload: dict = {"task_id": task_id, "run_id": run_id}
    if block_kind is not None:
        payload["block_kind"] = block_kind
    _publish_safe(
        conn,
        event_type="card.blocked",
        payload=payload,
    )


def emit_card_completed(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: Optional[int] = None,
) -> None:
    """Emit ``card.completed`` once on successful transition to done."""
    _publish_safe(
        conn,
        event_type="card.completed",
        payload={"task_id": task_id, "run_id": run_id},
    )


def emit_status_changed(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    from_status: Optional[str],
    to_status: str,
    run_id: Optional[int] = None,
) -> None:
    """Emit ``card.status_changed`` on every actual status transition.

    Caller should compare before/after and skip when unchanged.
    """
    if from_status == to_status:
        return
    payload: dict = {
        "task_id": task_id,
        "from": from_status,
        "to": to_status,
    }
    if run_id is not None:
        payload["run_id"] = run_id
    _publish_safe(
        conn,
        event_type="card.status_changed",
        payload=payload,
    )


def emit_run_completed(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: Optional[int] = None,
    outcome: Optional[str] = None,
) -> None:
    """Emit ``run.completed`` on terminal run close (complete or fail).

    Payload is restricted to the allowlisted triple {task_id, run_id, outcome}.
    This is a NEW event type — no pre-existing ``run.completed`` exists in the
    Kanban bus (the gateway SSE ``run.completed`` is unrelated).  The
    lifecycle owner publishes from ``_end_run`` / ``_synthesize_ended_run``
    inside the mutation's write transaction, covering dispatcher, dashboard
    and swarm paths without separate per-caller emissions.
    """
    payload: dict = {"task_id": task_id}
    if run_id is not None:
        payload["run_id"] = run_id
    if outcome is not None:
        payload["outcome"] = outcome
    _publish_safe(
        conn,
        event_type="run.completed",
        payload=payload,
    )


def clear_bus_cache() -> None:
    """Reset the per-connection bus probe cache (for tests)."""
    _bus_probed.clear()

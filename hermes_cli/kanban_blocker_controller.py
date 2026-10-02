"""Read-only finite controller tick for blocked Kanban roots.

The controller never changes cards. It turns the graph into a deterministic
ledger that an existing scheduler/notifier can poll and deduplicate by each
escalation_key. This keeps human-held gates visible without unblocking cards,
spawning follow-ups, or growing the dependency graph.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

ROOT_BLOCK_KINDS = ("needs_input", "capability")
DEFAULT_STALE_GATE_SECONDS = 24 * 60 * 60
DEFAULT_READY_STARVATION_SECONDS = 60 * 60


def _event_payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        return json.loads(row["payload"] or "{}")
    except (TypeError, ValueError):
        return {}


def _root_gate(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    """Return the oldest operator-facing blocked ancestor, or the card itself."""
    row = conn.execute(
        """
        WITH RECURSIVE upstream(id) AS (
            SELECT id FROM tasks WHERE id = ?
            UNION
            SELECT links.parent_id FROM task_links AS links JOIN upstream ON links.child_id = upstream.id
        )
        SELECT tasks.* FROM tasks JOIN upstream ON upstream.id = tasks.id
         WHERE tasks.status = 'blocked' AND tasks.block_kind IN (?, ?)
         ORDER BY tasks.created_at ASC, tasks.id ASC LIMIT 1
        """, (task_id, *ROOT_BLOCK_KINDS),
    ).fetchone()
    if row is not None:
        return row
    fallback = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if fallback is None:
        raise ValueError(f"unknown task {task_id}")
    return fallback


def _gate_activity(conn: sqlite3.Connection, task_id: str) -> tuple[int, str]:
    events = conn.execute(
        "SELECT kind, payload, created_at FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
    ).fetchall()
    latest_block = next((row for row in reversed(events) if row["kind"] == "blocked"), None)
    blocked_at = int(latest_block["created_at"]) if latest_block is not None else 0
    reason = _event_payload(latest_block).get("reason") if latest_block is not None else None
    activity = max([blocked_at, *(int(row["created_at"]) for row in events if int(row["created_at"]) >= blocked_at)])
    return activity, str(reason or "no explicit block reason recorded")


def _ready_since(conn: sqlite3.Connection, task_id: str, created_at: int) -> int:
    row = conn.execute(
        """
        SELECT MAX(created_at) AS ts FROM task_events
         WHERE task_id = ? AND kind IN ('created', 'promoted', 'unblocked', 'reclaimed', 'status')
        """, (task_id,),
    ).fetchone()
    return int(row["ts"] or created_at)


def controller_tick(
    conn: sqlite3.Connection, *, now: Optional[int] = None,
    stale_gate_seconds: int = DEFAULT_STALE_GATE_SECONDS,
    ready_starvation_seconds: int = DEFAULT_READY_STARVATION_SECONDS,
) -> dict[str, Any]:
    """Produce a finite, read-only gate ledger and deduplicable escalations."""
    now = int(time.time()) if now is None else int(now)
    if stale_gate_seconds < 1 or ready_starvation_seconds < 1:
        raise ValueError("controller intervals must be positive")

    blocked = conn.execute("SELECT * FROM tasks WHERE status = 'blocked' ORDER BY priority DESC, created_at, id").fetchall()
    grouped: dict[str, list[sqlite3.Row]] = {}
    roots: dict[str, sqlite3.Row] = {}
    for task in blocked:
        root = _root_gate(conn, task["id"])
        roots[root["id"]] = root
        grouped.setdefault(root["id"], []).append(task)

    ledger: list[dict[str, Any]] = []
    escalations: list[dict[str, Any]] = []
    for root_id in sorted(grouped, key=lambda tid: (-int(roots[tid]["priority"] or 0), tid)):
        root = roots[root_id]
        last_check_at, reason = _gate_activity(conn, root_id)
        interval = 60 * 60 if int(root["priority"] or 0) >= 90 else 6 * 60 * 60
        next_check_at = last_check_at + interval
        member_ids = [row["id"] for row in grouped[root_id]]
        entry = {
            "root_gate_id": root_id,
            "owner": root["assignee"],
            "reason": reason,
            "block_kind": root["block_kind"],
            "last_check_at": last_check_at,
            "next_check_at": next_check_at,
            "blocked_task_ids": member_ids,
            "blocked_count": len(member_ids),
            "next_action": "verify the named changed condition; resolve only with evidence",
        }
        ledger.append(entry)
        if now >= next_check_at:
            escalations.append({
                "kind": "gate_check_due",
                "severity": "critical" if int(root["priority"] or 0) >= 90 else "warning",
                "root_gate_id": root_id,
                "owner": root["assignee"],
                "reason": reason,
                "next_check_at": next_check_at,
                "action": "verify_changed_condition",
                "command": f"hermes kanban show {root_id}",
                "escalation_key": f"gate-check-due:{root_id}:{last_check_at}",
            })
        if now >= last_check_at + stale_gate_seconds:
            escalations.append({
                "kind": "stale_gate",
                "severity": "critical" if int(root["priority"] or 0) >= 90 else "warning",
                "root_gate_id": root_id,
                "owner": root["assignee"],
                "reason": reason,
                "last_check_at": last_check_at,
                "action": "verify_changed_condition",
                "command": f"hermes kanban show {root_id}",
                "escalation_key": f"stale-gate:{root_id}:{last_check_at}",
            })

    ready = conn.execute(
        "SELECT * FROM tasks WHERE status = 'ready' AND current_run_id IS NULL ORDER BY priority DESC, created_at, id"
    ).fetchall()
    for task in ready:
        ready_since = _ready_since(conn, task["id"], int(task["created_at"]))
        if now < ready_since + ready_starvation_seconds:
            continue
        escalations.append({
            "kind": "ready_starvation",
            "severity": "critical" if int(task["priority"] or 0) >= 90 else "warning",
            "task_id": task["id"],
            "owner": task["assignee"],
            "ready_since_at": ready_since,
            "action": "inspect_dispatcher_and_profile_capacity",
            "command": f"hermes kanban diagnostics --task {task['id']}",
            "escalation_key": f"ready-starvation:{task['id']}:{ready_since}",
        })
    return {"generated_at": now, "ledger": ledger, "escalations": escalations}


def apply_tick(
    conn: sqlite3.Connection, report: dict[str, Any], *, author: str = "blocker-controller",
    dispatch_once_fn=None,
) -> dict[str, Any]:
    """Apply one bounded wake-up pass without unblocking or creating cards.

    One marker comment is written per due/stale root condition. Existing markers
    are durable idempotency receipts. A single normal dispatch pass is requested
    when at least one ready card is starved; callers may inject it for testing.
    """
    by_root: dict[str, dict[str, Any]] = {}
    ready_starved = False
    for escalation in report.get("escalations", []):
        if escalation.get("kind") == "ready_starvation":
            ready_starved = True
            continue
        if escalation.get("kind") not in {"gate_check_due", "stale_gate"}:
            continue
        root_id = escalation["root_gate_id"]
        prior = by_root.get(root_id)
        if prior is None or escalation["kind"] == "stale_gate":
            by_root[root_id] = escalation

    commented: list[str] = []
    with kb.write_txn(conn):
        for root_id, escalation in sorted(by_root.items()):
            marker = f"[blocker-controller:{escalation['escalation_key']}]"
            existing = kb.list_comments(conn, root_id)
            if any(marker in comment.body for comment in existing):
                continue
            owner = escalation.get("owner") or "unassigned"
            reason = str(escalation.get("reason") or "no explicit block reason recorded")[:500]
            body = (
                f"{marker} Owner: @{owner}. Changed condition: {reason}. "
                "Next permitted action: verify that condition and use resolve-block with evidence. "
                "This controller does not unblock or create downstream tasks."
            )
            kb.add_comment(conn, root_id, author, body)
            commented.append(root_id)

    dispatched = False
    if ready_starved:
        if dispatch_once_fn is None:
            from hermes_cli import kanban_db_dispatch as kbd
            dispatch_once_fn = kbd.dispatch_once
        dispatch_once_fn(conn, max_spawn=1)
        dispatched = True
    return {"commented_root_ids": commented, "dispatch_requested": dispatched}


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Kanban blocker controller tick")
    parser.add_argument("--json", action="store_true", help="Emit JSON (the scheduler/notifier integration format)")
    parser.add_argument("--now", type=int, help="Epoch time for deterministic replay")
    parser.add_argument("--apply", action="store_true", help="Post idempotent root wake-up comments and dispatch one starved Ready card")
    parser.add_argument("--stale-gate-seconds", type=int, default=DEFAULT_STALE_GATE_SECONDS)
    parser.add_argument("--ready-starvation-seconds", type=int, default=DEFAULT_READY_STARVATION_SECONDS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    with kbc.connect_closing() as conn:
        report = controller_tick(
            conn, now=args.now, stale_gate_seconds=args.stale_gate_seconds,
            ready_starvation_seconds=args.ready_starvation_seconds,
        )
        if args.apply:
            report["applied"] = apply_tick(conn, report)
    print(json.dumps(report, sort_keys=True) if args.json else json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

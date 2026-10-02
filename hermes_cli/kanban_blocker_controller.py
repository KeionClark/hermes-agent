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


def _blocked_root_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """Return distinct blocked admission roots reachable through parent links."""
    rows = conn.execute(
        """
        WITH RECURSIVE upstream(id) AS (
            SELECT id FROM tasks WHERE id = ?
            UNION
            SELECT links.parent_id FROM task_links AS links JOIN upstream ON links.child_id = upstream.id
        )
        SELECT tasks.id FROM tasks JOIN upstream ON upstream.id = tasks.id
         WHERE tasks.status = 'blocked' AND tasks.block_kind IN (?, ?)
         ORDER BY tasks.id
        """, (task_id, *ROOT_BLOCK_KINDS),
    ).fetchall()
    return [str(row["id"]) for row in rows]


def _root_gate(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    """Return a deterministic outer blocked ancestor suitable for one resolver.

    ``UNION`` is deliberate: dependency diamonds and historical cycles must not
    multiply recursive paths. A descendant's priority belongs to the outermost
    blocked gate, whose creator lineage has only that one admission root.
    """
    candidates = _blocked_root_ids(conn, task_id)
    outer = [
        root_id for root_id in candidates
        if not (set(_blocked_root_ids(conn, root_id)) - {root_id})
    ]
    selected = min(outer or candidates, default=task_id)
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (selected,)).fetchone()
    if row is None:
        raise ValueError(f"unknown task {task_id}")
    return row

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
    def effective_priority(root_id: str) -> int:
        return max(int(row["priority"] or 0) for row in grouped[root_id])

    for root_id in sorted(grouped, key=lambda tid: (-effective_priority(tid), tid)):
        root = roots[root_id]
        priority = effective_priority(root_id)
        last_check_at, reason = _gate_activity(conn, root_id)
        interval = 60 * 60 if priority >= 90 else 6 * 60 * 60
        next_check_at = last_check_at + interval
        member_ids = [row["id"] for row in grouped[root_id]]
        entry = {
            "root_gate_id": root_id,
            "owner": root["assignee"],
            "reason": reason,
            "block_kind": root["block_kind"],
            "effective_priority": priority,
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
                "severity": "critical" if priority >= 90 else "warning",
                "root_gate_id": root_id,
                "owner": root["assignee"],
                "reason": reason,
                "effective_priority": priority,
                "next_check_at": next_check_at,
                "action": "verify_changed_condition",
                "command": f"hermes kanban show {root_id}",
                "escalation_key": f"gate-check-due:{root_id}:{last_check_at}",
            })
        if now >= last_check_at + stale_gate_seconds:
            escalations.append({
                "kind": "stale_gate",
                "severity": "critical" if priority >= 90 else "warning",
                "root_gate_id": root_id,
                "owner": root["assignee"],
                "reason": reason,
                "effective_priority": priority,
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


def _condition_revision(conn: sqlite3.Connection, root_id: str) -> str:
    """Only the root owner may advance a terminal resolver's retry identity."""
    root = kb.get_task(conn, root_id)
    owner = root.assignee if root else None
    if not owner:
        return "initial"
    for comment in reversed(kb.list_comments(conn, root_id)):
        if comment.author == owner and "[changed-condition]" in comment.body.casefold():
            return f"comment-{comment.id}"
    return "initial"


def _active_resolver_id(conn: sqlite3.Connection, root_id: str) -> Optional[str]:
    """Return an unfinished resolver linked to this root by its durable create event."""
    for event in conn.execute("SELECT task_id, payload FROM task_events WHERE kind = 'created' ORDER BY id DESC"):
        resolver = _event_payload(event).get("blocked_root_resolver")
        if not isinstance(resolver, dict) or resolver.get("root") != root_id:
            continue
        task = kb.get_task(conn, event["task_id"])
        if task is not None and task.status not in {"done", "archived"}:
            return task.id
    return None


def _resolver_purpose(conn: sqlite3.Connection, root_id: str, reason: str) -> str:
    return "verify changed condition " + _condition_revision(conn, root_id) + ": " + " ".join(reason.split())[:90]


def _workspace_capability(conn: sqlite3.Connection, root_id: str) -> bool:
    row = conn.execute("SELECT last_failure_error FROM tasks WHERE id = ?", (root_id,)).fetchone()
    text = str(row["last_failure_error"] or "") if row else ""
    text += " " + " ".join(str(_event_payload(event).get("reason") or "")
                              for event in conn.execute("SELECT payload FROM task_events WHERE task_id = ?", (root_id,)))
    lowered = text.casefold()
    return any(token in lowered for token in ("workspace", "checkout", "no such file", "route", "repository"))


def apply_tick(conn: sqlite3.Connection, report: dict[str, Any], *, author: str = "blocker-controller",
               max_resolvers: int = 1) -> dict[str, Any]:
    """Perform one bounded wake-up: one resolver and targeted Ready alerts.

    A resolver is a runnable, idempotent task under the exact blocked-root
    admission contract. It carries lineage only, never a blocking parent edge.
    No elapsed time path unblocks a card or recreates a downstream graph.
    """
    due: dict[str, dict[str, Any]] = {}
    ready: list[dict[str, Any]] = []
    for escalation in report.get("escalations", []):
        if escalation.get("kind") == "ready_starvation":
            ready.append(escalation)
        elif escalation.get("kind") in {"gate_check_due", "stale_gate"}:
            root_id = escalation["root_gate_id"]
            if root_id not in due or escalation["kind"] == "stale_gate":
                due[root_id] = escalation

    resolver_ids: list[str] = []
    commented_ready_ids: list[str] = []
    ordered = sorted(
        due.values(),
        key=lambda item: (
            item["severity"] != "critical",
            -int(item.get("effective_priority") or 0),
            item["kind"] != "stale_gate",
            item["root_gate_id"],
        ),
    )
    for escalation in ordered[:max_resolvers]:
        root_id = escalation["root_gate_id"]
        root = kb.get_task(conn, root_id)
        if root is None or not root.assignee:
            continue
        reason = str(escalation.get("reason") or "no explicit block reason recorded")
        if root.block_kind not in ROOT_BLOCK_KINDS:
            if not _workspace_capability(conn, root_id):
                continue
            kb.block_task(conn, root_id, reason="capability: verify workspace/repository route", kind="capability")
            root = kb.get_task(conn, root_id)
        if _active_resolver_id(conn, root_id) is not None:
            continue
        purpose = _resolver_purpose(conn, root_id, reason)
        key = kb.blocked_root_resolver_idempotency_key(root_id, purpose)
        existing = conn.execute("SELECT id, status FROM tasks WHERE idempotency_key = ?", (key,)).fetchone()
        if existing is not None and existing["status"] in {"done", "archived"}:
            continue
        resolver_ids.append(kb.create_task(
            conn, title=f"Resolve gate: {root.title}",
            body=("Verify this specific changed condition before resolving the root: " + reason
                  + f"\n\nBefore completing, @{root.assignee} must post `[changed-condition] <specific evidence>` "
                    f"on root {root_id}. Without that owner receipt, this gate cannot retry."),
            assignee=root.assignee, priority=max(90, int(escalation.get("effective_priority") or 0)),
            created_by=author, creator_task_id=root_id,
            blocked_root_resolver_for=root_id, blocked_root_resolver_purpose=purpose,
            idempotency_key=key,
        ))

    with kb.write_txn(conn):
        for escalation in ready:
            task_id = escalation["task_id"]
            marker = f"[blocker-controller:{escalation['escalation_key']}]"
            if any(marker in comment.body for comment in kb.list_comments(conn, task_id)):
                continue
            kb.add_comment(conn, task_id, author,
                           f"{marker} Ready starvation recorded for @{escalation.get('owner') or 'unassigned'}. "
                           "Targeted action: inspect this card's dispatcher/profile capacity; no unrelated card was dispatched.")
            commented_ready_ids.append(task_id)
    return {"resolver_task_ids": resolver_ids, "ready_alert_task_ids": commented_ready_ids}


def systemd_units() -> dict[str, str]:
    """Installable, bounded 15-minute controller units; returned for review first."""
    return {
        "service": """[Unit]
Description=Hermes Kanban blocker controller

[Service]
Type=oneshot
User=hermes
WorkingDirectory=/var/lib/hermes/.hermes/hermes-agent
ExecStart=/var/lib/hermes/.local/bin/hermes --run-module hermes_cli.kanban_blocker_controller --apply --json
NoNewPrivileges=true
""",
        "timer": """[Unit]
Description=Run Hermes Kanban blocker controller every 15 minutes

[Timer]
OnBootSec=2m
OnActiveSec=2m
OnUnitActiveSec=2m
RandomizedDelaySec=60
Persistent=true

[Install]
WantedBy=timers.target
""",
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Kanban blocker controller tick")
    parser.add_argument("--json", action="store_true", help="Emit JSON (the scheduler/notifier integration format)")
    parser.add_argument("--now", type=int, help="Epoch time for deterministic replay")
    parser.add_argument("--apply", action="store_true", help="Create at most one idempotent root resolver and targeted Ready alerts")
    parser.add_argument("--print-systemd", action="store_true", help="Print reviewed service/timer text without installing it")
    parser.add_argument("--stale-gate-seconds", type=int, default=DEFAULT_STALE_GATE_SECONDS)
    parser.add_argument("--ready-starvation-seconds", type=int, default=DEFAULT_READY_STARVATION_SECONDS)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.print_systemd:
        print(json.dumps(systemd_units(), sort_keys=True))
        return 0
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

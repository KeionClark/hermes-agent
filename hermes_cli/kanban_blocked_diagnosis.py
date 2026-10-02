"""Opt-in, event-ID based blocked-card diagnosis; no automatic unblock or graph writes.

The watcher is a bounded poll of the canonical task_events stream, NOT a push
subscription. Each board owns its own cursor and receipts. Enabling it requires
an explicit scoped migration/activation ruling; importing this module is inert.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from hermes_cli import kanban_db as kb

AUTHOR = "blocked-diagnosis"
IDLE_SECONDS = 4 * 3600
# A tick must finish in bounded time even on a long-lived board.
BATCH_SIZE = 100
# Metadata is bounded from the block for unresolved receipts, and from the
# transition for resolved outcomes. This ledger is not an audit archive.
RETENTION_SECONDS = 30 * 86400


def migrate(conn: sqlite3.Connection) -> None:
    """Additive, explicitly invoked board-local migration; no reason/PII ledger."""
    with kb.write_txn(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS blocked_diagnosis_cursor (singleton INTEGER PRIMARY KEY CHECK(singleton=1), event_id INTEGER NOT NULL, candidate_event_id INTEGER NOT NULL DEFAULT 0)")
        conn.execute("CREATE TABLE IF NOT EXISTS blocked_diagnosis_receipts (task_id TEXT NOT NULL, blocked_event_id INTEGER NOT NULL, action TEXT NOT NULL, created_at INTEGER NOT NULL, blocked_at INTEGER NOT NULL DEFAULT 0, resolved_at INTEGER, transition_event_id INTEGER, PRIMARY KEY(task_id, blocked_event_id, action))")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(blocked_diagnosis_cursor)")}
        if "candidate_event_id" not in columns:
            conn.execute("ALTER TABLE blocked_diagnosis_cursor ADD COLUMN candidate_event_id INTEGER NOT NULL DEFAULT 0")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(blocked_diagnosis_receipts)")}
        for name, sql in (("blocked_at", "INTEGER NOT NULL DEFAULT 0"), ("resolved_at", "INTEGER"),
                          ("transition_event_id", "INTEGER")):
            if name not in columns:
                conn.execute(f"ALTER TABLE blocked_diagnosis_receipts ADD COLUMN {name} {sql}")
        # Pre-upgrade receipts keep their original event's timestamp, not their
        # later processing timestamp. Do not alter existing outcome semantics.
        conn.execute("UPDATE blocked_diagnosis_receipts SET blocked_at = "
                     "COALESCE((SELECT created_at FROM task_events WHERE id = blocked_event_id), created_at) "
                     "WHERE blocked_at = 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_blocked_diagnosis_receipts_age "
                     "ON blocked_diagnosis_receipts(action, blocked_at, resolved_at)")
        conn.execute("INSERT OR IGNORE INTO blocked_diagnosis_cursor (singleton, event_id) VALUES (1, 0)")


def _payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        value = json.loads(row["payload"] or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _parents(conn: sqlite3.Connection, task_id: str) -> list[dict[str, str]]:
    return [dict(row) for row in conn.execute(
        "SELECT l.parent_id AS id, COALESCE(t.status, 'missing') AS status, t.completed_at FROM task_links l "
        "LEFT JOIN tasks t ON t.id = l.parent_id WHERE l.child_id = ? ORDER BY l.parent_id", (task_id,))]


def _cycles(conn: sqlite3.Connection, task_id: str) -> list[list[str]]:
    # Walk edges without traversing a node twice in one path; report actual
    # edges on a path back to the original card, never rewrite them.
    result: list[list[str]] = []
    def visit(node: str, path: list[str]) -> None:
        if len(path) > 100:
            return
        for parent in _parents(conn, node):
            ident = parent["id"]
            if ident == task_id:
                result.append(path + [ident])
            elif ident not in path:
                visit(ident, path + [ident])
    visit(task_id, [task_id])
    return result


def _owner_marker(conn: sqlite3.Connection, task_id: str, owner: str | None, blocked_at: int) -> bool:
    """Only a post-block owner-authored explicit marker is admissible, not approval."""
    if not owner:
        return False
    return any(row["body"].startswith("[changed-condition] ") for row in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? AND author = ? AND created_at >= ? "
        "ORDER BY id DESC", (task_id, owner, blocked_at)))


def _activity(conn: sqlite3.Connection, task_id: str, blocked_at: int) -> int:
    row = conn.execute(
        "SELECT MAX(created_at) AS ts FROM task_events WHERE task_id = ? AND created_at >= ? "
        "AND kind != 'commented'", (task_id, blocked_at)).fetchone()
    comment = conn.execute(
        "SELECT MAX(created_at) AS ts FROM task_comments WHERE task_id = ? AND created_at >= ? "
        "AND author NOT IN ('blocked-diagnosis', 'blocker-controller')", (task_id, blocked_at)).fetchone()
    return max(blocked_at, row["ts"] or 0, comment["ts"] or 0)


def _diagnose(conn: sqlite3.Connection, row: sqlite3.Row, now: int) -> dict[str, Any]:
    task = conn.execute("SELECT * FROM tasks WHERE id = ?", (row["task_id"],)).fetchone()
    parents = _parents(conn, row["task_id"])
    cycles = _cycles(conn, row["task_id"])
    kind = str(_payload(row).get("kind") or (task["block_kind"] if task else "") or "")
    owner = task["assignee"] if task else None
    route = {
        "dependency": "wait_for_all_live_parents_done_and_link_replacement_for_archived_parent",
        "needs_input": "owner_verify_authoritative_input_no_auto_approval",
        "capability": "owner_verify_access_or_route_no_auto_approval",
        "transient": "owner_verify_changed_condition_before_one_bounded_retry",
    }.get(kind, "owner_classify_block")
    if cycles:
        route = "owner_review_cycle_edges_no_auto_unlink"
    elif any(p["status"] == "missing" or
             (p["status"] == "archived" and p["completed_at"] is None) for p in parents):
        route = "owner_link_replacement_for_unfulfilled_archived_parent"
    elif kind == "dependency" and any(p["status"] != "done" and
                                      not (p["status"] == "archived" and p["completed_at"] is not None)
                                      for p in parents):
        route = "wait_for_parent_completion"
    elif kind == "transient" and task and int(task["block_recurrences"] or 0) > 1:
        route = "owner_review_retry_cap_no_auto_retry"
    fresh = _owner_marker(conn, row["task_id"], owner, row["created_at"])
    return {"task_id": row["task_id"], "blocked_event_id": row["id"], "blocked_at": row["created_at"],
            "kind": kind, "status": task["status"] if task else "missing", "owner": owner,
            "parents": parents, "cycle_edges": cycles, "route": route,
            "owner_evidence_marker": fresh, "evidence": "post-block owner marker; verify original source" if fresh else "original blocked event; changed condition unverified",
            "event_to_action_seconds": max(0, now - row["created_at"]),
            "dispatch_mode": "bounded_poll_not_immediate",
            "idle_seconds": max(0, now - _activity(conn, row["task_id"], row["created_at"])),
            "historical_sweep_at": None, "historical_sweep_latency_seconds": None}


def _receipt(conn: sqlite3.Connection, diagnosis: dict[str, Any], action: str, now: int,
             outcome: dict[str, Any] | None = None) -> bool:
    task_id, event_id = diagnosis["task_id"], diagnosis["blocked_event_id"]
    inserted = conn.execute(
        "INSERT OR IGNORE INTO blocked_diagnosis_receipts "
        "(task_id, blocked_event_id, action, created_at, blocked_at, resolved_at, transition_event_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (task_id, event_id, action, now, diagnosis["blocked_at"],
         outcome["resolved_at"] if outcome else None,
         outcome["transition_event_id"] if outcome else None)).rowcount
    if inserted and (
        (action == "diagnose" and diagnosis["status"] in {"blocked", "todo"})
        or (action == "idle-4h" and diagnosis["status"] in {"blocked", "todo"})
    ):
        kb.add_comment(conn, task_id, AUTHOR,
            f"[blocked-diagnosis:{event_id}:{action}] @{diagnosis['owner'] or 'unassigned'}: "
            f"{diagnosis['route']}. Evidence: {diagnosis['evidence']}. "
            "No approval, unblock, new resolver card or graph edit is implied.")
    return bool(inserted)


def _outcome(conn: sqlite3.Connection, row: sqlite3.Row, route: str) -> dict[str, Any] | None:
    events = conn.execute("SELECT id, kind, payload, created_at FROM task_events WHERE task_id = ? AND id > ? ORDER BY id",
                          (row["task_id"], row["id"])).fetchall()
    for event in events:
        if event["kind"] in {"blocked", "dependency_wait", "block_loop_detected"}:
            break
        # Do not mistake blocked -> todo for resolution; only a real ready/running transition.
        if event["kind"] in {"unblocked", "promoted", "claimed", "status"}:
            payload = _payload(event)
            destination = payload.get("to_status") or payload.get("status")
            if event["kind"] == "unblocked" and not payload:
                destination = "ready"
            if event["kind"] == "promoted":
                destination = payload.get("status", "ready")
            if event["kind"] == "claimed":
                destination = "running"
            if destination in {"ready", "running"}:
                return {"task_id": row["task_id"], "blocked_event_id": row["id"],
                        "blocked_at": row["created_at"], "resolved_at": event["created_at"],
                        "route": route, "transition_event_id": event["id"]}
    return None


def consume(conn: sqlite3.Connection, *, now: int | None = None, limit: int = BATCH_SIZE) -> dict[str, Any]:
    """One bounded transaction; migration must already exist. Replay is idempotent."""
    if not 1 <= limit <= BATCH_SIZE:
        raise ValueError("limit outside bounded range")
    now = int(time.time()) if now is None else int(now)
    diagnoses, outcomes, alerts = [], [], []
    with kb.write_txn(conn):
        cursor = conn.execute("SELECT event_id, candidate_event_id FROM blocked_diagnosis_cursor WHERE singleton=1").fetchone()
        if cursor is None:
            raise RuntimeError("blocked diagnosis migration/activation required")
        # Purge by event age even when the card remains blocked forever. A
        # bounded batch per tick avoids holding the writer lock over the board.
        expired = conn.execute(
            "SELECT task_id, blocked_event_id FROM blocked_diagnosis_receipts "
            "WHERE (action = 'resolved' AND resolved_at <= ?) OR "
            "(action != 'resolved' AND blocked_at <= ?) LIMIT ?",
            (now - RETENTION_SECONDS, now - RETENTION_SECONDS, limit)).fetchall()
        for old in expired:
            conn.execute("DELETE FROM blocked_diagnosis_receipts WHERE task_id=? AND blocked_event_id=? "
                         "AND ((action='resolved' AND resolved_at <= ?) OR "
                         "(action!='resolved' AND blocked_at <= ?))",
                         (old["task_id"], old["blocked_event_id"],
                          now - RETENTION_SECONDS, now - RETENTION_SECONDS))
        rows = conn.execute("SELECT id, task_id, kind, payload, created_at FROM task_events WHERE id > ? ORDER BY id LIMIT ?",
                            (cursor["event_id"], limit)).fetchall()
        for event in rows:
            if event["kind"] in {"blocked", "block_loop_detected"} or (
                event["kind"] == "dependency_wait" and _payload(event).get("kind") == "dependency"
            ):
                diagnosis = _diagnose(conn, event, now)
                diagnoses.append(diagnosis)
                if event["created_at"] > now - RETENTION_SECONDS:
                    _receipt(conn, diagnosis, "diagnose", now)
            conn.execute("UPDATE blocked_diagnosis_cursor SET event_id = ? WHERE singleton=1", (event["id"],))
        # Only previously diagnosed blocks, bounded by the receipt table and live state.
        def candidate_batch(after: int) -> list[sqlite3.Row]:
            return conn.execute(
                "SELECT e.id, e.task_id, e.kind, e.payload, e.created_at FROM blocked_diagnosis_receipts r "
                "JOIN task_events e ON e.id = r.blocked_event_id AND e.task_id = r.task_id "
                "WHERE r.action = 'diagnose' AND e.id > ? AND r.blocked_at > ? "
                "AND NOT EXISTS (SELECT 1 FROM blocked_diagnosis_receipts x WHERE "
                "x.task_id=r.task_id AND x.blocked_event_id=r.blocked_event_id "
                "AND x.action IN ('resolved', 'closed')) ORDER BY e.id LIMIT ?",
                (after, now - RETENTION_SECONDS, limit)).fetchall()
        candidates = candidate_batch(cursor["candidate_event_id"])
        if not candidates:
            candidates = candidate_batch(0)
        for row in candidates:
            diagnosis = _diagnose(conn, row, now)
            if diagnosis["status"] in {"blocked", "todo"} and diagnosis["idle_seconds"] >= IDLE_SECONDS:
                if _receipt(conn, diagnosis, "idle-4h", now):
                    alerts.append(diagnosis)
            outcome = _outcome(conn, row, diagnosis["route"])
            if outcome and _receipt(conn, diagnosis, "resolved", now, outcome):
                outcomes.append(outcome)
            elif diagnosis["status"] in {"archived", "done", "missing"} or conn.execute(
                "SELECT 1 FROM task_events WHERE task_id = ? AND id > ? AND kind IN "
                "('blocked', 'dependency_wait', 'block_loop_detected') LIMIT 1",
                (row["task_id"], row["id"])).fetchone():
                _receipt(conn, diagnosis, "closed", now)
        if candidates:
            conn.execute("UPDATE blocked_diagnosis_cursor SET candidate_event_id=? WHERE singleton=1",
                         (candidates[-1]["id"],))
    return {"cursor": rows[-1]["id"] if rows else cursor["event_id"],
            "diagnoses": diagnoses, "idle_alerts": alerts, "outcomes": outcomes}

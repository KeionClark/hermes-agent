"""Opt-in, evidence-backed proposal queue. No proposal is executable by this module.

The goal and event-bus schemas are migrated separately. Call migrate_proposal_schema
only on an explicitly opted-in disposable board; this module never opens a board.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import time
import unicodedata
import uuid
from dataclasses import dataclass

_EXPIRY_SECONDS = 7 * 86400
_CAP = 10
_PRIOR = 0.3


@dataclass(frozen=True, slots=True)
class Proposal:
    id: str
    goal_id: str
    metric_id: str
    gap_measurement_id: int
    action_type: str
    hypothesis: str
    proposed_action: str
    impact: float
    confidence: float
    cost: float
    score: float
    decision_tier: str
    idempotency_key: str
    status: str
    created_at: float
    expires_at: float


def _normalized(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("identity must be text")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if not normalized:
        raise ValueError("identity cannot be empty")
    return normalized


def _factor(name: str, value: float, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not minimum <= value <= 1:
        raise ValueError(f"{name} must be between {minimum} and 1")
    return float(value)


def _row(row: sqlite3.Row) -> Proposal:
    return Proposal(**{name: row[name] for name in Proposal.__dataclass_fields__})


def migrate_proposal_schema(conn: sqlite3.Connection) -> None:
    """Explicit atomic migration, independent of the canonical connect path."""
    if conn.in_transaction:
        raise RuntimeError("migration requires an autocommit connection")
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS gap_proposals (
            id TEXT PRIMARY KEY, goal_id TEXT NOT NULL REFERENCES goals(id),
            metric_id TEXT NOT NULL, gap_measurement_id INTEGER NOT NULL REFERENCES goal_measurements(id),
            action_type TEXT NOT NULL, hypothesis TEXT NOT NULL, proposed_action TEXT NOT NULL,
            identity_hash TEXT NOT NULL, impact REAL NOT NULL, confidence REAL NOT NULL,
            requested_confidence REAL NOT NULL,
            cost REAL NOT NULL, score REAL NOT NULL, decision_tier TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE, status TEXT NOT NULL
                CHECK(status IN ('active','expired')),
            created_at REAL NOT NULL, expires_at REAL NOT NULL)""")
        conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS uq_active_proposal_identity
            ON gap_proposals(identity_hash) WHERE status = 'active'""")
        conn.execute("""CREATE INDEX IF NOT EXISTS idx_proposals_goal
            ON gap_proposals(goal_id, status, expires_at, score DESC)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS proposal_confidence_adjustments (
            outcome_id TEXT PRIMARY KEY, action_type TEXT NOT NULL,
            delta REAL NOT NULL, applied INTEGER NOT NULL CHECK(applied IN (0,1)))""")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def propose_for_gap(
    conn: sqlite3.Connection, *, goal_id: str, metric_id: str,
    gap_measurement_id: int, action_type: str, hypothesis: str,
    proposed_action: str, impact: float, confidence: float, cost: float,
    decision_tier: str, idempotency_key: str, now: float,
) -> Proposal:
    """Deduplicate and enqueue a fresh, latest breached measurement; never Act."""
    impact = _factor("impact", impact)
    confidence = _factor("confidence", confidence)
    cost = _factor("cost", cost, minimum=0.1)
    if not isinstance(now, (int, float)) or not math.isfinite(now):
        raise ValueError("now must be finite")
    if type(gap_measurement_id) is not int or gap_measurement_id <= 0:
        raise ValueError("gap_measurement_id must be a positive integer")
    kind = _normalized(action_type)
    hypothesis_identity = _normalized(hypothesis)
    for value in (goal_id, metric_id, proposed_action, decision_tier, idempotency_key):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("required proposal field missing")
    identity_hash = hashlib.sha256(
        repr((goal_id, kind, hypothesis_identity)).encode("utf-8")
    ).hexdigest()
    if conn.in_transaction:
        raise RuntimeError("propose_for_gap requires an autocommit connection")
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Keep retry identities forever, including expired rows. A reused key with
        # different content is an error, not an update of an old proposal.
        existing = conn.execute(
            "SELECT * FROM gap_proposals WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if existing is not None:
            if (existing["identity_hash"], existing["goal_id"], existing["metric_id"],
                existing["gap_measurement_id"], existing["proposed_action"],
                existing["impact"], existing["requested_confidence"], existing["cost"],
                existing["decision_tier"]) != (
                identity_hash, goal_id, metric_id, gap_measurement_id,
                proposed_action, impact, confidence, cost, decision_tier
            ):
                raise ValueError("conflicting proposal idempotency replay")
            result = _row(existing)
            conn.execute("COMMIT")
            return result
        conn.execute("UPDATE gap_proposals SET status = 'expired' "
                     "WHERE status = 'active' AND expires_at <= ?", (now,))
        row = conn.execute("""SELECT g.state, g.metric_id, g.source_id,
                g.max_age_seconds, g.metric_version, m.metric_id AS measurement_metric,
                m.source_id AS measurement_source, m.source_timestamp,
                m.observed_at, m.source_ref, m.payload_hash, m.edge_state,
                (SELECT MAX(id) FROM goal_measurements WHERE goal_id = g.id) AS latest_id
            FROM goals g JOIN goal_measurements m ON m.goal_id = g.id AND m.id = ?
            WHERE g.id = ?""", (gap_measurement_id, goal_id)).fetchone()
        if (row is None or row["state"] != "active" or row["metric_id"] != metric_id
                or row["measurement_metric"] != metric_id
                or row["measurement_source"] != row["source_id"]
                or row["edge_state"] != "breached"
                or row["latest_id"] != gap_measurement_id
                or (row["source_ref"] is None and row["payload_hash"] is None)
                or not math.isfinite(row["source_timestamp"])
                or not math.isfinite(row["observed_at"])
                or row["source_timestamp"] > now
                or now - row["source_timestamp"] > row["max_age_seconds"]):
            raise ValueError("no current provenance-backed active gap")
        # V1 measurements have no version snapshot. A changed goal cannot
        # prove that a prior measurement used its current metric definition.
        if row["metric_version"] != 1:
            raise ValueError("gap measurement metric version cannot be proven")
        duplicate = conn.execute("SELECT * FROM gap_proposals WHERE identity_hash = ? "
                                 "AND status = 'active'", (identity_hash,)).fetchone()
        if duplicate is not None:
            result = _row(duplicate)
            conn.execute("COMMIT")
            return result
        count = conn.execute("SELECT COUNT(*) FROM gap_proposals WHERE goal_id = ? "
                             "AND status = 'active'", (goal_id,)).fetchone()[0]
        if count >= _CAP:
            raise ValueError("active proposal capacity reached")
        # The caller's confidence is evidence-weighted, bounded by the
        # uncalibrated action prior until evaluated outcomes establish a record.
        calibrated = min(confidence, _action_prior(conn, kind))
        score = impact * calibrated / max(cost, 0.1)
        proposal_id = str(uuid.uuid4())
        conn.execute("""INSERT INTO gap_proposals
            (id,goal_id,metric_id,gap_measurement_id,action_type,hypothesis,
             proposed_action,identity_hash,impact,confidence,requested_confidence,cost,score,
             decision_tier,idempotency_key,status,created_at,expires_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'active',?,?)""",
            (proposal_id, goal_id, metric_id, gap_measurement_id, kind, hypothesis,
             proposed_action, identity_hash, impact, calibrated, confidence, cost, score,
             decision_tier, idempotency_key, now, now + _EXPIRY_SECONDS))
        result = _row(conn.execute("SELECT * FROM gap_proposals WHERE id = ?",
                                   (proposal_id,)).fetchone())
        conn.execute("COMMIT")
        return result
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def list_proposals(conn: sqlite3.Connection, *, goal_id: str | None = None,
                   now: float | None = None, limit: int = 10) -> list[Proposal]:
    """Read only active, unexpired proposals ordered by bounded score."""
    if now is None:
        now = time.time()
    if not isinstance(now, (int, float)) or not math.isfinite(now):
        raise ValueError("now must be finite")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    if goal_id is None:
        rows = conn.execute("SELECT * FROM gap_proposals WHERE status = 'active' "
                            "AND expires_at > ? ORDER BY score DESC, created_at, id LIMIT ?",
                            (now, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM gap_proposals WHERE status = 'active' "
                            "AND goal_id = ? AND expires_at > ? "
                            "ORDER BY score DESC, created_at, id LIMIT ?",
                            (goal_id, now, limit)).fetchall()
    return [_row(row) for row in rows]


def _action_prior(conn: sqlite3.Connection, action_type: str) -> float:
    row = conn.execute("SELECT COALESCE(SUM(delta), 0) FROM proposal_confidence_adjustments "
                       "WHERE action_type = ? AND applied = 1", (action_type,)).fetchone()
    return max(0.0, min(1.0, _PRIOR + row[0]))


def _rescore(conn: sqlite3.Connection, action_type: str) -> None:
    prior = _action_prior(conn, action_type)
    conn.execute("""UPDATE gap_proposals SET
        confidence = MIN(requested_confidence, ?),
        score = impact * MIN(requested_confidence, ?) / MAX(cost, 0.1)
        WHERE action_type = ? AND status = 'active'""", (prior, prior, action_type))


def apply_evaluated_outcome(conn: sqlite3.Connection, outcome_id: str) -> float | None:
    """Apply one comparable outcome to its action type's bounded prior.

    Fail closed on legacy measurements without provable version continuity,
    absent before/after evidence or mismatched proposal identity. The ledger
    persists the verdict; this queue never fabricates or evaluates an outcome.
    """
    if conn.in_transaction:
        raise RuntimeError("confidence adjustment requires an autocommit connection")
    conn.execute("BEGIN IMMEDIATE")
    try:
        prior = _apply_locked(conn, outcome_id)
        conn.execute("COMMIT")
        return prior
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _apply_locked(conn: sqlite3.Connection, outcome_id: str) -> float | None:
    outcome = conn.execute("SELECT * FROM outcomes WHERE id = ?", (outcome_id,)).fetchone()
    if outcome is None or outcome["status"] not in ("confirmed", "falsified"):
        return None
    action = _normalized(outcome["action_type"])
    applied = conn.execute("SELECT * FROM proposal_confidence_adjustments "
                           "WHERE outcome_id = ?", (outcome_id,)).fetchone()
    if applied is not None:
        return _action_prior(conn, action) if applied["applied"] else None
    before = outcome["before_measurement_id"]
    after = outcome["after_measurement_id"]
    if before is None or after is None or before == after:
        return None
    goal = conn.execute("SELECT metric_version, metric_id, source_id, max_age_seconds FROM goals WHERE id = ?",
                        (outcome["goal_id"],)).fetchone()
    # V1 goal_measurements contains no version snapshot. Only a never-changed
    # metric definition (version 1) can be proved from this schema.
    if (goal is None or goal["metric_version"] != 1
            or outcome["goal_metric_version"] != goal["metric_version"]):
        return None
    readings = conn.execute("SELECT * FROM goal_measurements WHERE id IN (?, ?)",
                            (before, after)).fetchall()
    if len(readings) != 2:
        return None
    by_id = {r["id"]: r for r in readings}
    for name, reading in (("before", by_id[before]), ("after", by_id[after])):
        if (reading["goal_id"] != outcome["goal_id"]
                or reading["metric_id"] != goal["metric_id"]
                or reading["source_id"] != goal["source_id"]
                or reading["value"] != outcome[f"{name}_value"]
                or reading["source_timestamp"] != outcome[f"{name}_source_timestamp"]
                or reading["source_ref"] != outcome[f"{name}_source_ref"]
                or reading["observed_at"] != outcome[f"{name}_observed_at"]):
            return None
    if (by_id[before]["source_timestamp"] > outcome["started_at"]
            or by_id[after]["source_timestamp"] <= outcome["completed_at"]
            or by_id[after]["source_timestamp"] < outcome["completed_at"] + outcome["settling_seconds"]
            or by_id[after]["source_timestamp"] > outcome["evaluated_at"]
            or outcome["evaluated_at"] - by_id[after]["source_timestamp"] > goal["max_age_seconds"]):
        return None
    movement = by_id[after]["value"] - by_id[before]["value"]
    confirmed = (movement > 0 if outcome["expected_direction"] == "higher"
                 else movement < 0 if outcome["expected_direction"] == "lower" else False)
    if (outcome["status"] == "confirmed") != confirmed:
        return None
    if not math.isfinite(movement):
        return None
    if (not isinstance(outcome["confidence_delta"], (int, float))
            or not math.isfinite(outcome["confidence_delta"]) or outcome["confidence_delta"] < 0):
        return None
    identity = hashlib.sha256(repr((outcome["goal_id"], action,
                                     _normalized(outcome["hypothesis"]))).encode()).hexdigest()
    if conn.execute("SELECT 1 FROM gap_proposals WHERE identity_hash = ?",
                    (identity,)).fetchone() is None:
        return None
    # Equal measurements falsify the hypothesis too: the ledger's absolute
    # effect is zero, but that comparable failure still lowers confidence.
    delta = min(0.1, max(0.01, outcome["confidence_delta"]))
    if outcome["status"] == "falsified":
        delta = -delta
    conn.execute("INSERT INTO proposal_confidence_adjustments "
                 "(outcome_id,action_type,delta,applied) VALUES (?,?,?,1)",
                 (outcome_id, action, delta))
    _rescore(conn, action)
    return _action_prior(conn, action)


def revert_evaluated_outcome(conn: sqlite3.Connection, outcome_id: str) -> float | None:
    """Subtract exactly the recorded adjustment once after ledger reversion."""
    if conn.in_transaction:
        raise RuntimeError("confidence adjustment requires an autocommit connection")
    conn.execute("BEGIN IMMEDIATE")
    try:
        outcome = conn.execute("SELECT status FROM outcomes WHERE id = ?", (outcome_id,)).fetchone()
        adjustment = conn.execute("SELECT action_type, applied FROM proposal_confidence_adjustments "
                                  "WHERE outcome_id = ?", (outcome_id,)).fetchone()
        result = None
        if outcome is not None and outcome["status"] == "reverted" and adjustment is not None:
            if adjustment["applied"]:
                conn.execute("UPDATE proposal_confidence_adjustments SET applied = 0 "
                             "WHERE outcome_id = ?", (outcome_id,))
                _rescore(conn, adjustment["action_type"])
            result = _action_prior(conn, adjustment["action_type"])
        conn.execute("COMMIT")
        return result
    except BaseException:
        conn.execute("ROLLBACK")
        raise

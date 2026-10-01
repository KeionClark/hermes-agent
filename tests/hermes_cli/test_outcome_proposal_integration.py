"""Real SQLite integration for dark outcome and proposal paths."""
import json
import sqlite3
import time

from hermes_cli.event_bus import migrate_event_bus
from hermes_cli.goal_metrics import activate_goal, create_goal, migrate_goal_schema, record_measurement
from hermes_cli.outcome_ledger import (complete_action, create_outcome, evaluate_due_outcome,
                                       migrate_outcome_schema, revert_outcome)
from hermes_cli.proposal_queue import (apply_evaluated_outcome, list_proposals,
                                       migrate_proposal_schema, revert_evaluated_outcome)
from hermes_cli.ooda_controller import propose_gap_once


def test_same_edge_remeasurement_atomic_replay_and_proposal(tmp_path):
    db = tmp_path / "board.db"
    def connect():
        conn = sqlite3.connect(db, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    a, b = connect(), connect()
    for migration in (migrate_event_bus, migrate_goal_schema, migrate_outcome_schema,
                      migrate_proposal_schema):
        migration(a)
    t = time.time() - 30
    a.execute("BEGIN IMMEDIATE")
    create_goal(a, goal_id="g", metric_id="m", source_id="s", target_value=100,
                max_age_seconds=3600)
    activate_goal(a, "g", "local-test", t)
    a.execute("COMMIT")

    def measure(conn, value, ts, key):
        return record_measurement(conn, goal_id="g", metric_id="m", value=value,
                                  source_id="s", source_timestamp=ts, observed_at=ts,
                                  source_ref=key, idempotency_key=key)

    a.execute("BEGIN IMMEDIATE")
    baseline = measure(a, 50, t, "before")
    a.execute("COMMIT")
    candidate = {"goal_id": "g", "metric_id": "m", "action_type": "fix",
                 "hypothesis": "improve", "proposed_action": "inspect only", "impact": 0.8,
                 "confidence": 0.9, "cost": 0.1, "decision_tier": "investigation",
                 "idempotency_key": "candidate-1"}
    assert propose_gap_once(b, subscriber_id="p2", candidates=[candidate], now=t+1) == 1
    first = list_proposals(a, goal_id="g", now=t+1)[0]
    assert first.cost >= 0.1 and first.score == first.impact * first.confidence / first.cost
    assert propose_gap_once(b, subscriber_id="p2", candidates=[candidate], now=t+1) == 0

    a.execute("BEGIN IMMEDIATE")
    outcome = create_outcome(a, task_id="task", goal_id="g", hypothesis="improve",
                             action_type="fix", action_revision="rev-start",
                             expected_direction="higher", before_measurement_id=baseline.measurement.id,
                             started_at=t+2, idempotency_key="action-1")
    complete_action(a, outcome.id, action_revision="rev-final", completed_at=t+3)
    a.execute("COMMIT")
    b.execute("BEGIN IMMEDIATE")
    after = measure(b, 75, t+4, "after")
    assert not any(e["event_type"] == "goal.gap_detected" for e in after.events)
    assert b.execute("SELECT status FROM outcomes WHERE id=?", (outcome.id,)).fetchone()[0] == "confirmed"
    assert a.execute("SELECT status FROM outcomes WHERE id=?", (outcome.id,)).fetchone()[0] == "awaiting_remeasure"
    b.execute("ROLLBACK")
    assert a.execute("SELECT status FROM outcomes WHERE id=?", (outcome.id,)).fetchone()[0] == "awaiting_remeasure"
    b.execute("BEGIN IMMEDIATE")
    measure(b, 75, t+4, "after")
    b.execute("COMMIT")
    assert a.execute("SELECT after_source_ref, action_revision, before_metric_version, "
                     "after_metric_version FROM outcomes WHERE id=?",
                     (outcome.id,)).fetchone()[:] == ("after", "rev-final", 1, 1)
    event = a.execute("SELECT payload FROM hermes_events WHERE idempotency_key=?",
                      (f"outcome_eval:{outcome.id}",)).fetchone()
    assert set(json.loads(event[0])) == {"goal_id", "task_id", "metric_id", "status"}
    a.execute("BEGIN IMMEDIATE")
    assert evaluate_due_outcome(a, outcome.id, now=t+5).status == "confirmed"
    a.execute("COMMIT")
    assert a.execute("SELECT count(*) FROM hermes_events WHERE idempotency_key=?",
                     (f"outcome_eval:{outcome.id}",)).fetchone()[0] == 1
    assert apply_evaluated_outcome(a, outcome.id) > 0.3
    assert apply_evaluated_outcome(b, outcome.id) > 0.3
    assert a.execute("SELECT count(*) FROM proposal_confidence_adjustments").fetchone()[0] == 1
    a.execute("BEGIN IMMEDIATE")
    revert_outcome(a, outcome.id, reason="test revert", now=t+6)
    a.execute("COMMIT")
    assert revert_evaluated_outcome(b, outcome.id) == 0.3
    assert revert_evaluated_outcome(a, outcome.id) == 0.3
    a.close()
    b.close()


def test_legacy_measurement_is_not_relabelled_by_v2_migration(tmp_path):
    from hermes_cli import goal_metrics
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db, isolation_level=None)
    conn.row_factory = sqlite3.Row
    migrate_event_bus(conn)
    # Exercise the real V1 migration SQL; the upgrade must preserve unknown
    # historical versions rather than filling in a goal's current version.
    conn.execute("BEGIN IMMEDIATE")
    for statement in goal_metrics._GOAL_SCHEMA_V1.split(";"):
        if statement.strip():
            conn.execute(statement)
    conn.execute("INSERT INTO goal_schema_version(id,version) VALUES (1,1)")
    conn.execute("COMMIT")
    migrate_goal_schema(conn)
    assert conn.execute("SELECT version FROM goal_schema_version").fetchone()[0] == 2
    conn.execute("BEGIN IMMEDIATE")
    create_goal(conn, goal_id="g", metric_id="m", source_id="s")
    activate_goal(conn, "g", "local-test", time.time())
    conn.execute("INSERT INTO goal_measurements(goal_id,metric_id,value,source_id,source_timestamp,"
                 "observed_at,source_ref,idempotency_key,created_at) "
                 "VALUES('g','m',1,'s',1,1,'legacy','legacy',1)")
    conn.execute("COMMIT")
    assert conn.execute("SELECT metric_version FROM goal_measurements").fetchone()[0] is None
    migrate_outcome_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    try:
        create_outcome(conn, task_id="task", goal_id="g", hypothesis="h", action_type="fix",
                       action_revision="rev", expected_direction="higher", before_measurement_id=1,
                       started_at=2, idempotency_key="legacy-action")
    except ValueError as exc:
        assert "version" in str(exc)
    else:
        raise AssertionError("legacy unversioned baseline was accepted")
    conn.execute("ROLLBACK")
    conn.close()

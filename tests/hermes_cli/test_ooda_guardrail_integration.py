"""Guarded OODA admission against a disposable, explicitly migrated board."""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import replace
from unittest.mock import patch

import pytest

from hermes_cli import ooda_guardrails
from hermes_cli.event_bus import poll, subscribe
from hermes_cli.ooda_controller import GuardrailPolicy, _OBSERVE_EVENTS, run_once
from tests.hermes_cli.test_ooda_controller import (
    _create_active_goal, _disposable_board, _record_breached_measurement,
)
from tests.hermes_cli.test_ooda_guardrails import (
    _create_test_goal, _create_test_measurement, _create_test_outcome,
)


def policy(**changes):
    base = GuardrailPolicy(
        agent_id="dark-agent", lane_id="dark-lane", estimated_tokens=100,
        price_per_million_usd="1", approval_ref="synthetic-approval-not-live",
        caps={"task_token": "100", "agent_day": "0.0001",
              "lane_week": "0.0001", "global_day": "0.0001"},
        drift_max_ratio=0.5,
    )
    return replace(base, **changes)


@pytest.fixture
def board():
    conn, path = _disposable_board()
    try:
        yield conn, path
    finally:
        conn.close()
        os.unlink(path)


def gap(conn):
    _create_active_goal(conn)
    subscribe(conn, "guarded", event_types=_OBSERVE_EVENTS, start="beginning")
    _record_breached_measurement(conn, "test-goal-1", idem_key="guarded-first")


def act(conn, path, p=None):
    return run_once(conn, subscriber_id="guarded", permitted_board=path,
                    guardrail_policy=p or policy())


def test_default_and_missing_schema_fail_before_subscription(board):
    conn, path = board
    with pytest.raises(ValueError, match="explicit guardrail policy"):
        run_once(conn, subscriber_id="guarded", permitted_board=path)
    with pytest.raises(ooda_guardrails.GuardrailSchemaNotInitializedError):
        act(conn, path)
    assert conn.execute("SELECT COUNT(*) FROM hermes_event_cursors").fetchone()[0] == 0


def test_atomic_reservation_act_ack_and_replay(board):
    conn, path = board
    ooda_guardrails.migrate_guardrail_schema(conn)
    from hermes_cli.outcome_ledger import migrate_outcome_schema
    migrate_outcome_schema(conn)
    gap(conn)
    result = act(conn, path)
    assert result.acts_created == 1
    task = conn.execute("SELECT id, idempotency_key FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'").fetchone()
    reservation = conn.execute("SELECT task_id, status, approval_ref FROM guardrail_reservations").fetchone()
    assert (reservation[0], reservation[1], reservation[2]) == (task[0], "outstanding", policy().approval_ref)
    assert poll(conn, "guarded", limit=10) == []
    subscribe(conn, "replay", event_types=_OBSERVE_EVENTS, start="beginning")
    assert run_once(conn, subscriber_id="replay", permitted_board=path,
                    guardrail_policy=policy()).acts_deduplicated == 1
    assert conn.execute("SELECT COUNT(*) FROM guardrail_reservations").fetchone()[0] == 1


def test_cap_crossing_halts_without_ack_or_second_act(board):
    conn, path = board
    ooda_guardrails.migrate_guardrail_schema(conn)
    from hermes_cli.outcome_ledger import migrate_outcome_schema
    migrate_outcome_schema(conn)
    gap(conn)
    assert act(conn, path).acts_created == 1
    conn.execute("UPDATE tasks SET status = 'done' WHERE idempotency_key LIKE 'ooda:investigate:%'")
    _record_breached_measurement(conn, "test-goal-1", value=3, idem_key="recovery")
    _record_breached_measurement(conn, "test-goal-1", idem_key="second-gap")
    denied = act(conn, path)
    assert denied.acts_created == 0
    assert any("guardrail_halt" in r for r in denied.noop_reasons)
    assert len(poll(conn, "guarded", limit=10)) == 1
    assert conn.execute("SELECT COUNT(*) FROM guardrail_reservations").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM guardrail_halts WHERE resolved_at IS NULL").fetchone()[0] == 1
    assert act(conn, path).acts_created == 0
    assert conn.execute("SELECT COUNT(*) FROM guardrail_halts WHERE resolved_at IS NULL").fetchone()[0] == 1


def test_missing_price_and_unapproved_caps_rejected(board):
    conn, path = board
    ooda_guardrails.migrate_guardrail_schema(conn)
    with pytest.raises(ValueError, match="dollar amount"):
        act(conn, path, policy(price_per_million_usd="0"))
    with pytest.raises(ValueError, match="four caps"):
        act(conn, path, policy(caps={"global_day": "1"}))
    assert conn.execute("SELECT COUNT(*) FROM guardrail_reservations").fetchone()[0] == 0


def test_incomplete_outcome_schema_fails_safe_and_retains_gap(board):
    conn, path = board
    ooda_guardrails.migrate_guardrail_schema(conn)
    gap(conn)
    result = act(conn, path)
    assert result.acts_created == 0
    assert result.noop_reasons[0].startswith("guardrail_halt:act_failure:")
    assert len(poll(conn, "guarded", limit=10)) == 1
    assert conn.execute("SELECT COUNT(*) FROM guardrail_reservations").fetchone()[0] == 0


def test_insert_failure_rolls_back_reservation_and_ack(board):
    conn, path = board
    ooda_guardrails.migrate_guardrail_schema(conn)
    from hermes_cli.outcome_ledger import migrate_outcome_schema
    migrate_outcome_schema(conn)
    gap(conn)
    with patch("hermes_cli.ooda_controller._act_create_investigation", side_effect=RuntimeError("insert failed")):
        result = act(conn, path)
    assert result.acts_created == 0
    assert conn.execute("SELECT COUNT(*) FROM guardrail_reservations").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'").fetchone()[0] == 0
    assert poll(conn, "guarded", limit=10)
    assert conn.execute("SELECT COUNT(*) FROM guardrail_halts WHERE resolved_at IS NULL").fetchone()[0] == 1


def test_independent_trip_wire_returns_inert_proposal_and_halts(board):
    conn, path = board
    ooda_guardrails.migrate_guardrail_schema(conn)
    from hermes_cli.outcome_ledger import migrate_outcome_schema
    migrate_outcome_schema(conn)
    gap(conn)
    now = time.time()
    _create_test_goal(conn, goal_id="secondary", metric_id="secondary.metric",
                      source_id="independent-monitor")
    baseline = _create_test_measurement(
        conn, "secondary", 100, now - 20, metric_id="secondary.metric",
        source_id="independent-monitor",
    )
    _create_test_measurement(
        conn, "secondary", 40, now - 10, metric_id="secondary.metric",
        source_id="independent-monitor",
    )
    outcome_id = _create_test_outcome(conn, "prior-dark-act", "test-goal-1", None,
                                      started_at=now - 100)
    conn.execute("UPDATE outcomes SET status = 'confirmed', effect = 1, evaluated_at = ? WHERE id = ?",
                 (now - 5, outcome_id))
    denied = act(conn, path, policy(trip_wire=(outcome_id, "secondary", baseline, 0.8, 1.2)))
    assert denied.acts_created == 0
    assert denied.rollback_proposals[0].requires_approval is True
    assert denied.rollback_proposals[0].outcome_ids == [outcome_id]
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'").fetchone()[0] == 0
    assert poll(conn, "guarded", limit=10)[0].payload["goal_id"] == "test-goal-1"


def test_evaluated_outcome_drift_denominator_and_halt(board):
    conn, path = board
    ooda_guardrails.migrate_guardrail_schema(conn)
    from hermes_cli.outcome_ledger import migrate_outcome_schema
    migrate_outcome_schema(conn)
    gap(conn)
    assert act(conn, path).acts_created == 1
    task_id = conn.execute("SELECT task_id FROM guardrail_reservations").fetchone()[0]
    now = time.time()
    outcome_id = _create_test_outcome(conn, task_id, "test-goal-1", None,
                                      started_at=now - 50)
    conn.execute("UPDATE outcomes SET status = 'falsified', effect = -1, evaluated_at = ? WHERE id = ?",
                 (now - 5, outcome_id))
    conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (task_id,))
    _record_breached_measurement(conn, "test-goal-1", value=3, idem_key="drift-recovery")
    _record_breached_measurement(conn, "test-goal-1", idem_key="drift-second")
    roomy = {"task_token": "100", "agent_day": "1", "lane_week": "1", "global_day": "1"}
    assert act(conn, path, policy(caps=roomy)).acts_created == 1  # default floor: unknown
    conn.execute("UPDATE tasks SET status = 'done' WHERE idempotency_key LIKE 'ooda:investigate:%'")
    _record_breached_measurement(conn, "test-goal-1", value=3, idem_key="drift-recovery-2")
    _record_breached_measurement(conn, "test-goal-1", idem_key="drift-third")
    denied = act(conn, path, policy(drift_min_samples=1, caps=roomy))
    assert denied.acts_created == 0
    assert denied.rollback_proposals[0].outcome_ids == [outcome_id]
    assert poll(conn, "guarded", limit=10)

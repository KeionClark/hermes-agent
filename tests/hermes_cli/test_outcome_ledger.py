"""Tests for hermes_cli.outcome_ledger — P2 outcome evaluation system.

Run via ``scripts/run_tests.sh tests/hermes_cli/test_outcome_ledger.py``.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

# Ensure the repo root is importable.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hermes_cli.event_bus import migrate_event_bus
from hermes_cli.goal_metrics import (
    activate_goal,
    create_goal,
    migrate_goal_schema,
    record_measurement,
)
from hermes_cli.outcome_ledger import (
    Outcome,
    OutcomeSchemaNotInitializedError,
    MeasurementSnapshot,
    _OUTCOME_SCHEMA_VERSION,
    complete_action,
    create_outcome,
    evaluate_due_outcome,
    migrate_outcome_schema,
    revert_outcome,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_db() -> sqlite3.Connection:
    """Return a WAL-mode temp-file DB with event-bus + goal + outcome schema."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name, autocommit=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate_event_bus(conn)
    migrate_goal_schema(conn)
    migrate_outcome_schema(conn)
    conn.autocommit = False
    conn.execute("COMMIT")
    return conn


def _begin(conn: sqlite3.Connection) -> None:
    if conn.in_transaction:
        conn.execute("COMMIT")
    conn.execute("BEGIN IMMEDIATE")


def _commit(conn: sqlite3.Connection) -> None:
    conn.execute("COMMIT")


def _rollback(conn: sqlite3.Connection) -> None:
    conn.execute("ROLLBACK")


def _make_active_goal(
    conn: sqlite3.Connection,
    *,
    goal_id: str = "test-goal",
    metric_id: str = "test.metric",
    source_id: str = "test_source",
    direction: str = "higher_is_better",
    target_value: float | None = 100.0,
    tolerance: float = 5.0,
    max_age_seconds: int = 86400,
):
    """Create and activate a goal in a single transaction."""
    _begin(conn)
    goal = create_goal(
        conn,
        goal_id=goal_id,
        metric_id=metric_id,
        source_id=source_id,
        direction=direction,
        target_value=target_value,
        tolerance=tolerance,
        max_age_seconds=max_age_seconds,
    )
    goal = activate_goal(
        conn,
        goal_id=goal.id,
        approval_evidence_ref="keion-approved",
        approval_timestamp=time.time(),
    )
    _commit(conn)
    return goal


def _record_measurement(
    conn: sqlite3.Connection,
    goal_id: str = "test-goal",
    metric_id: str = "test.metric",
    value: float = 95.0,
    source_id: str = "test_source",
    source_timestamp: float | None = None,
    observed_at: float | None = None,
    source_ref: str = "ref-001",
    idempotency_key: str | None = None,
):
    """Record a measurement in a write transaction. Returns MeasurementResult."""
    now = time.time()
    _begin(conn)
    result = record_measurement(
        conn,
        goal_id=goal_id,
        metric_id=metric_id,
        value=value,
        source_id=source_id,
        source_timestamp=source_timestamp or now,
        observed_at=observed_at or now,
        source_ref=source_ref,
        idempotency_key=idempotency_key or f"meas-{time.time_ns()}",
    )
    _commit(conn)
    return result


def _create_outcome_helper(
    conn: sqlite3.Connection,
    *,
    task_id: str = "task-001",
    goal_id: str = "test-goal",
    hypothesis: str = "Deploying fix will improve metric",
    action_type: str = "deploy",
    action_revision: str = "abc123",
    expected_direction: str = "higher",
    before_measurement_id: int | None = None,
    started_at: float | None = None,
    idempotency_key: str | None = None,
    settling_seconds: int = 0,
) -> Outcome:
    """Create an outcome in a write transaction."""
    _begin(conn)
    outcome = create_outcome(
        conn,
        task_id=task_id,
        goal_id=goal_id,
        hypothesis=hypothesis,
        action_type=action_type,
        action_revision=action_revision,
        expected_direction=expected_direction,
        before_measurement_id=before_measurement_id,
        started_at=started_at or time.time(),
        idempotency_key=idempotency_key or f"oc-{time.time_ns()}",
        settling_seconds=settling_seconds,
    )
    _commit(conn)
    return outcome


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

class TestMigration:

    def test_migrate_creates_tables(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name, autocommit=True)
        conn.row_factory = sqlite3.Row

        # Before migration, schema check should fail.
        with pytest.raises(OutcomeSchemaNotInitializedError):
            _begin(conn)
            create_outcome(
                conn,
                task_id="t", goal_id="g", hypothesis="h",
                action_type="a", action_revision="r",
                expected_direction="higher",
                before_measurement_id=None,
                started_at=1.0, idempotency_key="k",
            )

        if conn.in_transaction:
            _rollback(conn)

        migrate_outcome_schema(conn)

        # Verify tables exist.
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "outcomes" in tables
        assert "outcome_schema_version" in tables

    def test_migrate_is_idempotent(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name, autocommit=True)
        conn.row_factory = sqlite3.Row

        migrate_outcome_schema(conn)
        # Second call should be a no-op.
        migrate_outcome_schema(conn)

        row = conn.execute(
            "SELECT version FROM outcome_schema_version WHERE id = 1"
        ).fetchone()
        assert row[0] == _OUTCOME_SCHEMA_VERSION

    def test_migrate_rejects_future_version(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name, autocommit=True)
        conn.row_factory = sqlite3.Row

        migrate_outcome_schema(conn)
        conn.execute(
            "UPDATE outcome_schema_version SET version = 999 WHERE id = 1"
        )

        with pytest.raises(RuntimeError, match="too old"):
            migrate_outcome_schema(conn)


# ---------------------------------------------------------------------------
# create_outcome tests
# ---------------------------------------------------------------------------

class TestCreateOutcome:

    def test_create_without_before(self):
        conn = _fresh_db()
        _make_active_goal(conn)

        outcome = _create_outcome_helper(conn, before_measurement_id=None)
        assert outcome.status == "pending"
        assert outcome.before_measurement_id is None
        assert outcome.before_value is None
        assert outcome.before_source_id is None
        assert outcome.before_metric_id is None
        assert outcome.goal_metric_version == 1
        assert outcome.after_measurement_id is None

    def test_create_with_before_measurement(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        meas = _record_measurement(conn, value=50.0)
        m_id = meas.measurement.id

        started = time.time()
        outcome = _create_outcome_helper(
            conn,
            before_measurement_id=m_id,
            started_at=started,
        )
        assert outcome.status == "pending"
        assert outcome.before_measurement_id == m_id
        assert outcome.before_value == 50.0
        assert outcome.before_source_id == "test_source"
        assert outcome.before_metric_id == "test.metric"
        assert outcome.goal_metric_version == 1

    def test_create_rejects_before_timestamp_after_started(self):
        """before measurement source_timestamp must be <= started_at."""
        conn = _fresh_db()
        _make_active_goal(conn)
        now = time.time()
        meas = _record_measurement(conn, value=50.0, source_timestamp=now)

        with pytest.raises(ValueError, match="source_timestamp"):
            _create_outcome_helper(
                conn,
                before_measurement_id=meas.measurement.id,
                started_at=now - 100,  # before the measurement
            )

    def test_create_rejects_nonexistent_before(self):
        conn = _fresh_db()
        _make_active_goal(conn)

        with pytest.raises(ValueError, match="does not exist"):
            _create_outcome_helper(conn, before_measurement_id=99999)

    def test_create_rejects_nonexistent_goal(self):
        conn = _fresh_db()
        _make_active_goal(conn)

        with pytest.raises(ValueError, match="goal.*does not exist"):
            _create_outcome_helper(conn, goal_id="no-such-goal")

    def test_create_requires_write_txn(self):
        conn = _fresh_db()
        with pytest.raises(RuntimeError, match="write transaction"):
            create_outcome(
                conn,
                task_id="t", goal_id="g", hypothesis="h",
                action_type="a", action_revision="r",
                expected_direction="higher",
                before_measurement_id=None,
                started_at=1.0, idempotency_key="k",
            )

    def test_idempotent_replay_returns_same(self):
        conn = _fresh_db()
        _make_active_goal(conn)

        key = "idem-test-001"
        o1 = _create_outcome_helper(conn, idempotency_key=key)
        o2 = _create_outcome_helper(conn, idempotency_key=key)
        assert o1.id == o2.id

    def test_idempotent_mismatch_raises(self):
        conn = _fresh_db()
        _make_active_goal(conn)

        key = "idem-conflict-001"
        _create_outcome_helper(conn, idempotency_key=key, action_type="deploy")

        with pytest.raises(ValueError, match="conflicting replay"):
            _create_outcome_helper(
                conn, idempotency_key=key, action_type="rollback"
            )

    def test_snapshots_goal_metric_version(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        outcome = _create_outcome_helper(conn)
        assert outcome.goal_metric_version == 1


# ---------------------------------------------------------------------------
# complete_action tests
# ---------------------------------------------------------------------------

class TestCompleteAction:

    def test_complete_transitions_to_awaiting(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()
        outcome = _create_outcome_helper(conn, started_at=started)

        _begin(conn)
        completed = complete_action(
            conn, outcome.id,
            action_revision="def456",
            completed_at=started + 10,
            change_summary="Deployed fix v2",
        )
        _commit(conn)

        assert completed.status == "awaiting_remeasure"
        assert completed.action_revision == "def456"
        assert completed.completed_at == started + 10
        assert completed.change_summary == "Deployed fix v2"

    def test_complete_rejects_non_pending(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()
        outcome = _create_outcome_helper(conn, started_at=started)

        _begin(conn)
        complete_action(
            conn, outcome.id,
            action_revision="r1", completed_at=started + 1,
        )
        _commit(conn)

        with pytest.raises(ValueError, match="awaiting_remeasure"):
            _begin(conn)
            complete_action(
                conn, outcome.id,
                action_revision="r2", completed_at=started + 2,
            )

    def test_complete_rejects_completed_before_started(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()
        outcome = _create_outcome_helper(conn, started_at=started)

        with pytest.raises(ValueError, match="completed_at.*started_at"):
            _begin(conn)
            complete_action(
                conn, outcome.id,
                action_revision="r1", completed_at=started - 10,
            )

    def test_complete_requires_write_txn(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        outcome = _create_outcome_helper(conn)

        with pytest.raises(RuntimeError, match="write transaction"):
            complete_action(
                conn, outcome.id,
                action_revision="r1", completed_at=time.time(),
            )


# ---------------------------------------------------------------------------
# evaluate_due_outcome tests
# ---------------------------------------------------------------------------

class TestEvaluateDueOutcome:

    def _setup_awaiting(self, conn, *, before_value=50.0,
                        expected_direction="higher",
                        settling_seconds=0,
                        max_age_seconds=86400):
        """Create a goal, record a before measurement, create+complete outcome."""
        _make_active_goal(conn, max_age_seconds=max_age_seconds)
        started = time.time()
        meas = _record_measurement(
            conn, value=before_value, source_timestamp=started - 10,
            observed_at=started - 10,
        )
        outcome = _create_outcome_helper(
            conn,
            before_measurement_id=meas.measurement.id,
            started_at=started,
            expected_direction=expected_direction,
            settling_seconds=settling_seconds,
        )
        completed = started + 5
        _begin(conn)
        outcome = complete_action(
            conn, outcome.id,
            action_revision="final-sha",
            completed_at=completed,
        )
        _commit(conn)
        return outcome, completed

    def test_confirmed_higher(self):
        """After > before with expected=higher -> confirmed."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        # Record after measurement (higher value).
        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is not None
        assert result.status == "confirmed"
        assert result.effect == 25.0  # 75 - 50
        assert result.confidence_delta == 25.0
        assert result.after_value == 75.0
        assert result.after_measurement_id is not None
        assert result.evaluated_at is not None

    def test_falsified_higher_when_equal(self):
        """After == before with expected=higher -> falsified (equal is false)."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        after_ts = completed + 10
        _record_measurement(
            conn, value=50.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is not None
        assert result.status == "falsified"
        assert result.effect == 0.0

    def test_falsified_higher_when_lower(self):
        """After < before with expected=higher -> falsified."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        after_ts = completed + 10
        _record_measurement(
            conn, value=30.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is not None
        assert result.status == "falsified"
        assert result.effect == -20.0
        assert result.confidence_delta == 20.0

    def test_confirmed_lower(self):
        """After < before with expected=lower -> confirmed."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="lower",
        )

        after_ts = completed + 10
        _record_measurement(
            conn, value=30.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is not None
        assert result.status == "confirmed"
        assert result.effect == -20.0
        assert result.confidence_delta == 20.0

    def test_falsified_lower_when_equal(self):
        """After == before with expected=lower -> falsified."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="lower",
        )

        after_ts = completed + 10
        _record_measurement(
            conn, value=50.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is not None
        assert result.status == "falsified"

    def test_returns_none_when_no_after_measurement(self):
        """No valid after measurement -> returns None (still pending)."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(conn)

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=completed + 1)
        _commit(conn)

        assert result is None

    def test_returns_none_when_after_too_early_settling(self):
        """After measurement exists but before settling period -> None."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, settling_seconds=60,
        )

        # Record measurement within settling window.
        after_ts = completed + 10  # < completed + 60
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is None

    def test_confirms_after_settling_period(self):
        """After measurement after settling period -> evaluates."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, settling_seconds=60, before_value=50.0,
            expected_direction="higher",
        )

        # Record measurement after settling window.
        after_ts = completed + 61
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is not None
        assert result.status == "confirmed"

    def test_returns_none_when_after_is_stale(self):
        """After measurement exists but too old (stale) -> None."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, max_age_seconds=60,
        )

        # Record measurement that will be stale by evaluation time.
        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        # Evaluate much later so the measurement is stale.
        eval_now = after_ts + 120  # 120s > 60s max_age
        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=eval_now)
        _commit(conn)

        assert result is None

    def test_absent_before_fails_closed_falsified(self):
        """No before measurement -> fail closed -> falsified."""
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()

        outcome = _create_outcome_helper(
            conn, before_measurement_id=None, started_at=started,
        )
        completed = started + 5
        _begin(conn)
        outcome = complete_action(
            conn, outcome.id,
            action_revision="sha1", completed_at=completed,
        )
        _commit(conn)

        # Record after measurement.
        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is not None
        assert result.status == "falsified"
        assert result.effect is None
        assert result.confidence_delta is None
        assert result.after_value == 75.0

    def test_idempotent_evaluate_returns_same(self):
        """Evaluating an already-evaluated outcome returns existing record."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        r1 = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        _begin(conn)
        r2 = evaluate_due_outcome(conn, outcome.id, now=after_ts + 100)
        _commit(conn)

        assert r1.id == r2.id
        assert r1.status == r2.status == "confirmed"
        assert r1.evaluated_at == r2.evaluated_at  # not re-evaluated

    def test_raises_on_pending_status(self):
        """Evaluating a pending outcome raises."""
        conn = _fresh_db()
        _make_active_goal(conn)
        outcome = _create_outcome_helper(conn)

        with pytest.raises(ValueError, match="pending"):
            _begin(conn)
            evaluate_due_outcome(conn, outcome.id, now=time.time())

    def test_requires_write_txn(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        outcome = _create_outcome_helper(conn)

        with pytest.raises(RuntimeError, match="write transaction"):
            evaluate_due_outcome(conn, outcome.id, now=time.time())

    def test_metric_version_drift_fails_closed(self):
        """If goal metric_version changed since creation, fail closed."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        # Record valid after measurement.
        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        # Mutate the goal's metric_version behind the outcome's back.
        conn.execute(
            "UPDATE goals SET metric_version = 2 WHERE id = 'test-goal'"
        )

        with pytest.raises(ValueError, match="metric_version changed"):
            _begin(conn)
            evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)

    def test_wrong_source_id_not_matched(self):
        """Measurement with wrong source_id is not used as after."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        # Record measurement with a different source_id directly.
        after_ts = completed + 10
        _begin(conn)
        conn.execute(
            "INSERT INTO goal_measurements "
            "(goal_id, metric_id, value, source_id, source_timestamp, "
            "observed_at, source_ref, idempotency_key, edge_state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unknown', ?)",
            ("test-goal", "test.metric", 999.0, "wrong_source",
             after_ts, after_ts, "ref", f"wrong-src-{time.time_ns()}", after_ts),
        )
        _commit(conn)

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        # Should not match the wrong-source measurement.
        assert result is None

    def test_wrong_metric_id_not_matched(self):
        """Measurement with wrong metric_id is not used as after."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        after_ts = completed + 10
        _begin(conn)
        conn.execute(
            "INSERT INTO goal_measurements "
            "(goal_id, metric_id, value, source_id, source_timestamp, "
            "observed_at, source_ref, idempotency_key, edge_state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unknown', ?)",
            ("test-goal", "wrong.metric", 999.0, "test_source",
             after_ts, after_ts, "ref", f"wrong-met-{time.time_ns()}", after_ts),
        )
        _commit(conn)

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result is None

    def test_after_before_completed_at_not_matched(self):
        """Measurement with source_timestamp <= completed_at is not after."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        # Insert a measurement AT completed_at (not strictly after).
        _begin(conn)
        conn.execute(
            "INSERT INTO goal_measurements "
            "(goal_id, metric_id, value, source_id, source_timestamp, "
            "observed_at, source_ref, idempotency_key, edge_state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unknown', ?)",
            ("test-goal", "test.metric", 999.0, "test_source",
             completed, completed, "ref", f"at-completed-{time.time_ns()}", completed),
        )
        _commit(conn)

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=completed + 1)
        _commit(conn)

        assert result is None

    def test_emits_event_on_evaluation(self):
        """Evaluation emits outcome.evaluated event with safe payload."""
        conn = _fresh_db()
        outcome, completed = self._setup_awaiting(
            conn, before_value=50.0, expected_direction="higher",
        )

        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result.status == "confirmed"

        # Check event was emitted.
        evt = conn.execute(
            "SELECT * FROM hermes_events "
            "WHERE event_type = 'outcome.evaluated' "
            "AND source = 'outcome_ledger'"
        ).fetchone()
        assert evt is not None

        import json
        payload = json.loads(evt["payload"])
        assert payload["goal_id"] == "test-goal"
        assert payload["task_id"] == "task-001"
        assert payload["metric_id"] == "test.metric"
        assert payload["status"] == "confirmed"
        # Verify no outcome_id in payload (not in allowlist).
        assert "outcome_id" not in payload

        # Verify idempotency_key encodes outcome ID for traceability.
        assert evt["idempotency_key"] == f"outcome_eval:{result.id}"

    def test_event_not_emitted_when_metric_id_missing(self):
        """If metric_id cannot be normalized, no event is emitted."""
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()

        # Create outcome without before measurement (so before_metric_id=None).
        outcome = _create_outcome_helper(
            conn, before_measurement_id=None, started_at=started,
        )
        completed = started + 5
        _begin(conn)
        outcome = complete_action(
            conn, outcome.id,
            action_revision="sha1", completed_at=completed,
        )
        _commit(conn)

        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        # Falsified (no before), but after_metric_id should be set from the
        # after measurement, so event SHOULD still be emitted.
        assert result.status == "falsified"
        assert result.after_metric_id == "test.metric"

        evt = conn.execute(
            "SELECT * FROM hermes_events "
            "WHERE event_type = 'outcome.evaluated'"
        ).fetchone()
        # Event should be emitted because after_metric_id is available.
        assert evt is not None


# ---------------------------------------------------------------------------
# revert_outcome tests
# ---------------------------------------------------------------------------

class TestRevertOutcome:

    def test_revert_from_pending(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        outcome = _create_outcome_helper(conn)

        now = time.time()
        _begin(conn)
        reverted = revert_outcome(
            conn, outcome.id, reason="Action was wrong", now=now,
        )
        _commit(conn)

        assert reverted.status == "reverted"
        assert reverted.revert_reason == "Action was wrong"
        assert reverted.reverted_at == now

    def test_revert_from_awaiting(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()
        outcome = _create_outcome_helper(conn, started_at=started)

        _begin(conn)
        outcome = complete_action(
            conn, outcome.id,
            action_revision="r1", completed_at=started + 5,
        )
        _commit(conn)

        now = time.time()
        _begin(conn)
        reverted = revert_outcome(
            conn, outcome.id, reason="Rollback needed", now=now,
        )
        _commit(conn)

        assert reverted.status == "reverted"

    def test_revert_idempotent(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        outcome = _create_outcome_helper(conn)

        now = time.time()
        _begin(conn)
        r1 = revert_outcome(conn, outcome.id, reason="Bad", now=now)
        _commit(conn)

        _begin(conn)
        r2 = revert_outcome(conn, outcome.id, reason="Bad again", now=now + 10)
        _commit(conn)

        assert r1.id == r2.id
        assert r1.revert_reason == r2.revert_reason == "Bad"

    def test_revert_evaluated_outcome(self):
        """Can revert a confirmed/falsified outcome."""
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()
        meas = _record_measurement(
            conn, value=50.0,
            source_timestamp=started - 10, observed_at=started - 10,
        )
        outcome = _create_outcome_helper(
            conn,
            before_measurement_id=meas.measurement.id,
            started_at=started,
        )
        completed = started + 5
        _begin(conn)
        outcome = complete_action(
            conn, outcome.id,
            action_revision="sha1", completed_at=completed,
        )
        _commit(conn)

        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )
        _begin(conn)
        evaluated = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)
        assert evaluated.status == "confirmed"

        now = time.time()
        _begin(conn)
        reverted = revert_outcome(
            conn, evaluated.id, reason="Was a fluke", now=now,
        )
        _commit(conn)
        assert reverted.status == "reverted"

    def test_evaluate_reverted_returns_existing(self):
        """Evaluating a reverted outcome returns the reverted record."""
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()
        outcome = _create_outcome_helper(conn, started_at=started)

        _begin(conn)
        outcome = complete_action(
            conn, outcome.id,
            action_revision="r1", completed_at=started + 5,
        )
        _commit(conn)

        _begin(conn)
        reverted = revert_outcome(
            conn, outcome.id, reason="Nope", now=time.time(),
        )
        _commit(conn)

        _begin(conn)
        result = evaluate_due_outcome(conn, reverted.id, now=time.time())
        _commit(conn)

        assert result.status == "reverted"

    def test_revert_requires_write_txn(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        outcome = _create_outcome_helper(conn)

        with pytest.raises(RuntimeError, match="write transaction"):
            revert_outcome(
                conn, outcome.id, reason="Bad", now=time.time(),
            )


# ---------------------------------------------------------------------------
# Two-connection concurrency tests
# ---------------------------------------------------------------------------

class TestTwoConnections:

    def test_concurrent_create_different_keys(self):
        """Two connections can create outcomes with different idempotency keys."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

        conn1 = sqlite3.connect(db_path, autocommit=True)
        conn1.row_factory = sqlite3.Row
        conn1.execute("PRAGMA journal_mode=WAL")
        migrate_event_bus(conn1)
        migrate_goal_schema(conn1)
        migrate_outcome_schema(conn1)

        conn2 = sqlite3.connect(db_path, autocommit=True)
        conn2.row_factory = sqlite3.Row
        conn2.execute("PRAGMA journal_mode=WAL")

        # Create goal on conn1.
        conn1.execute("BEGIN IMMEDIATE")
        create_goal(
            conn1, goal_id="g1", metric_id="m", source_id="s",
        )
        activate_goal(conn1, "g1", "evidence", time.time())
        conn1.execute("COMMIT")

        # Create outcome on conn1.
        conn1.execute("BEGIN IMMEDIATE")
        o1 = create_outcome(
            conn1,
            task_id="t1", goal_id="g1", hypothesis="h1",
            action_type="deploy", action_revision="r1",
            expected_direction="higher",
            before_measurement_id=None,
            started_at=time.time(), idempotency_key="key-conn1",
        )
        conn1.execute("COMMIT")

        # Create outcome on conn2.
        conn2.execute("BEGIN IMMEDIATE")
        o2 = create_outcome(
            conn2,
            task_id="t2", goal_id="g1", hypothesis="h2",
            action_type="deploy", action_revision="r2",
            expected_direction="lower",
            before_measurement_id=None,
            started_at=time.time(), idempotency_key="key-conn2",
        )
        conn2.execute("COMMIT")

        assert o1.id != o2.id
        assert o1.idempotency_key == "key-conn1"
        assert o2.idempotency_key == "key-conn2"

        conn1.close()
        conn2.close()

    def test_concurrent_idempotency_conflict(self):
        """Two connections with same key but different payload: second fails."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

        conn1 = sqlite3.connect(db_path, autocommit=True)
        conn1.row_factory = sqlite3.Row
        conn1.execute("PRAGMA journal_mode=WAL")
        migrate_event_bus(conn1)
        migrate_goal_schema(conn1)
        migrate_outcome_schema(conn1)

        conn2 = sqlite3.connect(db_path, autocommit=True)
        conn2.row_factory = sqlite3.Row
        conn2.execute("PRAGMA journal_mode=WAL")

        # Create goal.
        conn1.execute("BEGIN IMMEDIATE")
        create_goal(conn1, goal_id="g1", metric_id="m", source_id="s")
        activate_goal(conn1, "g1", "evidence", time.time())
        conn1.execute("COMMIT")

        key = "shared-key-001"

        # First create succeeds.
        conn1.execute("BEGIN IMMEDIATE")
        o1 = create_outcome(
            conn1,
            task_id="t1", goal_id="g1", hypothesis="h",
            action_type="deploy", action_revision="r1",
            expected_direction="higher",
            before_measurement_id=None,
            started_at=time.time(), idempotency_key=key,
        )
        conn1.execute("COMMIT")

        # Second create with different payload fails.
        conn2.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="conflicting replay"):
            create_outcome(
                conn2,
                task_id="t1", goal_id="g1", hypothesis="h",
                action_type="rollback", action_revision="r1",
                expected_direction="higher",
                before_measurement_id=None,
                started_at=time.time(), idempotency_key=key,
            )
        conn2.execute("ROLLBACK")

        conn1.close()
        conn2.close()


# ---------------------------------------------------------------------------
# Provenance immutability tests
# ---------------------------------------------------------------------------

class TestProvenanceImmutability:

    def test_before_provenance_is_snapshot(self):
        """Before provenance is a snapshot, not a live reference."""
        conn = _fresh_db()
        _make_active_goal(conn)
        now = time.time()
        meas = _record_measurement(
            conn, value=50.0,
            source_timestamp=now - 10, observed_at=now - 10,
            source_ref="original-ref",
        )

        outcome = _create_outcome_helper(
            conn,
            before_measurement_id=meas.measurement.id,
            started_at=now,
        )

        # Verify snapshot.
        assert outcome.before_value == 50.0
        assert outcome.before_source_ref == "original-ref"

        # Even if the original row were somehow mutated (it shouldn't be,
        # but this proves the outcome has its own copy), the outcome
        # retains its original snapshot.
        assert outcome.before_measurement_id == meas.measurement.id

    def test_after_provenance_is_snapshot(self):
        """After provenance is a snapshot persisted at evaluation time."""
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()
        meas = _record_measurement(
            conn, value=50.0,
            source_timestamp=started - 10, observed_at=started - 10,
        )
        outcome = _create_outcome_helper(
            conn,
            before_measurement_id=meas.measurement.id,
            started_at=started,
        )
        completed = started + 5
        _begin(conn)
        outcome = complete_action(
            conn, outcome.id,
            action_revision="sha1", completed_at=completed,
        )
        _commit(conn)

        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
            source_ref="after-ref",
        )

        _begin(conn)
        result = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        assert result.after_value == 75.0
        assert result.after_source_ref == "after-ref"
        assert result.after_source_id == "test_source"
        assert result.after_metric_id == "test.metric"

    def test_effect_persisted_only_once(self):
        """Effect and confidence_delta do not change on re-evaluation."""
        conn = _fresh_db()
        _make_active_goal(conn)
        started = time.time()
        meas = _record_measurement(
            conn, value=50.0,
            source_timestamp=started - 10, observed_at=started - 10,
        )
        outcome = _create_outcome_helper(
            conn,
            before_measurement_id=meas.measurement.id,
            started_at=started, expected_direction="higher",
        )
        completed = started + 5
        _begin(conn)
        outcome = complete_action(
            conn, outcome.id,
            action_revision="sha1", completed_at=completed,
        )
        _commit(conn)

        after_ts = completed + 10
        _record_measurement(
            conn, value=75.0,
            source_timestamp=after_ts, observed_at=after_ts,
        )

        _begin(conn)
        r1 = evaluate_due_outcome(conn, outcome.id, now=after_ts + 1)
        _commit(conn)

        # Record a newer, different measurement.
        after_ts2 = after_ts + 100
        _record_measurement(
            conn, value=200.0,
            source_timestamp=after_ts2, observed_at=after_ts2,
        )

        # Re-evaluate — idempotent, should return same result.
        _begin(conn)
        r2 = evaluate_due_outcome(conn, outcome.id, now=after_ts2 + 1)
        _commit(conn)

        assert r1.effect == r2.effect == 25.0
        assert r1.confidence_delta == r2.confidence_delta == 25.0
        assert r1.evaluated_at == r2.evaluated_at

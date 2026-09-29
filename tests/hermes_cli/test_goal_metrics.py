"""Tests for hermes_cli.goal_metrics — P0b goal/measurement/gap system.

Run via ``scripts/run_tests.sh tests/hermes_cli/test_goal_metrics.py``.
"""

from __future__ import annotations

import json
import math
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

from hermes_cli.event_bus import migrate_event_bus, poll, subscribe
from hermes_cli.goal_metrics import (
    BoardSnapshot,
    Goal,
    Measurement,
    MeasurementResult,
    activate_goal,
    create_goal,
    get_goal,
    list_goals,
    migrate_goal_schema,
    pause_goal,
    record_board_snapshot,
    record_measurement,
    snapshot_board_counts,
    _compute_gap,
    _evaluate_edge,
    _GOAL_SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_db() -> sqlite3.Connection:
    """Return a WAL-mode temp-file DB with event-bus + goal-metrics schema."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name, autocommit=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate_event_bus(conn)
    migrate_goal_schema(conn)
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
    title: str | None = None,
) -> Goal:
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
        title=title,
    )
    goal = activate_goal(
        conn,
        goal_id=goal.id,
        approval_evidence_ref="keion-approved-2026-09-29",
        approval_timestamp=time.time(),
    )
    _commit(conn)
    return goal


def _record(
    conn: sqlite3.Connection,
    goal_id: str = "test-goal",
    metric_id: str = "test.metric",
    value: float = 95.0,
    source_id: str = "test_source",
    source_timestamp: float | None = None,
    observed_at: float | None = None,
    source_ref: str = "ref-001",
    payload_hash: str | None = None,
    idempotency_key: str | None = None,
) -> MeasurementResult:
    """Record a measurement in a write transaction."""
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
        payload_hash=payload_hash,
        idempotency_key=idempotency_key or f"key-{time.time_ns()}",
    )
    _commit(conn)
    return result


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

class TestMigration:
    """Goal schema migration: idempotent, atomic, independent."""

    def test_fresh_db_creates_all_tables(self):
        conn = _fresh_db()
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "goal_schema_version" in tables
        assert "goals" in tables
        assert "goal_measurements" in tables
        conn.close()

    def test_idempotent_migration(self):
        conn = _fresh_db()
        # Second migration should be a no-op.
        conn.autocommit = True
        migrate_goal_schema(conn)
        migrate_goal_schema(conn)
        row = conn.execute(
            "SELECT version FROM goal_schema_version WHERE id = 1"
        ).fetchone()
        assert row[0] == _GOAL_SCHEMA_VERSION
        conn.close()

    def test_future_schema_version_raises(self):
        conn = _fresh_db()
        conn.autocommit = True
        conn.execute(
            "UPDATE goal_schema_version SET version = 999 WHERE id = 1"
        )
        with pytest.raises(RuntimeError, match="too old"):
            migrate_goal_schema(conn)
        conn.close()

    def test_legacy_board_without_goals_gets_migration(self):
        """A DB that has event bus but no goals gets goals on migration."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name, autocommit=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        migrate_event_bus(conn)
        # No goals tables yet.
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "goals" not in tables

        migrate_goal_schema(conn)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "goals" in tables
        assert "goal_measurements" in tables
        conn.close()

    def test_does_not_alter_task_events(self):
        """Migration must not create or modify task_events table."""
        conn = _fresh_db()
        # task_events doesn't exist in our test DB (only in real kanban).
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        # Our migration should only add goal-specific tables.
        assert "goal_schema_version" in tables
        assert "goals" in tables
        assert "goal_measurements" in tables
        # Should NOT have task_events (we didn't create it).
        assert "task_events" not in tables
        conn.close()


# ---------------------------------------------------------------------------
# Goal CRUD tests
# ---------------------------------------------------------------------------

class TestGoalCRUD:
    """Goal creation, activation, pausing, listing."""

    def test_create_goal_draft(self):
        conn = _fresh_db()
        _begin(conn)
        goal = create_goal(
            conn,
            metric_id="test.metric",
            source_id="test_source",
            direction="higher_is_better",
            target_value=100.0,
            tolerance=5.0,
            max_age_seconds=3600,
        )
        _commit(conn)
        assert goal.state == "draft"
        assert goal.target_value == 100.0
        assert goal.tolerance == 5.0
        assert goal.approval_evidence_ref is None
        conn.close()

    def test_create_goal_requires_txn(self):
        conn = _fresh_db()
        with pytest.raises(RuntimeError, match="write transaction"):
            create_goal(
                conn,
                metric_id="m",
                source_id="s",
            )
        conn.close()

    def test_activate_requires_approval(self):
        conn = _fresh_db()
        _begin(conn)
        goal = create_goal(
            conn, metric_id="m", source_id="s",
            target_value=10.0, max_age_seconds=3600,
        )
        goal = activate_goal(
            conn, goal.id,
            approval_evidence_ref="approved-by-keion",
            approval_timestamp=time.time(),
        )
        _commit(conn)
        assert goal.state == "active"
        assert goal.approval_evidence_ref == "approved-by-keion"
        conn.close()

    def test_activate_without_evidence_raises(self):
        conn = _fresh_db()
        _begin(conn)
        goal = create_goal(
            conn, metric_id="m", source_id="s",
            target_value=10.0, max_age_seconds=3600,
        )
        with pytest.raises(ValueError, match="approval_evidence_ref"):
            activate_goal(conn, goal.id, "", time.time())
        _rollback(conn)
        conn.close()

    def test_pause_goal(self):
        conn = _fresh_db()
        goal = _make_active_goal(conn)
        _begin(conn)
        paused = pause_goal(conn, goal.id)
        _commit(conn)
        assert paused.state == "paused"
        conn.close()

    def test_get_goal(self):
        conn = _fresh_db()
        _begin(conn)
        goal = create_goal(
            conn, metric_id="m", source_id="s", max_age_seconds=3600,
        )
        _commit(conn)
        retrieved = get_goal(conn, goal.id)
        assert retrieved is not None
        assert retrieved.id == goal.id
        assert get_goal(conn, "nonexistent") is None
        conn.close()

    def test_list_goals_with_filter(self):
        conn = _fresh_db()
        _begin(conn)
        create_goal(
            conn, goal_id="g1", metric_id="m", source_id="s",
            target_value=10.0, max_age_seconds=3600,
        )
        create_goal(
            conn, goal_id="g2", metric_id="m2", source_id="s2",
            target_value=20.0, max_age_seconds=3600,
        )
        _commit(conn)

        _begin(conn)
        activate_goal(conn, "g1", "evidence", time.time())
        _commit(conn)

        all_goals = list_goals(conn)
        assert len(all_goals) == 2
        active = list_goals(conn, state="active")
        assert len(active) == 1
        assert active[0].id == "g1"
        drafts = list_goals(conn, state="draft")
        assert len(drafts) == 1
        assert drafts[0].id == "g2"
        conn.close()

    def test_parent_fk_enforcement(self):
        conn = _fresh_db()
        _begin(conn)
        parent = create_goal(
            conn, goal_id="parent", metric_id="m", source_id="s",
            max_age_seconds=3600,
        )
        child = create_goal(
            conn, goal_id="child", parent_id="parent",
            metric_id="m2", source_id="s", max_age_seconds=3600,
        )
        assert child.parent_id == "parent"

        with pytest.raises(ValueError, match="does not exist"):
            create_goal(
                conn, parent_id="nonexistent",
                metric_id="m3", source_id="s", max_age_seconds=3600,
            )
        _rollback(conn)
        conn.close()

    def test_nonfinite_target_rejected(self):
        conn = _fresh_db()
        _begin(conn)
        with pytest.raises(ValueError, match="finite"):
            create_goal(
                conn, metric_id="m", source_id="s",
                target_value=float("nan"), max_age_seconds=3600,
            )
        _rollback(conn)

        _begin(conn)
        with pytest.raises(ValueError, match="finite"):
            create_goal(
                conn, metric_id="m", source_id="s",
                target_value=float("inf"), max_age_seconds=3600,
            )
        _rollback(conn)
        conn.close()

    def test_negative_tolerance_rejected(self):
        conn = _fresh_db()
        _begin(conn)
        with pytest.raises(ValueError, match="non-negative"):
            create_goal(
                conn, metric_id="m", source_id="s",
                tolerance=-1.0, max_age_seconds=3600,
            )
        _rollback(conn)
        conn.close()

    def test_zero_max_age_rejected(self):
        conn = _fresh_db()
        _begin(conn)
        with pytest.raises(ValueError, match="positive"):
            create_goal(
                conn, metric_id="m", source_id="s",
                max_age_seconds=0,
            )
        _rollback(conn)
        conn.close()

    def test_no_seeded_active_goals(self):
        """Active goals require explicit approval — none are seeded."""
        conn = _fresh_db()
        active = list_goals(conn, state="active")
        assert len(active) == 0
        conn.close()

    def test_metric_version_tracked(self):
        conn = _fresh_db()
        _begin(conn)
        goal = create_goal(
            conn, metric_id="m", source_id="s", max_age_seconds=3600,
        )
        _commit(conn)
        assert goal.metric_version == 1
        conn.close()


# ---------------------------------------------------------------------------
# Gap computation tests
# ---------------------------------------------------------------------------

class TestGapComputation:
    """Pure gap and edge-state evaluation."""

    def test_higher_is_better_gap(self):
        # target=100, value=90 → gap = 10 (positive, target not met)
        assert _compute_gap(90.0, 100.0, "higher_is_better") == 10.0
        # value=110 → gap = -10 (negative, exceeds target)
        assert _compute_gap(110.0, 100.0, "higher_is_better") == -10.0
        # value=100 → gap = 0 (exactly at target)
        assert _compute_gap(100.0, 100.0, "higher_is_better") == 0.0

    def test_lower_is_better_gap(self):
        # target=10, value=15 → gap = 5 (positive, target not met)
        assert _compute_gap(15.0, 10.0, "lower_is_better") == 5.0
        # value=5 → gap = -5 (negative, better than target)
        assert _compute_gap(5.0, 10.0, "lower_is_better") == -5.0
        # value=10 → gap = 0
        assert _compute_gap(10.0, 10.0, "lower_is_better") == 0.0

    def test_exact_boundary_is_healthy(self):
        # gap=5, tolerance=5 → healthy (exact boundary is healthy)
        state, mag = _evaluate_edge(5.0, 5.0, "unknown")
        assert state == "healthy"
        assert mag == 0.0

    def test_within_tolerance_is_healthy(self):
        state, mag = _evaluate_edge(3.0, 5.0, "unknown")
        assert state == "healthy"
        assert mag == 0.0

    def test_beyond_tolerance_is_breached(self):
        state, mag = _evaluate_edge(8.0, 5.0, "unknown")
        assert state == "breached"
        assert mag == 3.0

    def test_negative_gap_is_healthy(self):
        state, mag = _evaluate_edge(-10.0, 5.0, "unknown")
        assert state == "healthy"
        assert mag == 0.0

    def test_zero_tolerance(self):
        state, mag = _evaluate_edge(0.0, 0.0, "unknown")
        assert state == "healthy"
        assert mag == 0.0

        state, mag = _evaluate_edge(0.001, 0.0, "unknown")
        assert state == "breached"
        assert mag == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Measurement recording tests
# ---------------------------------------------------------------------------

class TestMeasurementRecording:
    """Append-only measurement with validation and dedup."""

    def test_basic_measurement(self):
        conn = _fresh_db()
        goal = _make_active_goal(conn)
        result = _record(conn, value=95.0)
        assert not result.is_replay
        assert result.measurement.value == 95.0
        assert result.measurement.edge_state == "healthy"
        conn.close()

    def test_requires_active_goal(self):
        conn = _fresh_db()
        _begin(conn)
        goal = create_goal(
            conn, goal_id="draft-goal",
            metric_id="test.metric", source_id="test_source",
            target_value=100.0, max_age_seconds=86400,
        )
        _commit(conn)

        _begin(conn)
        with pytest.raises(ValueError, match="not 'active'"):
            record_measurement(
                conn,
                goal_id="draft-goal",
                metric_id="test.metric",
                value=50.0,
                source_id="test_source",
                source_timestamp=time.time(),
                observed_at=time.time(),
                source_ref="ref",
                idempotency_key="k1",
            )
        _rollback(conn)
        conn.close()

    def test_mismatched_metric_rejected(self):
        conn = _fresh_db()
        _make_active_goal(conn, metric_id="correct.metric")
        _begin(conn)
        with pytest.raises(ValueError, match="metric_id"):
            record_measurement(
                conn,
                goal_id="test-goal",
                metric_id="wrong.metric",
                value=50.0,
                source_id="test_source",
                source_timestamp=time.time(),
                observed_at=time.time(),
                source_ref="ref",
                idempotency_key="k1",
            )
        _rollback(conn)
        conn.close()

    def test_mismatched_source_rejected(self):
        conn = _fresh_db()
        _make_active_goal(conn, source_id="correct_source")
        _begin(conn)
        with pytest.raises(ValueError, match="source_id"):
            record_measurement(
                conn,
                goal_id="test-goal",
                metric_id="test.metric",
                value=50.0,
                source_id="wrong_source",
                source_timestamp=time.time(),
                observed_at=time.time(),
                source_ref="ref",
                idempotency_key="k1",
            )
        _rollback(conn)
        conn.close()

    def test_nonfinite_value_rejected(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        for bad in (float("nan"), float("inf"), float("-inf")):
            _begin(conn)
            with pytest.raises(ValueError, match="finite"):
                record_measurement(
                    conn,
                    goal_id="test-goal",
                    metric_id="test.metric",
                    value=bad,
                    source_id="test_source",
                    source_timestamp=time.time(),
                    observed_at=time.time(),
                    source_ref="ref",
                    idempotency_key=f"k-{bad}",
                )
            _rollback(conn)
        conn.close()

    def test_future_timestamp_rejected(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        future_ts = time.time() + 600  # 10 min in future
        _begin(conn)
        with pytest.raises(ValueError, match="future"):
            record_measurement(
                conn,
                goal_id="test-goal",
                metric_id="test.metric",
                value=95.0,
                source_id="test_source",
                source_timestamp=future_ts,
                observed_at=time.time(),
                source_ref="ref",
                idempotency_key="k1",
            )
        _rollback(conn)
        conn.close()

    def test_stale_measurement_rejected(self):
        conn = _fresh_db()
        _make_active_goal(conn, max_age_seconds=60)
        old_ts = time.time() - 120  # 2 min old, max_age=60s
        _begin(conn)
        with pytest.raises(ValueError, match="stale"):
            record_measurement(
                conn,
                goal_id="test-goal",
                metric_id="test.metric",
                value=95.0,
                source_id="test_source",
                source_timestamp=old_ts,
                observed_at=time.time(),
                source_ref="ref",
                idempotency_key="k1",
            )
        _rollback(conn)
        conn.close()

    def test_missing_provenance_rejected(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        _begin(conn)
        with pytest.raises(ValueError, match="provenance"):
            record_measurement(
                conn,
                goal_id="test-goal",
                metric_id="test.metric",
                value=95.0,
                source_id="test_source",
                source_timestamp=time.time(),
                observed_at=time.time(),
                source_ref=None,
                payload_hash=None,
                idempotency_key="k1",
            )
        _rollback(conn)
        conn.close()

    def test_monotonic_source_timestamp(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        now = time.time()
        # First measurement at now.
        _record(conn, source_timestamp=now, idempotency_key="k1")
        # Second at now - 10 should fail.
        _begin(conn)
        with pytest.raises(ValueError, match="monotonic"):
            record_measurement(
                conn,
                goal_id="test-goal",
                metric_id="test.metric",
                value=95.0,
                source_id="test_source",
                source_timestamp=now - 10,
                observed_at=now,
                source_ref="ref",
                idempotency_key="k2",
            )
        _rollback(conn)
        conn.close()


# ---------------------------------------------------------------------------
# Idempotency tests
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Same-key replay and conflicting-key detection."""

    def test_same_key_replay_returns_prior(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        now = time.time()
        result1 = _record(
            conn, value=95.0, source_timestamp=now,
            payload_hash="abc123", idempotency_key="k-dedup",
        )
        result2 = _record(
            conn, value=95.0, source_timestamp=now,
            payload_hash="abc123", idempotency_key="k-dedup",
        )
        assert result2.is_replay
        assert result2.measurement.id == result1.measurement.id
        assert result2.measurement.value == result1.measurement.value
        conn.close()

    def test_conflicting_payload_fails_closed(self):
        conn = _fresh_db()
        _make_active_goal(conn)
        now = time.time()
        _record(
            conn, value=95.0, source_timestamp=now,
            payload_hash="hash1", idempotency_key="k-conflict",
        )
        _begin(conn)
        with pytest.raises(ValueError, match="conflicting"):
            record_measurement(
                conn,
                goal_id="test-goal",
                metric_id="test.metric",
                value=95.0,
                source_id="test_source",
                source_timestamp=now,
                observed_at=now,
                source_ref="ref",
                payload_hash="hash2",
                idempotency_key="k-conflict",
            )
        _rollback(conn)
        conn.close()

    def test_replay_returns_prior_events(self):
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=0.0)
        now = time.time()
        # First reading breached → emits events.
        result1 = _record(
            conn, value=50.0, source_timestamp=now,
            payload_hash="abc", idempotency_key="k-events",
        )
        assert len(result1.events) == 2  # gap_detected + threshold_crossed

        # Replay returns same events.
        result2 = _record(
            conn, value=50.0, source_timestamp=now,
            payload_hash="abc", idempotency_key="k-events",
        )
        assert result2.is_replay
        assert len(result2.events) == 2
        assert result2.events[0]["event_id"] == result1.events[0]["event_id"]
        conn.close()


# ---------------------------------------------------------------------------
# Edge transition tests
# ---------------------------------------------------------------------------

class TestEdgeTransitions:
    """Gap transitions: healthy↔breached with event emission."""

    def test_first_healthy_no_event(self):
        """First valid reading that is healthy emits no event."""
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=5.0)
        now = time.time()
        result = _record(
            conn, value=97.0, source_timestamp=now,
            idempotency_key="first-healthy",
        )
        assert result.measurement.edge_state == "healthy"
        assert len(result.events) == 0
        conn.close()

    def test_first_breached_emits_crossing(self):
        """First reading breached emits initial crossing once."""
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=5.0)
        now = time.time()
        result = _record(
            conn, value=80.0, source_timestamp=now,
            idempotency_key="first-breached",
        )
        assert result.measurement.edge_state == "breached"
        assert len(result.events) == 2
        types = {e["event_type"] for e in result.events}
        assert types == {"goal.gap_detected", "metric.threshold_crossed"}
        conn.close()

    def test_healthy_to_breached_emits(self):
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=5.0)
        now = time.time()
        # Healthy first.
        _record(conn, value=97.0, source_timestamp=now, idempotency_key="h1")
        # Then breach.
        result = _record(
            conn, value=80.0, source_timestamp=now + 1,
            idempotency_key="b1",
        )
        assert result.measurement.edge_state == "breached"
        assert len(result.events) == 2
        types = {e["event_type"] for e in result.events}
        assert types == {"goal.gap_detected", "metric.threshold_crossed"}
        conn.close()

    def test_breached_to_healthy_emits_recovery(self):
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=5.0)
        now = time.time()
        # First breached.
        _record(conn, value=80.0, source_timestamp=now, idempotency_key="b1")
        # Then recovery.
        result = _record(
            conn, value=97.0, source_timestamp=now + 1,
            idempotency_key="r1",
        )
        assert result.measurement.edge_state == "healthy"
        assert len(result.events) == 1
        assert result.events[0]["event_type"] == "goal.achieved"
        conn.close()

    def test_repeated_breached_no_duplicate_events(self):
        """Repeated breached readings emit no repeated crossing."""
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=5.0)
        now = time.time()
        # First breached.
        r1 = _record(
            conn, value=80.0, source_timestamp=now,
            idempotency_key="b1",
        )
        assert len(r1.events) == 2

        # Second breached — no new events.
        r2 = _record(
            conn, value=75.0, source_timestamp=now + 1,
            idempotency_key="b2",
        )
        assert r2.measurement.edge_state == "breached"
        assert len(r2.events) == 0
        conn.close()

    def test_re_arm_after_recovery(self):
        """After recovery, a new breach emits again (re-arm)."""
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=5.0)
        now = time.time()

        # Breach.
        r1 = _record(
            conn, value=80.0, source_timestamp=now,
            idempotency_key="cycle-b1",
        )
        assert len(r1.events) == 2

        # Recovery.
        r2 = _record(
            conn, value=97.0, source_timestamp=now + 1,
            idempotency_key="cycle-r1",
        )
        assert len(r2.events) == 1
        assert r2.events[0]["event_type"] == "goal.achieved"

        # Re-breach.
        r3 = _record(
            conn, value=80.0, source_timestamp=now + 2,
            idempotency_key="cycle-b2",
        )
        assert len(r3.events) == 2
        types = {e["event_type"] for e in r3.events}
        assert types == {"goal.gap_detected", "metric.threshold_crossed"}
        conn.close()

    def test_lower_is_better_direction(self):
        """Lower-is-better goals compute gap and threshold correctly."""
        conn = _fresh_db()
        _make_active_goal(
            conn, target_value=10.0, tolerance=2.0,
            direction="lower_is_better",
        )
        now = time.time()

        # Value 8 → gap = 8-10 = -2 (healthy, better than target)
        r1 = _record(conn, value=8.0, source_timestamp=now, idempotency_key="l1")
        assert r1.measurement.edge_state == "healthy"

        # Value 15 → gap = 15-10 = 5, tolerance=2 → breached
        r2 = _record(
            conn, value=15.0, source_timestamp=now + 1,
            idempotency_key="l2",
        )
        assert r2.measurement.edge_state == "breached"
        assert len(r2.events) == 2
        conn.close()

    def test_goal_without_target_no_gap_evaluation(self):
        """Goal with target_value=None skips gap evaluation."""
        conn = _fresh_db()
        _make_active_goal(conn, target_value=None)
        result = _record(conn, value=42.0)
        assert result.measurement.edge_state == "unknown"
        assert result.measurement.gap_value is None
        assert len(result.events) == 0
        conn.close()


# ---------------------------------------------------------------------------
# Event bus integration tests
# ---------------------------------------------------------------------------

class TestEventBusIntegration:
    """Events are published to the P0a event bus correctly."""

    def test_events_in_hermes_events_table(self):
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=0.0)
        now = time.time()
        result = _record(
            conn, value=50.0, source_timestamp=now,
            idempotency_key="bus-test",
        )
        # Check events exist in hermes_events.
        rows = conn.execute(
            "SELECT * FROM hermes_events WHERE source = 'goal_metrics'"
        ).fetchall()
        assert len(rows) >= 2
        types = {r["event_type"] for r in rows}
        assert "goal.gap_detected" in types
        assert "metric.threshold_crossed" in types
        conn.close()

    def test_event_payload_has_required_fields(self):
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=5.0)
        now = time.time()
        _record(
            conn, value=80.0, source_timestamp=now,
            idempotency_key="payload-test",
        )
        rows = conn.execute(
            "SELECT payload FROM hermes_events WHERE source = 'goal_metrics'"
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            assert "goal_id" in payload
            assert "metric_id" in payload
            assert "value" in payload
            assert "threshold" in payload
            assert "direction" in payload
            assert "source_timestamp" in payload
            # No forbidden fields.
            for forbidden in ("title", "message", "raw", "content", "body"):
                assert forbidden not in payload
        conn.close()

    def test_no_forbidden_values_in_events(self):
        """Verify no forbidden payload keys leak into hermes_events."""
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=0.0, title="My Goal")
        now = time.time()
        _record(
            conn, value=50.0, source_timestamp=now,
            idempotency_key="forbidden-test",
        )
        rows = conn.execute(
            "SELECT payload FROM hermes_events"
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            # No title, message, body, raw, name, etc.
            for key in payload:
                assert key not in (
                    "title", "message", "body", "raw", "content",
                    "name", "email", "phone", "patient",
                )
        conn.close()

    def test_subscribers_poll_goal_events(self):
        """Two independent subscribers can poll and ACK goal events."""
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=0.0)

        # Register two subscribers.
        sub1_cursor = subscribe(
            conn, "sub-1",
            event_types=["goal.gap_detected"],
            start="beginning",
        )
        sub2_cursor = subscribe(
            conn, "sub-2",
            event_types=["metric.threshold_crossed"],
            start="beginning",
        )

        now = time.time()
        _record(
            conn, value=50.0, source_timestamp=now,
            idempotency_key="sub-test",
        )

        events1 = poll(conn, "sub-1")
        events2 = poll(conn, "sub-2")

        assert len(events1) >= 1
        assert events1[0].event_type == "goal.gap_detected"
        assert len(events2) >= 1
        assert events2[0].event_type == "metric.threshold_crossed"
        conn.close()


# ---------------------------------------------------------------------------
# Board snapshot adapter tests
# ---------------------------------------------------------------------------

class TestBoardSnapshotAdapter:
    """Read-only Kanban adapter for board counts."""

    def _make_board_db(self) -> sqlite3.Connection:
        """Create a mock board DB with tasks table matching kanban schema."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name, autocommit=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # Minimal tasks table.
        conn.execute(
            "CREATE TABLE tasks ("
            " id TEXT PRIMARY KEY,"
            " title TEXT,"
            " status TEXT NOT NULL,"
            " assignee TEXT,"
            " created_at INTEGER NOT NULL"
            ")"
        )
        migrate_event_bus(conn)
        migrate_goal_schema(conn)
        return conn

    def _insert_task(self, conn, task_id, status):
        conn.execute(
            "INSERT INTO tasks (id, title, status, assignee, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, f"Task {task_id}", status, "builder", int(time.time())),
        )

    def test_snapshot_counts(self):
        conn = self._make_board_db()
        self._insert_task(conn, "t1", "running")
        self._insert_task(conn, "t2", "blocked")
        self._insert_task(conn, "t3", "done")
        self._insert_task(conn, "t4", "archived")
        self._insert_task(conn, "t5", "blocked")

        snap = snapshot_board_counts(conn)
        assert snap.blocked_count == 2
        assert snap.non_archived_count == 4  # t1+t2+t3+t5
        assert snap.snapshot_id is not None
        assert snap.source_timestamp > 0
        conn.close()

    def test_snapshot_empty_board(self):
        conn = self._make_board_db()
        snap = snapshot_board_counts(conn)
        assert snap.blocked_count == 0
        assert snap.non_archived_count == 0
        conn.close()

    def test_snapshot_reconciles_with_board_stats_logic(self):
        """Counts match what board_stats would return."""
        conn = self._make_board_db()
        self._insert_task(conn, "t1", "running")
        self._insert_task(conn, "t2", "blocked")
        self._insert_task(conn, "t3", "ready")
        self._insert_task(conn, "t4", "todo")
        self._insert_task(conn, "t5", "archived")
        self._insert_task(conn, "t6", "done")
        self._insert_task(conn, "t7", "blocked")

        snap = snapshot_board_counts(conn)

        # Manual board_stats-style count:
        by_status = {}
        for row in conn.execute(
            "SELECT status, COUNT(*) AS n FROM tasks "
            "WHERE status != 'archived' GROUP BY status"
        ):
            by_status[row["status"]] = row["n"]

        expected_non_archived = sum(by_status.values())
        expected_blocked = by_status.get("blocked", 0)

        assert snap.non_archived_count == expected_non_archived
        assert snap.blocked_count == expected_blocked
        conn.close()

    def test_record_board_snapshot_as_measurement(self):
        """Record snapshot counts as a measurement against a goal."""
        conn = self._make_board_db()
        self._insert_task(conn, "t1", "running")
        self._insert_task(conn, "t2", "blocked")
        self._insert_task(conn, "t3", "blocked")

        # Create and activate a goal for blocked card count.
        _begin(conn)
        goal = create_goal(
            conn,
            goal_id="board-blocked",
            metric_id="board.blocked_card_count",
            source_id="kanban_board_adapter",
            direction="lower_is_better",
            target_value=0.0,
            tolerance=1.0,
            max_age_seconds=86400,
        )
        goal = activate_goal(
            conn, "board-blocked", "keion-approved", time.time(),
        )
        _commit(conn)

        snap = snapshot_board_counts(conn)

        _begin(conn)
        results = record_board_snapshot(conn, snap, "board-blocked")
        _commit(conn)

        assert len(results) == 1
        assert results[0].measurement.value == 2.0  # 2 blocked
        conn.close()

    def test_blocked_never_exceeds_non_archived(self):
        """Invariant: blocked_count <= non_archived_count for any board state."""
        conn = self._make_board_db()
        # All blocked.
        for i in range(5):
            self._insert_task(conn, f"t{i}", "blocked")

        snap = snapshot_board_counts(conn)
        assert snap.blocked_count <= snap.non_archived_count
        assert snap.blocked_count == 5
        assert snap.non_archived_count == 5

        # Add non-blocked tasks.
        self._insert_task(conn, "t10", "running")
        self._insert_task(conn, "t11", "done")
        snap2 = snapshot_board_counts(conn)
        assert snap2.blocked_count <= snap2.non_archived_count
        assert snap2.blocked_count == 5
        assert snap2.non_archived_count == 7
        conn.close()

    def test_snapshot_atomic_under_wal_interleaving(self):
        """Deterministic two-connection WAL race: blocked_count must never
        exceed non_archived_count.

        Reproduces the exact P0b defect: the old code performed two separate
        SELECTs on an autocommit connection, allowing a WAL writer to inject
        rows between them.  The fixed code uses a single-statement aggregate
        that cannot be split.

        Scenario:
          Before: 1 task ('ready'), counts = (1, 0)
          Injected between the two (old) SELECT points:
            DELETE the 'ready' task, INSERT 2 'blocked' tasks
          After: 2 tasks ('blocked', 'blocked'), counts = (2, 2)

        Old code returned (1, 2) — impossible: blocked > total.
        Fixed code must return either (1, 0) or (2, 2) — a valid snapshot.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()

        # Reader connection — autocommit, matching canonical kanban connection.
        reader = sqlite3.connect(tmp.name, isolation_level=None)
        reader.row_factory = sqlite3.Row
        reader.execute("PRAGMA journal_mode=WAL")

        # Writer connection — separate WAL reader/writer.
        writer = sqlite3.connect(tmp.name, isolation_level=None)
        writer.execute("PRAGMA journal_mode=WAL")

        # Set up the board: one 'ready' task.
        reader.execute(
            "CREATE TABLE tasks ("
            " id TEXT PRIMARY KEY,"
            " title TEXT,"
            " status TEXT NOT NULL,"
            " assignee TEXT,"
            " created_at INTEGER NOT NULL"
            ")"
        )
        reader.execute(
            "INSERT INTO tasks (id, title, status, assignee, created_at) "
            "VALUES ('t3', 'Task t3', 'ready', 'builder', 1000)"
        )

        # Arm the trace callback to inject writes via the writer connection
        # when the reader executes any statement mentioning 'blocked'.
        # With the OLD two-statement code, this fires between the SELECTs;
        # with the NEW single-statement code, the injection cannot split
        # the read because both counts come from one implicit transaction.
        triggered = False

        def inject_race(statement):
            nonlocal triggered
            if not triggered and "'blocked'" in statement:
                triggered = True
                writer.execute("DELETE FROM tasks WHERE id='t3'")
                writer.execute(
                    "INSERT INTO tasks (id, title, status, assignee, created_at) "
                    "VALUES ('t4', 'Task t4', 'blocked', 'builder', 1001)"
                )
                writer.execute(
                    "INSERT INTO tasks (id, title, status, assignee, created_at) "
                    "VALUES ('t5', 'Task t5', 'blocked', 'builder', 1002)"
                )

        reader.set_trace_callback(inject_race)
        snap = snapshot_board_counts(reader)
        reader.set_trace_callback(None)

        # The trace callback did fire (it matched 'blocked' in the query).
        assert triggered, "trace callback must have been triggered"

        # The invariant: blocked <= non_archived in any valid snapshot.
        assert snap.blocked_count <= snap.non_archived_count, (
            f"impossible snapshot: blocked={snap.blocked_count} > "
            f"non_archived={snap.non_archived_count}"
        )

        # Must be one of the two valid board states, not the old impossible (1,2).
        valid_states = {(1, 0), (2, 2)}
        actual = (snap.non_archived_count, snap.blocked_count)
        assert actual in valid_states, (
            f"snapshot {actual} is not a valid board state; "
            f"expected one of {valid_states}"
        )

        reader.close()
        writer.close()

    def test_snapshot_under_caller_owned_transaction(self):
        """snapshot_board_counts participates in the caller's transaction
        read view without starting a nested transaction or emitting
        COMMIT/ROLLBACK."""
        conn = self._make_board_db()
        self._insert_task(conn, "t1", "blocked")
        self._insert_task(conn, "t2", "running")

        # Start a write transaction on this connection.
        conn.execute("BEGIN IMMEDIATE")
        assert conn.in_transaction

        # snapshot_board_counts must work inside the caller's txn.
        snap = snapshot_board_counts(conn)
        assert snap.blocked_count == 1
        assert snap.non_archived_count == 2

        # The caller's transaction is still open.
        assert conn.in_transaction
        conn.execute("COMMIT")
        conn.close()

    def test_snapshot_equality_with_board_stats_at_same_snapshot(self):
        """Snapshot counts at the same read view must equal board_stats."""
        conn = self._make_board_db()
        self._insert_task(conn, "t1", "running")
        self._insert_task(conn, "t2", "blocked")
        self._insert_task(conn, "t3", "ready")
        self._insert_task(conn, "t4", "blocked")
        self._insert_task(conn, "t5", "archived")
        self._insert_task(conn, "t6", "done")

        # Take snapshot and manual board_stats in the same transaction
        # to guarantee the same read view.
        conn.execute("BEGIN")
        snap = snapshot_board_counts(conn)
        bs_total = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status != 'archived'"
        ).fetchone()[0]
        bs_blocked = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'blocked'"
        ).fetchone()[0]
        conn.execute("COMMIT")

        assert snap.non_archived_count == bs_total
        assert snap.blocked_count == bs_blocked
        assert snap.blocked_count <= snap.non_archived_count


# ---------------------------------------------------------------------------
# Rollback atomicity tests
# ---------------------------------------------------------------------------

class TestRollbackAtomicity:
    """Measurements and events are rolled back together."""

    def test_rollback_removes_measurement_and_events(self):
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=0.0)
        now = time.time()

        _begin(conn)
        result = record_measurement(
            conn,
            goal_id="test-goal",
            metric_id="test.metric",
            value=50.0,
            source_id="test_source",
            source_timestamp=now,
            observed_at=now,
            source_ref="ref",
            idempotency_key="rollback-test",
        )
        assert len(result.events) == 2
        _rollback(conn)

        # Nothing should be persisted.
        m_count = conn.execute(
            "SELECT COUNT(*) FROM goal_measurements"
        ).fetchone()[0]
        assert m_count == 0

        e_count = conn.execute(
            "SELECT COUNT(*) FROM hermes_events WHERE source = 'goal_metrics'"
        ).fetchone()[0]
        assert e_count == 0
        conn.close()


# ---------------------------------------------------------------------------
# Concurrent retry tests
# ---------------------------------------------------------------------------

class TestConcurrentRetries:
    """Idempotency under concurrent retry scenarios."""

    def test_two_identical_inserts_one_wins(self):
        """Two attempts with same idempotency key — second is replay."""
        conn = _fresh_db()
        _make_active_goal(conn, target_value=100.0, tolerance=5.0)
        now = time.time()

        r1 = _record(
            conn, value=80.0, source_timestamp=now,
            payload_hash="same", idempotency_key="race",
        )
        r2 = _record(
            conn, value=80.0, source_timestamp=now,
            payload_hash="same", idempotency_key="race",
        )
        assert not r1.is_replay
        assert r2.is_replay
        assert r1.measurement.id == r2.measurement.id

        # Only one measurement row.
        count = conn.execute(
            "SELECT COUNT(*) FROM goal_measurements WHERE goal_id = 'test-goal'"
        ).fetchone()[0]
        assert count == 1
        conn.close()


# ---------------------------------------------------------------------------
# Direction and tolerance boundary cases
# ---------------------------------------------------------------------------

class TestBoundaryCases:
    """Boundary conditions for direction, tolerance, gap."""

    def test_zero_gap_is_healthy(self):
        """Value exactly at target with zero tolerance is healthy."""
        conn = _fresh_db()
        _make_active_goal(
            conn, target_value=100.0, tolerance=0.0,
            direction="higher_is_better",
        )
        result = _record(conn, value=100.0, idempotency_key="exact")
        assert result.measurement.edge_state == "healthy"
        assert len(result.events) == 0
        conn.close()

    def test_barely_breached(self):
        """Value barely exceeding tolerance triggers breach."""
        conn = _fresh_db()
        _make_active_goal(
            conn, target_value=100.0, tolerance=5.0,
            direction="higher_is_better",
        )
        now = time.time()
        # Value = 94.9 → gap = 5.1 > tolerance 5.0 → breached
        result = _record(
            conn, value=94.9, source_timestamp=now,
            idempotency_key="barely",
        )
        assert result.measurement.edge_state == "breached"
        conn.close()

    def test_exactly_at_tolerance_boundary(self):
        """Value exactly at tolerance boundary is healthy."""
        conn = _fresh_db()
        _make_active_goal(
            conn, target_value=100.0, tolerance=5.0,
            direction="higher_is_better",
        )
        # Value = 95.0 → gap = 5.0 = tolerance → healthy
        result = _record(conn, value=95.0, idempotency_key="boundary")
        assert result.measurement.edge_state == "healthy"
        conn.close()

    def test_large_tolerance_all_healthy(self):
        """Large tolerance means everything is healthy."""
        conn = _fresh_db()
        _make_active_goal(
            conn, target_value=100.0, tolerance=1000.0,
        )
        result = _record(conn, value=0.0, idempotency_key="large-tol")
        assert result.measurement.edge_state == "healthy"
        conn.close()

    def test_threshold_in_event_payload_higher(self):
        """For higher_is_better, threshold = target - tolerance."""
        conn = _fresh_db()
        _make_active_goal(
            conn, target_value=100.0, tolerance=5.0,
            direction="higher_is_better",
        )
        now = time.time()
        result = _record(
            conn, value=80.0, source_timestamp=now,
            idempotency_key="thresh-h",
        )
        # Check event payload.
        row = conn.execute(
            "SELECT payload FROM hermes_events "
            "WHERE source = 'goal_metrics' AND event_type = 'metric.threshold_crossed'"
        ).fetchone()
        payload = json.loads(row["payload"])
        assert payload["threshold"] == 95.0  # 100 - 5
        assert payload["direction"] == "higher_is_better"
        conn.close()

    def test_threshold_in_event_payload_lower(self):
        """For lower_is_better, threshold = target + tolerance."""
        conn = _fresh_db()
        _make_active_goal(
            conn, target_value=10.0, tolerance=2.0,
            direction="lower_is_better",
        )
        now = time.time()
        result = _record(
            conn, value=20.0, source_timestamp=now,
            idempotency_key="thresh-l",
        )
        row = conn.execute(
            "SELECT payload FROM hermes_events "
            "WHERE source = 'goal_metrics' AND event_type = 'metric.threshold_crossed'"
        ).fetchone()
        payload = json.loads(row["payload"])
        assert payload["threshold"] == 12.0  # 10 + 2
        assert payload["direction"] == "lower_is_better"
        conn.close()


# ---------------------------------------------------------------------------
# Bus subscriber / poll / ACK / replay after reopen
# ---------------------------------------------------------------------------

class TestBusReplayAfterReopen:
    """Events survive connection close/reopen and can be replayed."""

    def test_events_survive_reopen(self):
        """Events persisted to disk survive close + reopen."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

        # First session: create, record, emit events.
        conn1 = sqlite3.connect(db_path, autocommit=True)
        conn1.row_factory = sqlite3.Row
        conn1.execute("PRAGMA journal_mode=WAL")
        conn1.execute("PRAGMA foreign_keys=ON")
        migrate_event_bus(conn1)
        migrate_goal_schema(conn1)
        conn1.autocommit = False
        conn1.execute("COMMIT")

        _begin(conn1)
        create_goal(
            conn1, goal_id="persist-goal",
            metric_id="m", source_id="s",
            target_value=100.0, tolerance=0.0, max_age_seconds=86400,
        )
        activate_goal(
            conn1, "persist-goal", "evidence", time.time(),
        )
        _commit(conn1)

        now = time.time()
        _begin(conn1)
        record_measurement(
            conn1,
            goal_id="persist-goal",
            metric_id="m",
            value=50.0,
            source_id="s",
            source_timestamp=now,
            observed_at=now,
            source_ref="ref",
            idempotency_key="persist-k",
        )
        _commit(conn1)
        conn1.close()

        # Second session: reopen and verify.
        conn2 = sqlite3.connect(db_path, autocommit=True)
        conn2.row_factory = sqlite3.Row
        conn2.execute("PRAGMA journal_mode=WAL")
        conn2.execute("PRAGMA foreign_keys=ON")

        # Subscribe from beginning.
        sub_cursor = subscribe(
            conn2, "replay-sub",
            event_types=["goal.gap_detected", "metric.threshold_crossed"],
            start="beginning",
        )
        events = poll(conn2, "replay-sub")
        assert len(events) >= 2
        types = {e.event_type for e in events}
        assert "goal.gap_detected" in types
        assert "metric.threshold_crossed" in types
        conn2.close()


# ---------------------------------------------------------------------------
# Legacy board (with task_events retained) test
# ---------------------------------------------------------------------------

class TestLegacyBoardCoexistence:
    """Goal metrics coexist with existing task_events table."""

    def test_existing_task_events_not_modified(self):
        """Migration doesn't touch an existing task_events table."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name, autocommit=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        # Create a fake task_events table.
        conn.execute(
            "CREATE TABLE task_events ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " task_id TEXT NOT NULL,"
            " kind TEXT NOT NULL,"
            " payload TEXT,"
            " created_at INTEGER NOT NULL"
            ")"
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES ('t1', 'created', '{}', 1234567890)"
        )

        # Run both migrations.
        migrate_event_bus(conn)
        migrate_goal_schema(conn)

        # task_events should still have exactly one row.
        count = conn.execute(
            "SELECT COUNT(*) FROM task_events"
        ).fetchone()[0]
        assert count == 1

        # And goal tables should exist independently.
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "goals" in tables
        assert "goal_measurements" in tables
        assert "task_events" in tables
        conn.close()

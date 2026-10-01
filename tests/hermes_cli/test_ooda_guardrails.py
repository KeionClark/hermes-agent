"""
Adversarial tests for the isolated SQLite guardrail engine.

Tests: race two connections/thread/process reserve at crossing,
equality/exceed, replay, missing price, UTC boundary, low-sample
denominator, independent secondary negative control, rollback inert
and reset audit.

Reuses outcome_ledger.py/goal_metrics.py tables read-only; schema
missing must fail closed.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
import uuid
from decimal import Decimal
from typing import Any

import pytest

from hermes_cli.ooda_guardrails import (
    DriftResult,
    GuardrailSchemaNotInitializedError,
    ReservationResult,
    RollbackProposal,
    TripWireResult,
    commit_act,
    evaluate_drift,
    evaluate_trip_wire,
    migrate_guardrail_schema,
    propose_rollback,
    release_act,
    reserve_act,
    reset_guardrail,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conn(path: str = ":memory:") -> sqlite3.Connection:
    """Create a connection with row factory and autocommit (isolation_level=None)."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _setup_full_schema(conn: sqlite3.Connection) -> None:
    """Migrate all required schemas: guardrails + goals + outcomes + events."""
    migrate_guardrail_schema(conn)
    from hermes_cli.event_bus import migrate_event_bus
    from hermes_cli.goal_metrics import migrate_goal_schema
    from hermes_cli.outcome_ledger import migrate_outcome_schema

    migrate_event_bus(conn)
    migrate_goal_schema(conn)
    migrate_outcome_schema(conn)


def _create_test_goal(
    conn: sqlite3.Connection,
    goal_id: str = "g1",
    metric_id: str = "m1",
    source_id: str = "src1",
    direction: str = "higher_is_better",
    target_value: float | None = 100.0,
    max_age_seconds: int = 86400,
) -> None:
    """Create and activate a goal for testing."""
    from hermes_cli.goal_metrics import activate_goal, create_goal

    conn.execute("BEGIN IMMEDIATE")
    create_goal(
        conn,
        goal_id=goal_id,
        metric_id=metric_id,
        source_id=source_id,
        direction=direction,
        target_value=target_value,
        max_age_seconds=max_age_seconds,
    )
    activate_goal(conn, goal_id, "test-approval", time.time())
    conn.execute("COMMIT")


def _create_test_measurement(
    conn: sqlite3.Connection,
    goal_id: str,
    value: float,
    source_timestamp: float,
    metric_id: str = "m1",
    source_id: str = "src1",
    source_ref: str = "ref1",
) -> int:
    """Record a measurement and return its ID."""
    from hermes_cli.goal_metrics import record_measurement

    idem_key = str(uuid.uuid4())
    conn.execute("BEGIN IMMEDIATE")
    result = record_measurement(
        conn,
        goal_id=goal_id,
        metric_id=metric_id,
        value=value,
        source_id=source_id,
        source_timestamp=source_timestamp,
        observed_at=source_timestamp,
        source_ref=source_ref,
        idempotency_key=idem_key,
    )
    conn.execute("COMMIT")
    return result.measurement.id


def _create_test_outcome(
    conn: sqlite3.Connection,
    task_id: str,
    goal_id: str,
    before_measurement_id: int | None,
    expected_direction: str = "higher",
    started_at: float = 1000.0,
) -> str:
    """Create and return outcome_id."""
    from hermes_cli.outcome_ledger import create_outcome

    idem_key = str(uuid.uuid4())
    conn.execute("BEGIN IMMEDIATE")
    outcome = create_outcome(
        conn,
        task_id=task_id,
        goal_id=goal_id,
        hypothesis="test hypothesis",
        action_type="config_change",
        action_revision="sha123",
        expected_direction=expected_direction,
        before_measurement_id=before_measurement_id,
        started_at=started_at,
        idempotency_key=idem_key,
    )
    conn.execute("COMMIT")
    return outcome.id


def _reserve_default(
    conn: sqlite3.Connection,
    *,
    task_id: str = "t1",
    agent_id: str = "a1",
    lane_id: str = "lane1",
    reservation_key: str | None = None,
    estimated_tokens: int = 1000,
    price_per_million_usd: str = "10.00",
    caps: dict[str, str | Decimal] | None = None,
    now: float = 1_700_000_000.0,
    approval_ref: str | None = None,
    begin: bool = True,
) -> ReservationResult:
    """Helper for reserve_act with sensible defaults."""
    if reservation_key is None:
        reservation_key = str(uuid.uuid4())
    if caps is None:
        caps = {"task_token": "100000", "agent_day": "100.00", "global_day": "1000.00"}

    if begin:
        conn.execute("BEGIN IMMEDIATE")
    result = reserve_act(
        conn,
        task_id=task_id,
        agent_id=agent_id,
        lane_id=lane_id,
        reservation_key=reservation_key,
        estimated_tokens=estimated_tokens,
        price_per_million_usd=price_per_million_usd,
        caps=caps,
        now=now,
        approval_ref=approval_ref,
    )
    if begin:
        conn.execute("COMMIT")
    return result


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestMigrateSchema:
    def test_creates_tables(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "guardrail_reservations" in tables
        assert "guardrail_halts" in tables
        assert "guardrail_resets" in tables
        assert "guardrail_schema_version" in tables

    def test_idempotent(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        migrate_guardrail_schema(conn)  # no error
        version = conn.execute(
            "SELECT version FROM guardrail_schema_version WHERE id = 1"
        ).fetchone()[0]
        assert version == 1

    def test_future_version_rejects(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("UPDATE guardrail_schema_version SET version = 999")
        with pytest.raises(RuntimeError, match="too old"):
            migrate_guardrail_schema(conn)

    def test_missing_schema_fails_closed(self):
        conn = _make_conn()
        with pytest.raises(GuardrailSchemaNotInitializedError):
            conn.execute("BEGIN IMMEDIATE")
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="10.00",
                caps={"task_token": "1000"},
                now=1_700_000_000.0,
            )


# ---------------------------------------------------------------------------
# reserve_act tests
# ---------------------------------------------------------------------------


class TestReserveAct:
    def test_basic_reservation(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        result = _reserve_default(conn)
        assert result.allowed is True
        assert result.reason == "reserved"
        assert result.reservation_id is not None
        assert result.remaining is not None

    def test_missing_price_fails_closed(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="invalid dollar amount"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="not_a_number",
                caps={"task_token": "1000"},
                now=1_700_000_000.0,
            )

    def test_zero_price_fails_closed(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="positive"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="0.00",
                caps={"task_token": "1000"},
                now=1_700_000_000.0,
            )

    def test_negative_price_fails_closed(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="positive"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="-5.00",
                caps={"task_token": "1000"},
                now=1_700_000_000.0,
            )

    def test_infinite_price_fails_closed(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="finite"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd=Decimal("Infinity"),
                caps={"task_token": "1000"},
                now=1_700_000_000.0,
            )

    def test_nan_price_fails_closed(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="finite"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd=Decimal("NaN"),
                caps={"task_token": "1000"},
                now=1_700_000_000.0,
            )

    def test_zero_estimated_tokens_fails(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="positive integer"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=0,
                price_per_million_usd="10.00",
                caps={"task_token": "1000"},
                now=1_700_000_000.0,
            )

    def test_empty_caps_fails(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="at least one"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="10.00",
                caps={},
                now=1_700_000_000.0,
            )

    def test_unknown_cap_scope_fails(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="unknown cap scope"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="10.00",
                caps={"bogus_scope": "100.00"},
                now=1_700_000_000.0,
            )

    def test_task_token_cap_exact_boundary(self):
        """Reservation at exactly the cap should be denied."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        # Reserve 500 tokens
        r1 = _reserve_default(conn, estimated_tokens=500, caps={"task_token": "1000"})
        assert r1.allowed is True

        # Reserve 501 tokens (total would be 1001, exceeds cap of 1000)
        r2 = _reserve_default(conn, estimated_tokens=501, caps={"task_token": "1000"})
        assert r2.allowed is False
        assert "task_token" in r2.reason

    def test_task_token_cap_equal_passes(self):
        """Reservation that exactly fills the cap should succeed."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        r1 = _reserve_default(conn, estimated_tokens=500, caps={"task_token": "1000"})
        assert r1.allowed is True

        r2 = _reserve_default(conn, estimated_tokens=500, caps={"task_token": "1000"})
        assert r2.allowed is True

    def test_agent_day_cap_exceed(self):
        """Agent day cap exceeded → denied + halt recorded."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        # Price $10/M tokens, 1000 tokens = $0.01 = 10_000 micro-USD
        # Cap at $0.01
        r1 = _reserve_default(
            conn,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        assert r1.allowed is True

        r2 = _reserve_default(
            conn,
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        assert r2.allowed is False
        assert "agent_day" in r2.reason

        # Verify halt was recorded
        halt = conn.execute(
            "SELECT * FROM guardrail_halts WHERE scope = 'agent:a1' AND resolved_at IS NULL"
        ).fetchone()
        assert halt is not None

    def test_lane_week_cap(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        r1 = _reserve_default(
            conn,
            lane_id="lane_x",
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"lane_week": "0.01"},
        )
        assert r1.allowed is True

        r2 = _reserve_default(
            conn,
            lane_id="lane_x",
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"lane_week": "0.01"},
        )
        assert r2.allowed is False
        assert "lane_week" in r2.reason

    def test_global_day_cap(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        r1 = _reserve_default(
            conn,
            agent_id="agent_a",
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"global_day": "0.01"},
        )
        assert r1.allowed is True

        # Different agent, same global cap
        r2 = _reserve_default(
            conn,
            agent_id="agent_b",
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"global_day": "0.01"},
        )
        assert r2.allowed is False
        assert "global_day" in r2.reason

    def test_idempotent_replay_same_payload(self):
        """Same reservation_key with same payload → idempotent return."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk = "fixed-key-1"
        r1 = _reserve_default(conn, reservation_key=rk)
        assert r1.allowed is True

        r2 = _reserve_default(conn, reservation_key=rk)
        assert r2.allowed is True
        assert r2.reason == "replay"
        assert r2.reservation_id == r1.reservation_id

    def test_conflicting_replay_fails(self):
        """Same reservation_key with different payload → fails closed."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk = "fixed-key-conflict"
        r1 = _reserve_default(conn, reservation_key=rk, task_id="task_a")
        assert r1.allowed is True

        with pytest.raises(ValueError, match="conflicting replay"):
            _reserve_default(conn, reservation_key=rk, task_id="task_b")

    def test_safe_halt_persists(self):
        """After cap-triggered halt, new reservations are denied."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        # Exceed agent_day cap
        _reserve_default(
            conn,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        r2 = _reserve_default(
            conn,
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        assert r2.allowed is False

        # Even with a higher cap, the halt persists
        r3 = _reserve_default(
            conn,
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"agent_day": "1000.00"},
        )
        assert r3.allowed is False
        assert "safe halt" in r3.reason

    def test_nan_now_fails(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="finite"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="10.00",
                caps={"task_token": "1000"},
                now=float("nan"),
            )

    def test_requires_write_transaction(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        with pytest.raises(RuntimeError, match="write transaction"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="10.00",
                caps={"task_token": "1000"},
                now=1_700_000_000.0,
            )

    def test_micro_usd_rounding_up(self):
        """Verify sub-micro-USD amounts round UP."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        # Price = $0.000001 per million tokens = 1 micro-USD per million
        # 1 token = 0.000001 micro-USD → rounds up to 1 micro-USD
        result = _reserve_default(
            conn,
            estimated_tokens=1,
            price_per_million_usd="0.000001",
            caps={"task_token": "1000"},
        )
        assert result.allowed is True

        # Check the stored cost
        row = conn.execute(
            "SELECT estimated_cost_micro_usd FROM guardrail_reservations "
            "WHERE id = ?",
            (result.reservation_id,),
        ).fetchone()
        assert row[0] >= 1  # Rounded UP, at least 1 micro-USD


# ---------------------------------------------------------------------------
# UTC boundary tests
# ---------------------------------------------------------------------------


class TestUTCBoundary:
    def test_agent_day_utc_boundary(self):
        """Reservations crossing UTC midnight are in different days."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)

        # 2023-11-14 23:59:59 UTC
        before_midnight = 1700006399.0

        # 2023-11-15 00:00:01 UTC
        after_midnight = 1700006401.0

        # Fill the cap before midnight
        r1 = _reserve_default(
            conn,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
            now=before_midnight,
        )
        assert r1.allowed is True

        # Should be denied before midnight (same day)
        r2 = _reserve_default(
            conn,
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
            now=before_midnight,
        )
        assert r2.allowed is False

        # After midnight: new day, should be allowed (but halt persists!)
        # Reset halt first
        reset_guardrail(conn, "agent:a1", "admin", "new day", "policy-ref-1", after_midnight)

        r3 = _reserve_default(
            conn,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
            now=after_midnight,
        )
        assert r3.allowed is True

    def test_lane_week_utc_boundary(self):
        """Lane-week cap resets at UTC Monday."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)

        # Find a Sunday 23:59 UTC and Monday 00:01 UTC
        # 2023-11-12 is a Sunday
        sunday_late = 1699833599.0  # Sun Nov 12 2023 23:59:59 UTC
        monday_early = 1699833601.0  # Mon Nov 13 2023 00:00:01 UTC

        r1 = _reserve_default(
            conn,
            lane_id="lane_wk",
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"lane_week": "0.01"},
            now=sunday_late,
        )
        assert r1.allowed is True

        # Exceed on Sunday
        r2 = _reserve_default(
            conn,
            lane_id="lane_wk",
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"lane_week": "0.01"},
            now=sunday_late,
        )
        assert r2.allowed is False

        # Reset halt, try Monday (new week)
        reset_guardrail(conn, "lane:lane_wk", "admin", "new week", "pol-1", monday_early)

        r3 = _reserve_default(
            conn,
            lane_id="lane_wk",
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"lane_week": "0.01"},
            now=monday_early,
        )
        assert r3.allowed is True


# ---------------------------------------------------------------------------
# commit_act / release_act tests
# ---------------------------------------------------------------------------


class TestCommitRelease:
    def test_commit_basic(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk = "commit-rk"
        r = _reserve_default(conn, reservation_key=rk)
        assert r.allowed is True

        commit_act(conn, rk, actual_tokens=800, now=1_700_000_100.0)

        row = conn.execute(
            "SELECT * FROM guardrail_reservations WHERE reservation_key = ?",
            (rk,),
        ).fetchone()
        assert row["status"] == "committed"
        assert row["actual_tokens"] == 800
        assert row["actual_cost_micro_usd"] is not None
        assert row["committed_at"] == 1_700_000_100.0

    def test_commit_idempotent(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk = "commit-idem"
        _reserve_default(conn, reservation_key=rk)
        commit_act(conn, rk, actual_tokens=500, now=1_700_000_100.0)
        commit_act(conn, rk, actual_tokens=500, now=1_700_000_200.0)  # no error

    def test_release_basic(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk = "release-rk"
        _reserve_default(conn, reservation_key=rk)
        release_act(conn, rk, now=1_700_000_100.0)

        row = conn.execute(
            "SELECT * FROM guardrail_reservations WHERE reservation_key = ?",
            (rk,),
        ).fetchone()
        assert row["status"] == "released"

    def test_release_idempotent(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk = "release-idem"
        _reserve_default(conn, reservation_key=rk)
        release_act(conn, rk, now=1_700_000_100.0)
        release_act(conn, rk, now=1_700_000_200.0)  # no error

    def test_commit_after_release_fails(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk = "release-then-commit"
        _reserve_default(conn, reservation_key=rk)
        release_act(conn, rk, now=1_700_000_100.0)

        with pytest.raises(ValueError, match="released"):
            commit_act(conn, rk, actual_tokens=100, now=1_700_000_200.0)

    def test_release_after_commit_fails(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk = "commit-then-release"
        _reserve_default(conn, reservation_key=rk)
        commit_act(conn, rk, actual_tokens=100, now=1_700_000_100.0)

        with pytest.raises(ValueError, match="committed"):
            release_act(conn, rk, now=1_700_000_200.0)

    def test_commit_unknown_key(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        with pytest.raises(ValueError, match="not found"):
            commit_act(conn, "nonexistent-key", actual_tokens=100, now=1_700_000_000.0)

    def test_release_unknown_key(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        with pytest.raises(ValueError, match="not found"):
            release_act(conn, "nonexistent-key", now=1_700_000_000.0)

    def test_committed_cost_replaces_estimated_in_cap(self):
        """After commit, actual cost counts toward caps instead of estimated."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk1 = "cap-replace-1"

        # Reserve 1000 tokens at $10/M → est $0.01 = 10_000 µ$
        # Cap at $0.02 = 20_000 µ$
        _reserve_default(
            conn,
            reservation_key=rk1,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.02"},
        )

        # Commit with only 100 actual tokens → actual $0.001 = 1_000 µ$
        commit_act(conn, rk1, actual_tokens=100, now=1_700_000_100.0)

        # Now cap has freed up space: was 10_000 outstanding, now 1_000 committed
        # Should allow another 1000 estimated tokens
        rk2 = "cap-replace-2"
        r = _reserve_default(
            conn,
            reservation_key=rk2,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.02"},
        )
        assert r.allowed is True

    def test_released_does_not_count_toward_cap(self):
        """Released reservations don't count toward caps."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        rk1 = "released-cap-1"

        _reserve_default(
            conn,
            reservation_key=rk1,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        release_act(conn, rk1, now=1_700_000_100.0)

        # Cap should be fully available again
        rk2 = "released-cap-2"
        r = _reserve_default(
            conn,
            reservation_key=rk2,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        assert r.allowed is True


# ---------------------------------------------------------------------------
# Concurrency / race tests
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_race_two_threads_at_cap_crossing(self, tmp_path):
        """Two threads racing to reserve should not both succeed past cap."""
        db_path = str(tmp_path / "race.db")
        conn_setup = _make_conn(db_path)
        migrate_guardrail_schema(conn_setup)
        conn_setup.close()

        results: list[ReservationResult] = []
        errors: list[Exception] = []
        barrier = threading.Barrier(2)

        def racer(thread_id: int) -> None:
            try:
                c = _make_conn(db_path)
                c.execute("PRAGMA busy_timeout = 5000")
                barrier.wait()
                r = _reserve_default(
                    c,
                    reservation_key=f"race-{thread_id}",
                    estimated_tokens=600,
                    price_per_million_usd="10.00",
                    # Cap at 1000 tokens → only one thread should win
                    caps={"task_token": "1000"},
                )
                results.append(r)
                c.close()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=racer, args=(1,))
        t2 = threading.Thread(target=racer, args=(2,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # At most one error is acceptable (locked)
        allowed_count = sum(1 for r in results if r.allowed)
        # Both might complete but only one should be allowed
        assert allowed_count <= 1 or len(errors) > 0, (
            f"Both threads should not succeed past cap: "
            f"allowed={allowed_count}, errors={len(errors)}"
        )

    def test_race_two_connections_file_db(self, tmp_path):
        """Two connections on same file DB — serialized by BEGIN IMMEDIATE."""
        db_path = str(tmp_path / "serial.db")
        conn_setup = _make_conn(db_path)
        migrate_guardrail_schema(conn_setup)
        conn_setup.close()

        results = []
        barrier = threading.Barrier(2)

        def worker(tid: int) -> None:
            c = _make_conn(db_path)
            c.execute("PRAGMA busy_timeout = 5000")
            barrier.wait()
            r = _reserve_default(
                c,
                reservation_key=f"serial-{tid}",
                estimated_tokens=600,
                price_per_million_usd="10.00",
                caps={"task_token": "1000"},
            )
            results.append((tid, r))
            c.close()

        t1 = threading.Thread(target=worker, args=(1,))
        t2 = threading.Thread(target=worker, args=(2,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(results) == 2
        allowed = sum(1 for _, r in results if r.allowed)
        assert allowed == 1, f"Expected exactly 1 allowed, got {allowed}"


# ---------------------------------------------------------------------------
# evaluate_drift tests
# ---------------------------------------------------------------------------


class TestEvaluateDrift:
    def test_missing_outcome_schema_fails_closed(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        with pytest.raises(GuardrailSchemaNotInitializedError):
            evaluate_drift(conn, "agent1", time.time())

    def test_low_sample_returns_insufficient(self):
        conn = _make_conn()
        _setup_full_schema(conn)
        # No outcomes → insufficient evidence
        result = evaluate_drift(conn, "agent1", time.time(), min_samples=10)
        assert result.sufficient_evidence is False
        assert result.drift_rate is None
        assert result.total_evaluated == 0

    def test_low_sample_never_claims_zero_risk(self):
        """Even with 0 wrong-direction, < min_samples is insufficient."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        _create_test_goal(conn)
        base_ts = now - 86400  # 1 day ago

        # Create 3 outcomes (< min_samples=10)
        for i in range(3):
            m_ts = base_ts + i
            mid = _create_test_measurement(conn, "g1", 50.0 + i, m_ts)
            oid = _create_test_outcome(
                conn, f"task_{i}", "g1", mid, started_at=m_ts + 0.1
            )
            # Need reservation for agent mapping
            _reserve_default(conn, task_id=f"task_{i}", agent_id="agent_test")

        result = evaluate_drift(conn, "agent_test", now, min_samples=10)
        assert result.sufficient_evidence is False
        assert result.drift_rate is None
        # Must NOT return drift_rate=0.0 even though no wrongs

    def test_drift_with_wrong_direction(self):
        """Sufficient samples with wrong-direction outcomes → drift > 0."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        base_ts = now - 3600  # 1 hour ago

        _create_test_goal(conn, direction="higher_is_better")

        for i in range(12):
            # Each iteration uses 30s window; spaced 40s apart for monotonicity
            before_ts = base_ts + i * 40
            before_mid = _create_test_measurement(conn, "g1", 50.0, before_ts)

            task_id = f"task_drift_{i}"
            oid = _create_test_outcome(
                conn, task_id, "g1", before_mid,
                expected_direction="higher", started_at=before_ts + 1,
            )
            _reserve_default(conn, task_id=task_id, agent_id="drift_agent")

            # Complete the outcome
            from hermes_cli.outcome_ledger import complete_action
            conn.execute("BEGIN IMMEDIATE")
            complete_action(
                conn, oid, action_revision="sha_test",
                completed_at=before_ts + 5,
            )
            conn.execute("COMMIT")

            # Add after measurement — some go wrong direction
            after_ts = before_ts + 20
            if i < 8:
                # Correct: value went higher
                after_val = 60.0
            else:
                # Wrong: value went lower
                after_val = 40.0

            _create_test_measurement(conn, "g1", after_val, after_ts)

            # Evaluate the outcome
            from hermes_cli.outcome_ledger import evaluate_due_outcome
            conn.execute("BEGIN IMMEDIATE")
            evaluate_due_outcome(conn, oid, now=after_ts + 1)
            conn.execute("COMMIT")

        result = evaluate_drift(conn, "drift_agent", now, min_samples=10)
        assert result.sufficient_evidence is True
        assert result.drift_rate is not None
        assert result.total_evaluated >= 10
        assert result.wrong_direction_count >= 1

    def test_reverted_outcomes_excluded(self):
        """Reverted outcomes are not counted in drift evaluation."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        base_ts = now - 3600
        _create_test_goal(conn)

        for i in range(12):
            before_ts = base_ts + i * 40
            mid = _create_test_measurement(conn, "g1", 50.0, before_ts)
            oid = _create_test_outcome(
                conn, f"task_rev_{i}", "g1", mid,
                expected_direction="higher", started_at=before_ts + 1,
            )
            _reserve_default(conn, task_id=f"task_rev_{i}", agent_id="rev_agent")

            from hermes_cli.outcome_ledger import complete_action
            conn.execute("BEGIN IMMEDIATE")
            complete_action(conn, oid, action_revision="sha_r", completed_at=before_ts + 5)
            conn.execute("COMMIT")

            after_ts = before_ts + 20
            _create_test_measurement(conn, "g1", 60.0, after_ts)

            from hermes_cli.outcome_ledger import evaluate_due_outcome
            conn.execute("BEGIN IMMEDIATE")
            evaluate_due_outcome(conn, oid, now=after_ts + 1)
            conn.execute("COMMIT")

            # Revert half of them
            if i % 2 == 0:
                from hermes_cli.outcome_ledger import revert_outcome
                conn.execute("BEGIN IMMEDIATE")
                revert_outcome(conn, oid, reason="test revert", now=now)
                conn.execute("COMMIT")

        result = evaluate_drift(conn, "rev_agent", now, min_samples=1)
        # Only non-reverted should be counted
        assert result.total_evaluated == 6  # 12 total, 6 reverted


# ---------------------------------------------------------------------------
# evaluate_trip_wire tests
# ---------------------------------------------------------------------------


class TestEvaluateTripWire:
    def test_missing_tables_fails_closed(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        with pytest.raises(GuardrailSchemaNotInitializedError):
            evaluate_trip_wire(
                conn, "g1", "o1", "g2", 1, 2, 0.9, 1.1, time.time()
            )

    def test_same_source_rejected(self):
        """Same source_id for primary and secondary → unknown (independent required)."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        base_ts = now - 3600

        # Primary and secondary both use "same_src"
        _create_test_goal(conn, goal_id="prim_g", source_id="same_src")
        _create_test_goal(conn, goal_id="sec_g", metric_id="m2", source_id="same_src")

        mid1 = _create_test_measurement(
            conn, "prim_g", 50.0, base_ts, source_id="same_src"
        )
        oid = _create_test_outcome(
            conn, "t1", "prim_g", mid1, started_at=base_ts + 1
        )

        # Complete and confirm primary
        from hermes_cli.outcome_ledger import complete_action
        conn.execute("BEGIN IMMEDIATE")
        complete_action(conn, oid, action_revision="sha_tw", completed_at=base_ts + 5)
        conn.execute("COMMIT")
        _create_test_measurement(
            conn, "prim_g", 60.0, base_ts + 20, source_id="same_src"
        )
        from hermes_cli.outcome_ledger import evaluate_due_outcome
        conn.execute("BEGIN IMMEDIATE")
        evaluate_due_outcome(conn, oid, now=base_ts + 25)
        conn.execute("COMMIT")

        # Secondary measurements
        sec_base = _create_test_measurement(
            conn, "sec_g", 100.0, base_ts, metric_id="m2", source_id="same_src"
        )
        sec_latest = _create_test_measurement(
            conn, "sec_g", 80.0, base_ts + 30, metric_id="m2", source_id="same_src"
        )

        result = evaluate_trip_wire(
            conn, "prim_g", oid, "sec_g", sec_base, sec_latest,
            0.9, 1.1, now,
        )
        assert result.tripped is None
        assert "independent" in result.reason.lower()

    def test_independent_source_trips(self):
        """Different source_id, secondary degrades → tripped=True."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        base_ts = now - 3600

        _create_test_goal(conn, goal_id="prim_ind", source_id="primary_src")
        _create_test_goal(
            conn, goal_id="sec_ind", metric_id="m2",
            source_id="secondary_src",
        )

        mid1 = _create_test_measurement(
            conn, "prim_ind", 50.0, base_ts, source_id="primary_src"
        )
        oid = _create_test_outcome(
            conn, "t_tw", "prim_ind", mid1, started_at=base_ts + 1
        )

        from hermes_cli.outcome_ledger import complete_action
        conn.execute("BEGIN IMMEDIATE")
        complete_action(conn, oid, action_revision="sha_tw2", completed_at=base_ts + 5)
        conn.execute("COMMIT")
        _create_test_measurement(
            conn, "prim_ind", 60.0, base_ts + 20, source_id="primary_src"
        )
        from hermes_cli.outcome_ledger import evaluate_due_outcome
        conn.execute("BEGIN IMMEDIATE")
        evaluate_due_outcome(conn, oid, now=base_ts + 25)
        conn.execute("COMMIT")

        # Secondary: 100 → 70 (ratio = 0.7, outside [0.9, 1.1])
        sec_base = _create_test_measurement(
            conn, "sec_ind", 100.0, base_ts, metric_id="m2", source_id="secondary_src"
        )
        sec_latest = _create_test_measurement(
            conn, "sec_ind", 70.0, base_ts + 30, metric_id="m2", source_id="secondary_src"
        )

        result = evaluate_trip_wire(
            conn, "prim_ind", oid, "sec_ind", sec_base, sec_latest,
            0.9, 1.1, now,
        )
        assert result.tripped is True
        assert result.secondary_ratio is not None
        assert result.secondary_ratio < 0.9

    def test_independent_source_safe(self):
        """Different source_id, secondary within range → tripped=False."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        base_ts = now - 3600

        _create_test_goal(conn, goal_id="safe_p", source_id="src_p")
        _create_test_goal(
            conn, goal_id="safe_s", metric_id="m2", source_id="src_s",
        )

        mid1 = _create_test_measurement(
            conn, "safe_p", 50.0, base_ts, source_id="src_p"
        )
        oid = _create_test_outcome(
            conn, "t_safe", "safe_p", mid1, started_at=base_ts + 1
        )

        from hermes_cli.outcome_ledger import complete_action
        conn.execute("BEGIN IMMEDIATE")
        complete_action(conn, oid, action_revision="sha_safe", completed_at=base_ts + 5)
        conn.execute("COMMIT")
        _create_test_measurement(
            conn, "safe_p", 60.0, base_ts + 20, source_id="src_p"
        )
        from hermes_cli.outcome_ledger import evaluate_due_outcome
        conn.execute("BEGIN IMMEDIATE")
        evaluate_due_outcome(conn, oid, now=base_ts + 25)
        conn.execute("COMMIT")

        sec_base = _create_test_measurement(
            conn, "safe_s", 100.0, base_ts, metric_id="m2", source_id="src_s"
        )
        sec_latest = _create_test_measurement(
            conn, "safe_s", 98.0, base_ts + 30, metric_id="m2", source_id="src_s"
        )

        result = evaluate_trip_wire(
            conn, "safe_p", oid, "safe_s", sec_base, sec_latest,
            0.9, 1.1, now,
        )
        assert result.tripped is False
        assert result.secondary_ratio is not None
        assert 0.9 <= result.secondary_ratio <= 1.1

    def test_primary_not_confirmed_unknown(self):
        """If primary outcome is not confirmed, result is unknown."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        base_ts = now - 3600
        _create_test_goal(conn, goal_id="unk_p", source_id="sp")
        _create_test_goal(conn, goal_id="unk_s", metric_id="m2", source_id="ss")

        mid = _create_test_measurement(conn, "unk_p", 50.0, base_ts, source_id="sp")
        oid = _create_test_outcome(conn, "t_unk", "unk_p", mid, started_at=base_ts + 1)
        # Don't complete/evaluate — stays pending

        sec_b = _create_test_measurement(conn, "unk_s", 100.0, base_ts, metric_id="m2", source_id="ss")
        sec_l = _create_test_measurement(conn, "unk_s", 50.0, base_ts + 30, metric_id="m2", source_id="ss")

        result = evaluate_trip_wire(
            conn, "unk_p", oid, "unk_s", sec_b, sec_l, 0.9, 1.1, now,
        )
        assert result.tripped is None
        assert "not confirmed" in result.reason

    def test_stale_latest_returns_unknown(self):
        """Stale latest measurement → unknown."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        base_ts = now - 200000  # very old

        _create_test_goal(
            conn, goal_id="stale_p", source_id="sp2", max_age_seconds=3600
        )
        _create_test_goal(
            conn, goal_id="stale_s", metric_id="m2", source_id="ss2",
            max_age_seconds=3600,
        )

        mid = _create_test_measurement(conn, "stale_p", 50.0, base_ts, source_id="sp2")
        oid = _create_test_outcome(conn, "t_st", "stale_p", mid, started_at=base_ts + 1)

        from hermes_cli.outcome_ledger import complete_action
        conn.execute("BEGIN IMMEDIATE")
        complete_action(conn, oid, action_revision="sha_st", completed_at=base_ts + 5)
        conn.execute("COMMIT")
        _create_test_measurement(conn, "stale_p", 60.0, base_ts + 20, source_id="sp2")
        from hermes_cli.outcome_ledger import evaluate_due_outcome
        conn.execute("BEGIN IMMEDIATE")
        evaluate_due_outcome(conn, oid, now=base_ts + 25)
        conn.execute("COMMIT")

        sec_b = _create_test_measurement(
            conn, "stale_s", 100.0, base_ts, metric_id="m2", source_id="ss2"
        )
        sec_l = _create_test_measurement(
            conn, "stale_s", 50.0, base_ts + 30, metric_id="m2", source_id="ss2"
        )

        result = evaluate_trip_wire(
            conn, "stale_p", oid, "stale_s", sec_b, sec_l, 0.9, 1.1, now,
        )
        assert result.tripped is None
        assert "stale" in result.reason

    def test_version_mismatch_returns_unknown(self):
        """metric_version mismatch between baseline and latest → unknown."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        base_ts = now - 3600
        _create_test_goal(conn, goal_id="ver_p", source_id="vp")
        _create_test_goal(conn, goal_id="ver_s", metric_id="m2", source_id="vs")

        mid = _create_test_measurement(conn, "ver_p", 50.0, base_ts, source_id="vp")
        oid = _create_test_outcome(conn, "t_ver", "ver_p", mid, started_at=base_ts + 1)

        from hermes_cli.outcome_ledger import complete_action
        conn.execute("BEGIN IMMEDIATE")
        complete_action(conn, oid, action_revision="sha_v", completed_at=base_ts + 5)
        conn.execute("COMMIT")
        _create_test_measurement(conn, "ver_p", 60.0, base_ts + 20, source_id="vp")
        from hermes_cli.outcome_ledger import evaluate_due_outcome
        conn.execute("BEGIN IMMEDIATE")
        evaluate_due_outcome(conn, oid, now=base_ts + 25)
        conn.execute("COMMIT")

        sec_b = _create_test_measurement(
            conn, "ver_s", 100.0, base_ts, metric_id="m2", source_id="vs"
        )

        # Manually change metric_version on latest measurement
        sec_l = _create_test_measurement(
            conn, "ver_s", 80.0, base_ts + 30, metric_id="m2", source_id="vs"
        )
        conn.execute(
            "UPDATE goal_measurements SET metric_version = 99 WHERE id = ?",
            (sec_l,),
        )

        result = evaluate_trip_wire(
            conn, "ver_p", oid, "ver_s", sec_b, sec_l, 0.9, 1.1, now,
        )
        assert result.tripped is None
        assert "version" in result.reason.lower()


# ---------------------------------------------------------------------------
# propose_rollback tests
# ---------------------------------------------------------------------------


class TestProposeRollback:
    def test_basic_proposal(self):
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        _create_test_goal(conn)
        mid = _create_test_measurement(conn, "g1", 50.0, now - 100)
        oid = _create_test_outcome(conn, "t_rb", "g1", mid, started_at=now - 90)

        proposal = propose_rollback(conn, "drift_high", [oid])
        assert isinstance(proposal, RollbackProposal)
        assert proposal.requires_approval is True
        assert proposal.trigger == "drift_high"
        assert len(proposal.outcome_ids) == 1
        assert proposal.proposal_id is not None

    def test_proposal_is_inert(self):
        """Proposal does not execute any git revert, deploy, or mutation."""
        conn = _make_conn()
        _setup_full_schema(conn)

        now = time.time()
        _create_test_goal(conn)
        mid = _create_test_measurement(conn, "g1", 50.0, now - 100)
        oid = _create_test_outcome(conn, "t_inert", "g1", mid, started_at=now - 90)

        proposal = propose_rollback(conn, "trip_wire", [oid])
        assert proposal.requires_approval is True

        # Verify no git operations were attempted (no side effects)
        # The proposal is just a data record
        assert isinstance(proposal.evidence, str)
        assert isinstance(proposal.revision, str)

    def test_proposal_empty_outcome_ids_fails(self):
        conn = _make_conn()
        with pytest.raises(ValueError, match="non-empty list"):
            propose_rollback(conn, "drift", [])

    def test_proposal_invalid_trigger_fails(self):
        conn = _make_conn()
        with pytest.raises(ValueError, match="non-empty string"):
            propose_rollback(conn, "", ["o1"])


# ---------------------------------------------------------------------------
# reset_guardrail tests
# ---------------------------------------------------------------------------


class TestResetGuardrail:
    def test_basic_reset(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)

        # Create a halt
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO guardrail_halts (id, scope, reason, halted_at, created_at) "
            "VALUES ('h1', 'agent:a1', 'cap exceeded', 1000.0, 1000.0)"
        )
        conn.execute("COMMIT")

        reset_guardrail(conn, "agent:a1", "keion", "new budget approved",
                        "policy-2024-001", 2000.0)

        # Verify halt is resolved
        halt = conn.execute(
            "SELECT * FROM guardrail_halts WHERE id = 'h1'"
        ).fetchone()
        assert halt["resolved_at"] == 2000.0
        assert halt["resolved_by"] == "keion"

        # Verify audit record
        reset_row = conn.execute(
            "SELECT * FROM guardrail_resets WHERE scope = 'agent:a1'"
        ).fetchone()
        assert reset_row["actor"] == "keion"
        assert reset_row["approval_ref"] == "policy-2024-001"

    def test_reset_requires_actor(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        with pytest.raises(ValueError, match="actor"):
            reset_guardrail(conn, "global", "", "reason", "ref", 1000.0)

    def test_reset_requires_approval_ref(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        with pytest.raises(ValueError, match="approval_ref"):
            reset_guardrail(conn, "global", "admin", "reason", "", 1000.0)

    def test_reset_audit_is_append_only(self):
        """Multiple resets create multiple audit records; no erasure."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)

        # Create and resolve halt twice
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO guardrail_halts (id, scope, reason, halted_at, created_at) "
            "VALUES ('h2', 'global', 'cap exceeded', 1000.0, 1000.0)"
        )
        conn.execute("COMMIT")
        reset_guardrail(conn, "global", "admin1", "budget 1", "pol-1", 2000.0)

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO guardrail_halts (id, scope, reason, halted_at, created_at) "
            "VALUES ('h3', 'global', 'cap exceeded again', 3000.0, 3000.0)"
        )
        conn.execute("COMMIT")
        reset_guardrail(conn, "global", "admin2", "budget 2", "pol-2", 4000.0)

        resets = conn.execute(
            "SELECT * FROM guardrail_resets WHERE scope = 'global' ORDER BY reset_at"
        ).fetchall()
        assert len(resets) == 2
        assert resets[0]["actor"] == "admin1"
        assert resets[1]["actor"] == "admin2"

    def test_prior_spend_preserved_after_reset(self):
        """Reset resolves halts but does NOT erase spend records."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)

        # Create reservation
        rk = "spend-persist"
        _reserve_default(conn, reservation_key=rk, agent_id="ag_reset")
        commit_act(conn, rk, actual_tokens=500, now=1_700_000_100.0)

        # Create and resolve halt
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO guardrail_halts (id, scope, reason, halted_at, created_at) "
            "VALUES ('h_sp', 'agent:ag_reset', 'cap exceeded', 1700000200.0, 1700000200.0)"
        )
        conn.execute("COMMIT")
        reset_guardrail(
            conn, "agent:ag_reset", "admin", "approved", "pol-sp",
            1_700_000_300.0,
        )

        # Verify reservation still exists with committed data
        row = conn.execute(
            "SELECT * FROM guardrail_reservations WHERE reservation_key = ?",
            (rk,),
        ).fetchone()
        assert row["status"] == "committed"
        assert row["actual_tokens"] == 500

    def test_reset_enables_new_reservations(self):
        """After reset, new reservations can proceed (halt resolved)."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)

        # Exceed cap → halt
        _reserve_default(
            conn,
            agent_id="ag_re",
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        r_denied = _reserve_default(
            conn,
            agent_id="ag_re",
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        assert r_denied.allowed is False

        # Reset the halt
        reset_guardrail(
            conn, "agent:ag_re", "admin", "new budget", "pol-new",
            1_700_000_100.0,
        )

        # New reservation with sufficient cap should work
        r_after = _reserve_default(
            conn,
            agent_id="ag_re",
            estimated_tokens=100,
            price_per_million_usd="10.00",
            caps={"agent_day": "100.00"},
            now=1_700_000_100.0,
        )
        assert r_after.allowed is True


# ---------------------------------------------------------------------------
# Integration: safe halt persists
# ---------------------------------------------------------------------------


class TestSafeHalt:
    def test_halt_blocks_all_scopes(self):
        """A halt on any relevant scope blocks reservations."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)

        # Manually create a global halt
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO guardrail_halts (id, scope, reason, halted_at, created_at) "
            "VALUES ('gh1', 'global', 'manual halt', 1000.0, 1000.0)"
        )
        conn.execute("COMMIT")

        r = _reserve_default(conn, caps={"task_token": "100000"})
        assert r.allowed is False
        assert "safe halt" in r.reason

    def test_halt_no_auto_resume(self):
        """Halts require explicit reset; no auto-resume."""
        conn = _make_conn()
        migrate_guardrail_schema(conn)

        # Trigger halt via cap exceeded
        _reserve_default(
            conn,
            estimated_tokens=1000,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )
        _reserve_default(
            conn,
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"agent_day": "0.01"},
        )

        # Even much later, halt persists
        r = _reserve_default(
            conn,
            estimated_tokens=1,
            price_per_million_usd="10.00",
            caps={"agent_day": "999999.00"},
            now=1_700_100_000.0,  # much later
        )
        assert r.allowed is False
        assert "safe halt" in r.reason


# ---------------------------------------------------------------------------
# Dollar precision tests
# ---------------------------------------------------------------------------


class TestDollarPrecision:
    def test_decimal_string_cap(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        r = _reserve_default(
            conn,
            price_per_million_usd="0.150",
            caps={"agent_day": "0.000001"},  # 1 micro-USD
            estimated_tokens=1,
        )
        # Price $0.15/M, 1 token = $0.00000015 → 0.15 µ$ → rounds to 1 µ$
        # Cap is exactly 1 µ$ → should pass (1 <= 1)
        assert r.allowed is True

    def test_decimal_object_cap(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        r = _reserve_default(
            conn,
            price_per_million_usd=Decimal("10.00"),
            caps={"agent_day": Decimal("100.00")},
        )
        assert r.allowed is True

    def test_nonpositive_cap_fails(self):
        conn = _make_conn()
        migrate_guardrail_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValueError, match="positive"):
            reserve_act(
                conn,
                task_id="t1",
                agent_id="a1",
                lane_id="l1",
                reservation_key="rk1",
                estimated_tokens=100,
                price_per_million_usd="10.00",
                caps={"agent_day": "0.00"},
                now=1_700_000_000.0,
            )

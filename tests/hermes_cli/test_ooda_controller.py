"""Tests for hermes_cli.ooda_controller — P1 opt-in finite OODA consumer.

Run via ``scripts/run_tests.sh tests/hermes_cli/test_ooda_controller.py``.

Covers: disposable board bus→gap→Act→completion feedback, no spawn on no-op,
restart replay, concurrency/burst, capacity release, unauthorized rights gate,
old DB migration, and subscriber confinement.
"""
from __future__ import annotations

import json
import os
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

from hermes_cli.event_bus import (
    Event,
    ack,
    migrate_event_bus,
    poll,
    publish,
    subscribe,
)
from hermes_cli.kanban_event_publish import (
    clear_bus_cache,
    emit_run_completed,
)
from hermes_cli.ooda_controller import (
    OodaResult,
    _OBSERVE_EVENTS,
    _count_active_ooda_acts,
    _decide,
    _orient_gap,
    _orient_run_completed,
    _stable_idempotency_key,
    run_once,
)


# ---------------------------------------------------------------------------
# Disposable board helper
# ---------------------------------------------------------------------------

# Minimal kanban tasks schema (subset matching what kanban_db.py creates).
_TASKS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS tasks (
    id                TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    body              TEXT,
    status            TEXT NOT NULL DEFAULT 'ready',
    assignee          TEXT,
    result            TEXT,
    completed_at      INTEGER,
    created_at        INTEGER NOT NULL,
    claim_lock        TEXT,
    claim_expires     INTEGER,
    worker_pid        INTEGER,
    worker_started_at INTEGER,
    current_run_id    INTEGER,
    priority          INTEGER NOT NULL DEFAULT 0,
    idempotency_key   TEXT,
    tenant            TEXT,
    block_kind        TEXT,
    block_recurrences INTEGER DEFAULT 0,
    resume_after      INTEGER,
    continuation_count INTEGER DEFAULT 0,
    workspace_kind    TEXT,
    workspace_path    TEXT,
    branch_name       TEXT,
    max_runtime_seconds INTEGER,
    model_override    TEXT,
    provider_override TEXT,
    reasoning_effort  TEXT,
    goal_mode         INTEGER DEFAULT 0,
    goal_max_turns    INTEGER,
    session_id        TEXT,
    project_id        TEXT,
    current_step_key  TEXT,
    completion_contract TEXT,
    initial_status    TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_idempotency
    ON tasks(idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL,
    run_id     INTEGER
);

CREATE TABLE IF NOT EXISTS task_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    profile      TEXT,
    step_key     TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    outcome      TEXT,
    summary      TEXT,
    error        TEXT,
    metadata     TEXT,
    started_at   INTEGER NOT NULL,
    ended_at     INTEGER,
    claim_lock   TEXT,
    claim_expires INTEGER,
    worker_pid   INTEGER,
    worker_started_at INTEGER
);

CREATE TABLE IF NOT EXISTS task_parents (
    parent_id TEXT NOT NULL,
    child_id  TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_attachments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    filename   TEXT NOT NULL,
    content    BLOB,
    content_type TEXT,
    size       INTEGER,
    created_by TEXT,
    created_at INTEGER NOT NULL
);
"""


def _disposable_board() -> tuple[sqlite3.Connection, str]:
    """Create a disposable board DB with event bus + goal metrics + tasks schema.

    Returns (conn, db_path). Caller must close & cleanup.
    The DB is created under /tmp to avoid the delegation fence
    (HERMES_DELEGATED_CHILD_CONTEXT fences paths under ~/.hermes).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir="/tmp")
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")

    # Create tasks schema first (event_bus references need this for publisher).
    for stmt in _TASKS_SCHEMA.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)

    # Migrate event bus.
    migrate_event_bus(conn)

    # Migrate goal metrics.
    from hermes_cli.goal_metrics import migrate_goal_schema
    migrate_goal_schema(conn)

    clear_bus_cache()

    return conn, db_path


def _create_active_goal(conn, *, goal_id="test-goal-1", metric_id="board.blocked_card_count",
                        target_value=5.0, direction="lower_is_better", tolerance=0.0,
                        max_age_seconds=86400):
    """Create and activate a goal on the disposable board."""
    from hermes_cli.goal_metrics import create_goal, activate_goal
    from hermes_cli.kanban_db_connect import write_txn

    with write_txn(conn):
        create_goal(
            conn,
            goal_id=goal_id,
            metric_id=metric_id,
            source_id="test",
            direction=direction,
            target_value=target_value,
            tolerance=tolerance,
            max_age_seconds=max_age_seconds,
        )
    with write_txn(conn):
        activate_goal(
            conn, goal_id,
            approval_evidence_ref="test-approval",
            approval_timestamp=time.time(),
        )
    return goal_id


def _record_breached_measurement(conn, goal_id, *, value=10.0, metric_id="board.blocked_card_count",
                                  source_timestamp=None, idem_key=None):
    """Record a measurement that breaches the goal threshold."""
    from hermes_cli.goal_metrics import record_measurement
    from hermes_cli.kanban_db_connect import write_txn

    now = time.time()
    if source_timestamp is None:
        source_timestamp = now
    if idem_key is None:
        idem_key = f"test-meas-{now}"

    with write_txn(conn):
        result = record_measurement(
            conn,
            goal_id=goal_id,
            metric_id=metric_id,
            value=value,
            source_id="test",
            source_timestamp=source_timestamp,
            observed_at=now,
            source_ref="test-source-ref",
            idempotency_key=idem_key,
        )
    return result


def _inject_task(conn, task_id, *, status="ready", idempotency_key=None, title="test"):
    """Directly inject a task row (bypasses create_task's full logic)."""
    conn.execute(
        "INSERT OR IGNORE INTO tasks (id, title, status, created_at, idempotency_key) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, title, status, int(time.time()), idempotency_key),
    )


# ---------------------------------------------------------------------------
# Test: Subscription and event type confinement
# ---------------------------------------------------------------------------

class TestSubscription:
    """Verify subscriber setup and event type filtering."""

    def test_observe_events_excludes_card_created(self):
        """card.created must NOT be in the subscription (recursion guard)."""
        assert "card.created" not in _OBSERVE_EVENTS

    def test_observe_events_includes_gap_and_run_completed(self):
        assert "goal.gap_detected" in _OBSERVE_EVENTS
        assert "metric.threshold_crossed" not in _OBSERVE_EVENTS
        assert "run.completed" in _OBSERVE_EVENTS

    def test_subscriber_registration(self):
        conn, db_path = _disposable_board()
        try:
            subscribe(conn, "test-sub", event_types=_OBSERVE_EVENTS, start="latest")
            # Re-registration is idempotent.
            cursor = subscribe(conn, "test-sub", event_types=_OBSERVE_EVENTS, start="latest")
            assert isinstance(cursor, int)
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: No-op on empty bus
# ---------------------------------------------------------------------------

class TestNoOpOnEmpty:
    """run_once with no events returns a no-op result."""

    def test_noop_on_empty_bus(self):
        conn, db_path = _disposable_board()
        try:
            result = run_once(
                conn,
                subscriber_id="test-noop",
                max_events=10,
                permitted_board=db_path,
            )
            assert result.events_processed == 0
            assert result.events_skipped == 0
            assert result.acts_created == 0
            assert "no_events" in result.noop_reasons
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: Full bus → gap → Act pipeline
# ---------------------------------------------------------------------------

class TestBusToGapToAct:
    """End-to-end: gap event on bus → orient → decide → create investigation card."""

    def test_gap_event_creates_investigation_card(self):
        conn, db_path = _disposable_board()
        try:
            # Setup: active goal with breached measurement.
            goal_id = _create_active_goal(conn)
            meas_result = _record_breached_measurement(conn, goal_id)

            # The breached measurement should have emitted goal.gap_detected
            # and metric.threshold_crossed on the bus.
            assert not meas_result.is_replay
            assert {e["event_type"] for e in meas_result.events} == {
                "goal.gap_detected", "metric.threshold_crossed"
            }

            # Register subscriber BEFORE events (start='beginning' to catch them).
            subscribe(conn, "test-gap-act", event_types=_OBSERVE_EVENTS, start="beginning")

            # Verify events are visible.
            events = poll(conn, "test-gap-act", limit=100)
            gap_events = [e for e in events if e.event_type == "goal.gap_detected"]
            assert len(gap_events) == 1, f"Expected one actionable edge, got: {[e.event_type for e in events]}"

            # Run OODA.
            result = run_once(
                conn,
                subscriber_id="test-gap-act",
                max_events=10,
                permitted_board=db_path,
                assignee="default",
            )

            assert result.acts_created == 1
            assert result.events_processed + result.events_skipped >= 1

            # Verify investigation card was created with correct idempotency key.
            row = conn.execute(
                "SELECT * FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'"
            ).fetchone()
            assert row is not None
            assert row["assignee"] == "default"
            assert conn.execute("SELECT COUNT(*) FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'").fetchone()[0] == 1
            assert "Investigate" in row["title"]
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: Completion feedback (run.completed)
# ---------------------------------------------------------------------------

class TestCompletionFeedback:
    """run.completed events are consumed as capacity release signals."""

    def test_run_completed_is_processed_not_acted(self):
        conn, db_path = _disposable_board()
        try:
            # Register subscriber from beginning.
            subscribe(conn, "test-feedback", event_types=_OBSERVE_EVENTS, start="beginning")

            # Publish a run.completed event.
            conn.execute("BEGIN IMMEDIATE")
            publish(
                conn,
                event_type="run.completed",
                source="kanban",
                payload={"task_id": "t_abc123", "run_id": 42, "outcome": "completed"},
            )
            conn.execute("COMMIT")

            # Run OODA.
            result = run_once(
                conn,
                subscriber_id="test-feedback",
                max_events=10,
                permitted_board=db_path,
            )

            # run.completed should be processed (ACKed) but NOT acted on.
            assert result.events_processed == 1
            assert result.acts_created == 0

            # Subscriber cursor should have advanced past the event.
            remaining = poll(conn, "test-feedback", limit=10)
            assert len(remaining) == 0
        finally:
            conn.close()
            os.unlink(db_path)

    def test_emit_run_completed_publishes_allowlisted_payload(self):
        """emit_run_completed in the publisher produces a valid bus event."""
        conn, db_path = _disposable_board()
        try:
            subscribe(conn, "test-emit-rc", event_types=frozenset({"run.completed"}), start="beginning")

            conn.execute("BEGIN IMMEDIATE")
            emit_run_completed(conn, "t_test123", run_id=99, outcome="completed")
            conn.execute("COMMIT")

            events = poll(conn, "test-emit-rc", limit=10)
            assert len(events) == 1
            evt = events[0]
            assert evt.event_type == "run.completed"
            assert evt.payload["task_id"] == "t_test123"
            assert evt.payload["run_id"] == 99
            assert evt.payload["outcome"] == "completed"
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: Restart replay and idempotent dedup
# ---------------------------------------------------------------------------

class TestRestartReplay:
    """Reprocessing the same event after restart produces the same card (dedup)."""

    def test_restart_replay_deduplicates(self):
        conn, db_path = _disposable_board()
        try:
            goal_id = _create_active_goal(conn)
            _record_breached_measurement(conn, goal_id, idem_key="replay-meas-1")

            # First run with subscriber starting from beginning.
            subscribe(conn, "test-replay", event_types=_OBSERVE_EVENTS, start="beginning")
            result1 = run_once(
                conn,
                subscriber_id="test-replay",
                max_events=10,
                max_active_acts=2,  # The global in-flight budget applies on replay.
                permitted_board=db_path,
            )
            first_cards = conn.execute(
                "SELECT id, idempotency_key FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'"
            ).fetchall()
            assert len(first_cards) >= 1, f"Expected at least 1 card, got {len(first_cards)}"

            # Remember the count.
            first_count = len(first_cards)

            # Simulate restart: new subscriber seeing the same events from beginning.
            # The idempotency key check in _act_create_investigation should dedup.
            subscribe(conn, "test-replay-2", event_types=_OBSERVE_EVENTS, start="beginning")

            result2 = run_once(
                conn,
                subscriber_id="test-replay-2",
                max_events=10,
                max_active_acts=2,
                permitted_board=db_path,
            )

            all_cards = conn.execute(
                "SELECT id FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'"
            ).fetchall()
            # Should still be exactly the same count (dedup'd).
            assert len(all_cards) == first_count, \
                f"Expected {first_count} cards after replay, got {len(all_cards)}"
            # Second run should have deduplicated.
            assert result2.acts_deduplicated >= 1 or result2.acts_created == 0
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: Concurrency / burst / capacity limit
# ---------------------------------------------------------------------------

class TestConcurrencyBurst:
    """Max active acts enforcement under burst conditions."""

    def test_capacity_limit_enforced(self):
        conn, db_path = _disposable_board()
        try:
            # Pre-populate 2 active OODA investigation cards.
            _inject_task(
                conn, "t_ooda_1", status="running",
                idempotency_key="ooda:investigate:g1:m1:1",
            )
            _inject_task(
                conn, "t_ooda_2", status="ready",
                idempotency_key="ooda:investigate:g2:m2:2",
            )

            # Create a gap event.
            goal_id = _create_active_goal(conn)
            _record_breached_measurement(conn, goal_id, idem_key="burst-meas")

            subscribe(conn, "test-burst", event_types=_OBSERVE_EVENTS, start="beginning")

            result = run_once(
                conn,
                subscriber_id="test-burst",
                max_events=10,
                max_active_acts=2,
                permitted_board=db_path,
            )

            # Should NOT create a third card (at capacity).
            assert result.acts_created == 0
            any_capacity = any("at_capacity" in r for r in result.noop_reasons)
            assert any_capacity, f"Expected at_capacity noop, got: {result.noop_reasons}"
        finally:
            conn.close()
            os.unlink(db_path)

    def test_capacity_release_allows_new_act(self):
        """After done/archived cards free capacity, a new act is permitted."""
        conn, db_path = _disposable_board()
        try:
            # One active + one done = only 1 active.
            _inject_task(
                conn, "t_ooda_active", status="running",
                idempotency_key="ooda:investigate:ga:ma:1",
            )
            _inject_task(
                conn, "t_ooda_done", status="done",
                idempotency_key="ooda:investigate:gd:md:2",
            )

            goal_id = _create_active_goal(conn)
            _record_breached_measurement(conn, goal_id, idem_key="cap-rel-meas")

            subscribe(conn, "test-cap-rel", event_types=_OBSERVE_EVENTS, start="beginning")

            result = run_once(
                conn,
                subscriber_id="test-cap-rel",
                max_events=10,
                max_active_acts=2,
                permitted_board=db_path,
            )

            # Should create (1 active + 1 new = 2 <= max).
            assert result.acts_created >= 1
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: No-op on healthy / stale / inactive goal
# ---------------------------------------------------------------------------

class TestOrientNoOp:
    """Orient returns None when gap is not verified active/fresh."""

    def test_orient_noop_on_healthy_measurement(self):
        conn, db_path = _disposable_board()
        try:
            goal_id = _create_active_goal(conn, target_value=5.0, direction="lower_is_better")
            # Record a healthy measurement (value=3 < target=5 for lower_is_better).
            _record_breached_measurement(conn, goal_id, value=3.0, idem_key="healthy-meas")

            # Create a fake gap event.
            fake_event = Event(
                id=999, event_type="goal.gap_detected", source="goal_metrics",
                payload={"goal_id": goal_id, "metric_id": "board.blocked_card_count"},
                created_at=time.time(), idempotency_key=None,
            )

            result = _orient_gap(conn, fake_event, now=time.time())
            # edge_state should be healthy, so orient returns None.
            assert result is None
        finally:
            conn.close()
            os.unlink(db_path)

    def test_orient_noop_on_paused_goal(self):
        conn, db_path = _disposable_board()
        try:
            from hermes_cli.goal_metrics import pause_goal
            from hermes_cli.kanban_db_connect import write_txn

            goal_id = _create_active_goal(conn)
            with write_txn(conn):
                pause_goal(conn, goal_id)

            fake_event = Event(
                id=998, event_type="goal.gap_detected", source="goal_metrics",
                payload={"goal_id": goal_id, "metric_id": "board.blocked_card_count"},
                created_at=time.time(), idempotency_key=None,
            )
            result = _orient_gap(conn, fake_event, now=time.time())
            assert result is None
        finally:
            conn.close()
            os.unlink(db_path)

    def test_orient_noop_on_stale_measurement(self):
        conn, db_path = _disposable_board()
        try:
            goal_id = _create_active_goal(conn, max_age_seconds=60)
            # Record a valid breached measurement first.
            _record_breached_measurement(conn, goal_id, value=10.0, idem_key="stale-meas")

            # Now artificially age the measurement by updating source_timestamp.
            old_ts = time.time() - 120  # 2 minutes ago, exceeds max_age 60s
            conn.execute(
                "UPDATE goal_measurements SET source_timestamp = ? WHERE goal_id = ?",
                (old_ts, goal_id),
            )

            fake_event = Event(
                id=997, event_type="goal.gap_detected", source="goal_metrics",
                payload={"goal_id": goal_id, "metric_id": "board.blocked_card_count"},
                created_at=time.time(), idempotency_key=None,
            )
            # now is well past max_age.
            result = _orient_gap(conn, fake_event, now=time.time())
            assert result is None
        finally:
            conn.close()
            os.unlink(db_path)

    def test_orient_noop_on_nonexistent_goal(self):
        conn, db_path = _disposable_board()
        try:
            fake_event = Event(
                id=996, event_type="goal.gap_detected", source="goal_metrics",
                payload={"goal_id": "nonexistent-goal", "metric_id": "board.blocked_card_count"},
                created_at=time.time(), idempotency_key=None,
            )
            result = _orient_gap(conn, fake_event, now=time.time())
            assert result is None
        finally:
            conn.close()
            os.unlink(db_path)

    def test_orient_noop_on_missing_metric_id(self):
        conn, db_path = _disposable_board()
        try:
            fake_event = Event(
                id=995, event_type="goal.gap_detected", source="goal_metrics",
                payload={"goal_id": "some-goal"},
                created_at=time.time(), idempotency_key=None,
            )
            result = _orient_gap(conn, fake_event, now=time.time())
            assert result is None
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: Decision rights gate
# ---------------------------------------------------------------------------

class TestDecisionRightsGate:
    """Decide function enforces the rights classifier."""

    def test_decide_at_capacity(self):
        should_act, reason = _decide(
            {"event_id": 1, "goal_id": "g1"},
            max_active_acts=2,
            active_count=2,
        )
        assert not should_act
        assert reason == "at_capacity"

    def test_decide_no_evidence_ref(self):
        should_act, reason = _decide(
            {"goal_id": "g1"},
            max_active_acts=2,
            active_count=0,
        )
        assert not should_act
        assert reason == "no_evidence_ref"

    def test_decide_authorized_with_evidence(self):
        should_act, reason = _decide(
            {"event_id": 42, "goal_id": "g1"},
            max_active_acts=2,
            active_count=0,
        )
        assert should_act
        assert reason == "authorized"

    def test_decide_rights_module_rejection(self):
        """When the rights module rejects, decide returns False."""
        mock_decision = mock.MagicMock()
        mock_decision.outcome = "keion_decision"

        with mock.patch.dict("sys.modules", {
            "hermes_cli.ooda_decision_rights": mock.MagicMock(
                ActionType=mock.MagicMock(create_investigation_card="create_investigation_card"),
                ActionContext=mock.MagicMock(return_value=mock.MagicMock()),
                Decision=mock.MagicMock(),
                classify=mock.MagicMock(return_value=mock_decision),
            )
        }):
            # Clear the module cache so the mock is picked up.
            import importlib
            should_act, reason = _decide(
                {"event_id": 42, "goal_id": "g1"},
                max_active_acts=2,
                active_count=0,
            )
            assert not should_act
            assert "rights_gate:keion_decision" == reason


# ---------------------------------------------------------------------------
# Test: Stable idempotency key
# ---------------------------------------------------------------------------

class TestIdempotencyKey:
    """Idempotency key is deterministic for same inputs."""

    def test_stable_key(self):
        k1 = _stable_idempotency_key("goal-1", "metric-a", 42)
        k2 = _stable_idempotency_key("goal-1", "metric-a", 42)
        assert k1 == k2
        assert "ooda:investigate:" in k1

    def test_different_inputs_different_keys(self):
        k1 = _stable_idempotency_key("goal-1", "metric-a", 42)
        k2 = _stable_idempotency_key("goal-2", "metric-a", 42)
        assert k1 != k2


# ---------------------------------------------------------------------------
# Test: Active OODA act counting
# ---------------------------------------------------------------------------

class TestActiveActCounting:
    """_count_active_ooda_acts correctly counts non-terminal OODA cards."""

    def test_counts_running_and_ready(self):
        conn, db_path = _disposable_board()
        try:
            _inject_task(conn, "t1", status="running", idempotency_key="ooda:investigate:a:b:1")
            _inject_task(conn, "t2", status="ready", idempotency_key="ooda:investigate:a:b:2")
            _inject_task(conn, "t3", status="done", idempotency_key="ooda:investigate:a:b:3")
            _inject_task(conn, "t4", status="archived", idempotency_key="ooda:investigate:a:b:4")
            _inject_task(conn, "t5", status="blocked", idempotency_key="ooda:investigate:a:b:5")
            _inject_task(conn, "t6", status="running", idempotency_key="unrelated:key")

            count = _count_active_ooda_acts(conn)
            # running(t1) + ready(t2) + blocked(t5) = 3; done/archived excluded, unrelated excluded
            assert count == 3
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: run_once input validation
# ---------------------------------------------------------------------------

class TestRunOnceValidation:
    """Argument validation for run_once."""

    def test_rejects_in_transaction(self):
        conn, db_path = _disposable_board()
        try:
            conn.execute("BEGIN IMMEDIATE")
            with pytest.raises(RuntimeError, match="autocommit"):
                run_once(conn, subscriber_id="x", permitted_board=db_path)
            conn.execute("ROLLBACK")
        finally:
            conn.close()
            os.unlink(db_path)

    def test_rejects_invalid_max_events(self):
        conn, db_path = _disposable_board()
        try:
            with pytest.raises(ValueError, match="max_events"):
                run_once(conn, subscriber_id="x", max_events=0, permitted_board=db_path)
            with pytest.raises(ValueError, match="max_events"):
                run_once(conn, subscriber_id="x", max_events=1001, permitted_board=db_path)
        finally:
            conn.close()
            os.unlink(db_path)

    def test_rejects_empty_subscriber_id(self):
        conn, db_path = _disposable_board()
        try:
            with pytest.raises(ValueError, match="subscriber_id"):
                run_once(conn, subscriber_id="", permitted_board=db_path)
        finally:
            conn.close()
            os.unlink(db_path)

    def test_rejects_empty_permitted_board(self):
        conn, db_path = _disposable_board()
        try:
            with pytest.raises(ValueError, match="permitted_board"):
                run_once(conn, subscriber_id="x", permitted_board="")
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: run.completed orient
# ---------------------------------------------------------------------------

class TestOrientRunCompleted:
    """_orient_run_completed extracts capacity release context."""

    def test_extracts_capacity_release(self):
        evt = Event(
            id=100, event_type="run.completed", source="kanban",
            payload={"task_id": "t_abc", "run_id": 5, "outcome": "completed"},
            created_at=time.time(), idempotency_key=None,
        )
        ctx = _orient_run_completed(mock.MagicMock(), evt)
        assert ctx is not None
        assert ctx["type"] == "capacity_release"
        assert ctx["task_id"] == "t_abc"
        assert ctx["outcome"] == "completed"

    def test_returns_none_on_missing_task_id(self):
        evt = Event(
            id=101, event_type="run.completed", source="kanban",
            payload={"run_id": 5},
            created_at=time.time(), idempotency_key=None,
        )
        ctx = _orient_run_completed(mock.MagicMock(), evt)
        assert ctx is None


# ---------------------------------------------------------------------------
# Test: Old DB migration compatibility
# ---------------------------------------------------------------------------

class TestOldDBMigration:
    """Verify the controller works on a DB that had no event bus initially."""

    def test_works_on_freshly_migrated_db(self):
        """A DB created without event bus, then migrated, should work."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir="/tmp")
        db_path = tmp.name
        tmp.close()

        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")

        # Create tasks table first (simulating old DB).
        for stmt in _TASKS_SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)

        # Now migrate event bus (simulating upgrade).
        migrate_event_bus(conn)
        from hermes_cli.goal_metrics import migrate_goal_schema
        migrate_goal_schema(conn)
        clear_bus_cache()

        try:
            result = run_once(
                conn,
                subscriber_id="test-old-db",
                max_events=10,
                permitted_board=db_path,
            )
            assert result.events_processed == 0
            assert "no_events" in result.noop_reasons
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: Self-emitted card.created does not cause recursion
# ---------------------------------------------------------------------------

class TestNoRecursion:
    """Controller must not subscribe to card.created (which it emits)."""

    def test_card_created_not_in_subscription(self):
        assert "card.created" not in _OBSERVE_EVENTS

    def test_full_pipeline_does_not_see_own_card_created(self):
        """After Act creates a card, the controller's subscriber should NOT
        see the resulting card.created event (because it's not subscribed)."""
        conn, db_path = _disposable_board()
        try:
            goal_id = _create_active_goal(conn)
            _record_breached_measurement(conn, goal_id, idem_key="recursion-meas")

            subscribe(conn, "test-recursion", event_types=_OBSERVE_EVENTS, start="beginning")

            # First run creates a card.
            result1 = run_once(
                conn,
                subscriber_id="test-recursion",
                max_events=10,
                permitted_board=db_path,
            )
            assert result1.acts_created >= 1

            # Second run should have no events (card.created is filtered out).
            result2 = run_once(
                conn,
                subscriber_id="test-recursion",
                max_events=10,
                permitted_board=db_path,
            )
            assert result2.acts_created == 0
            assert "no_events" in result2.noop_reasons
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: Burst processing with multiple gap events
# ---------------------------------------------------------------------------

class TestBurstProcessing:
    """Multiple gap events in one run_once invocation."""

    def test_processes_multiple_events_up_to_capacity(self):
        conn, db_path = _disposable_board()
        try:
            # Create two goals with different metrics.
            goal1 = _create_active_goal(
                conn, goal_id="burst-g1",
                metric_id="board.blocked_card_count",
                target_value=5.0, direction="lower_is_better",
            )
            goal2 = _create_active_goal(
                conn, goal_id="burst-g2",
                metric_id="board.non_archived_card_count",
                target_value=3.0, direction="lower_is_better",
            )

            _record_breached_measurement(conn, goal1, value=10.0, idem_key="burst-m1")
            _record_breached_measurement(
                conn, goal2, value=8.0,
                metric_id="board.non_archived_card_count", idem_key="burst-m2",
            )

            subscribe(conn, "test-burst-multi", event_types=_OBSERVE_EVENTS, start="beginning")

            result = run_once(
                conn,
                subscriber_id="test-burst-multi",
                max_events=20,
                max_active_acts=2,
                permitted_board=db_path,
            )

            # Should create up to max_active_acts investigation cards.
            cards = conn.execute(
                "SELECT id FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'"
            ).fetchall()
            assert len(cards) <= 2
        finally:
            conn.close()
            os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test: Publisher emit_run_completed integration
# ---------------------------------------------------------------------------

class TestPublisherEmitRunCompleted:
    """Integration tests for emit_run_completed in kanban_event_publish."""

    def test_emit_with_all_fields(self):
        conn, db_path = _disposable_board()
        try:
            subscribe(conn, "test-pub", event_types=frozenset({"run.completed"}), start="beginning")

            conn.execute("BEGIN IMMEDIATE")
            emit_run_completed(conn, "t_xyz", run_id=100, outcome="failed")
            conn.execute("COMMIT")

            events = poll(conn, "test-pub", limit=10)
            assert len(events) == 1
            p = events[0].payload
            assert p["task_id"] == "t_xyz"
            assert p["run_id"] == 100
            assert p["outcome"] == "failed"
        finally:
            conn.close()
            os.unlink(db_path)

    def test_emit_with_minimal_fields(self):
        conn, db_path = _disposable_board()
        try:
            subscribe(conn, "test-pub-min", event_types=frozenset({"run.completed"}), start="beginning")

            conn.execute("BEGIN IMMEDIATE")
            emit_run_completed(conn, "t_minimal")
            conn.execute("COMMIT")

            events = poll(conn, "test-pub-min", limit=10)
            assert len(events) == 1
            p = events[0].payload
            assert p["task_id"] == "t_minimal"
            assert "run_id" not in p  # None is omitted
            assert "outcome" not in p
        finally:
            conn.close()
            os.unlink(db_path)

    def test_does_not_crash_on_legacy_db(self):
        """emit_run_completed on a DB without event bus tables is a no-op."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir="/tmp")
        db_path = tmp.name
        tmp.close()

        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        clear_bus_cache()

        try:
            # No bus migration — legacy DB.
            conn.execute("BEGIN IMMEDIATE")
            # Should not raise.
            emit_run_completed(conn, "t_legacy", run_id=1, outcome="completed")
            conn.execute("COMMIT")
        finally:
            conn.close()
            os.unlink(db_path)


class TestLogicalEdgeAndAdmission:
    def test_recovery_rebreach_and_restart_at_full_capacity(self):
        conn, db_path = _disposable_board()
        try:
            goal_id = _create_active_goal(conn)
            subscribe(conn, "edges", event_types=_OBSERVE_EVENTS, start="beginning")
            first = _record_breached_measurement(conn, goal_id, idem_key="edge-1")
            assert run_once(conn, subscriber_id="edges", max_active_acts=1, permitted_board=db_path).acts_created == 1
            subscribe(conn, "restart", event_types=_OBSERVE_EVENTS, start="beginning")
            replay = run_once(conn, subscriber_id="restart", max_active_acts=1, permitted_board=db_path)
            assert (replay.acts_created, replay.acts_deduplicated) == (0, 1)
            healthy = _record_breached_measurement(conn, goal_id, value=3, idem_key="healthy")
            assert {e["event_type"] for e in healthy.events} == {"goal.achieved"}
            second = _record_breached_measurement(conn, goal_id, idem_key="edge-2")
            assert {e["event_type"] for e in second.events} == {
                "goal.gap_detected", "metric.threshold_crossed"
            }
            assert first.events[0]["event_id"] != second.events[0]["event_id"]
            assert run_once(conn, subscriber_id="edges", max_active_acts=1, permitted_board=db_path).acts_created == 0
            conn.execute("UPDATE tasks SET status = 'done' WHERE idempotency_key LIKE 'ooda:investigate:%'")
            # The original subscriber retains the unserved edge until release.
            assert run_once(conn, subscriber_id="edges", max_active_acts=1,
                            permitted_board=db_path).acts_created == 1
            assert conn.execute("SELECT COUNT(*) FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'").fetchone()[0] == 2
        finally:
            conn.close()
            os.unlink(db_path)

    def test_file_board_mismatch_and_invalid_assignee_fail_before_subscribe(self):
        conn, db_path = _disposable_board()
        other, other_path = _disposable_board()
        try:
            goal_id = _create_active_goal(conn)
            _record_breached_measurement(conn, goal_id, idem_key="admission")
            with pytest.raises(ValueError, match="permitted_board"):
                run_once(conn, subscriber_id="wrong-board", permitted_board=other_path)
            with pytest.raises(ValueError, match="assignee"):
                run_once(conn, subscriber_id="wrong-profile", permitted_board=db_path,
                         assignee="profile-that-does-not-exist")
            assert conn.execute("SELECT COUNT(*) FROM tasks WHERE idempotency_key LIKE 'ooda:investigate:%'").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM hermes_event_cursors").fetchone()[0] == 0
        finally:
            conn.close()
            other.close()
            os.unlink(db_path)
            os.unlink(other_path)

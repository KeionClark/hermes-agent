"""Prove canonical connect is schema-preserving for goal tables (t_539aaab5).

After this fix, ``connect()`` and ``init_db()`` MUST NOT create goal-related
tables (``goals``, ``goal_measurements``, ``goal_schema_version``) on any
board.  Goal tables are created only via explicit ``ensure_goal_schema()``.

Run via ``scripts/run_tests.sh tests/hermes_cli/test_canonical_goal_migration_gate.py``.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Goal tables that must NOT appear after a default connect/init_db.
_GOAL_TABLES = {"goals", "goal_measurements", "goal_schema_version"}

# P0a tables that MUST appear after a default connect/init_db.
_CORE_TABLES = {"tasks", "task_events", "task_comments", "task_runs", "hermes_events"}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    """Return all user table names on a connection."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _fresh_board_via_connect() -> tuple[sqlite3.Connection, Path]:
    """Create a fresh board via the standard connect() path."""
    from hermes_cli.kanban_db_connect import _INITIALIZED_PATHS, connect

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)
    # Clear any cached init state for this path.
    _INITIALIZED_PATHS.discard(str(path.resolve()))
    conn = connect(path)
    return conn, path


class TestCanonicalConnectNoGoalTables:
    """connect() and init_db() must not create goal tables."""

    def test_fresh_board_connect_has_no_goal_tables(self):
        """A fresh board opened via connect() has core tables but NO goal tables."""
        conn, path = _fresh_board_via_connect()
        try:
            tables = _table_names(conn)
            # Core tables must exist.
            for t in _CORE_TABLES:
                assert t in tables, f"missing core table: {t}"
            # Goal tables must NOT exist.
            for t in _GOAL_TABLES:
                assert t not in tables, (
                    f"goal table '{t}' was created by connect() — "
                    f"canonical board is not schema-preserving"
                )
        finally:
            conn.close()

    def test_init_db_has_no_goal_tables(self):
        """init_db() re-runs migrations but must not create goal tables."""
        from hermes_cli.kanban_db_connect import _INITIALIZED_PATHS, init_db, connect

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        path = Path(tmp.name)
        _INITIALIZED_PATHS.discard(str(path.resolve()))

        # init_db forces a re-migration.
        init_db(path)

        conn = connect(path)
        try:
            tables = _table_names(conn)
            for t in _GOAL_TABLES:
                assert t not in tables, (
                    f"goal table '{t}' was created by init_db() — "
                    f"canonical board is not schema-preserving"
                )
        finally:
            conn.close()

    def test_reconnect_after_init_preserves_no_goal_tables(self):
        """A second connect() to the same board still has no goal tables."""
        from hermes_cli.kanban_db_connect import _INITIALIZED_PATHS, connect

        conn, path = _fresh_board_via_connect()
        conn.close()

        # Clear cache to force re-init on next connect (simulates restart).
        _INITIALIZED_PATHS.discard(str(path.resolve()))

        conn2 = connect(path)
        try:
            tables = _table_names(conn2)
            for t in _GOAL_TABLES:
                assert t not in tables, (
                    f"goal table '{t}' appeared after reconnect — "
                    f"restart would migrate canonical board"
                )
        finally:
            conn2.close()


class TestLegacyBoardNoGoalMigration:
    """A canonical-shaped board (has hermes_events but no goals) must stay
    goal-free after connect/restart."""

    def test_canonical_shaped_board_stays_goal_free(self):
        """Board with hermes_events but no goals stays goal-free on connect."""
        from hermes_cli.kanban_db_connect import _INITIALIZED_PATHS, connect

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        path = Path(tmp.name)
        _INITIALIZED_PATHS.discard(str(path.resolve()))

        # First connect creates the canonical shape.
        conn = connect(path)
        assert "hermes_events" in _table_names(conn)
        assert "goals" not in _table_names(conn)
        conn.close()

        # Clear cache and reconnect (simulates gateway restart on new code).
        _INITIALIZED_PATHS.discard(str(path.resolve()))
        conn2 = connect(path)
        try:
            tables = _table_names(conn2)
            assert "hermes_events" in tables
            assert "goals" not in tables, (
                "goal tables appeared on reconnect to canonical-shaped board"
            )
        finally:
            conn2.close()


class TestExplicitEnsureGoalSchema:
    """ensure_goal_schema() creates goal tables on opt-in."""

    def test_ensure_goal_schema_creates_tables(self):
        """Explicit ensure_goal_schema() on a fresh board creates goal tables."""
        from hermes_cli.kanban_db_connect import ensure_goal_schema

        conn, path = _fresh_board_via_connect()
        try:
            # Before: no goal tables.
            tables_before = _table_names(conn)
            for t in _GOAL_TABLES:
                assert t not in tables_before

            # After explicit opt-in: goal tables exist.
            ensure_goal_schema(conn)
            tables_after = _table_names(conn)
            for t in _GOAL_TABLES:
                assert t in tables_after, f"ensure_goal_schema did not create '{t}'"
        finally:
            conn.close()

    def test_ensure_goal_schema_idempotent(self):
        """Calling ensure_goal_schema twice is safe."""
        from hermes_cli.kanban_db_connect import ensure_goal_schema

        conn, path = _fresh_board_via_connect()
        try:
            ensure_goal_schema(conn)
            ensure_goal_schema(conn)  # Must not raise.
            tables = _table_names(conn)
            for t in _GOAL_TABLES:
                assert t in tables
        finally:
            conn.close()


class TestGoalCrudFailsClosed:
    """Goal CRUD on a board without goal tables raises
    GoalSchemaNotInitializedError, not a raw OperationalError."""

    def test_get_goal_fails_closed(self):
        from hermes_cli.goal_metrics import GoalSchemaNotInitializedError, get_goal

        conn, _ = _fresh_board_via_connect()
        try:
            with pytest.raises(GoalSchemaNotInitializedError):
                get_goal(conn, "nonexistent")
        finally:
            conn.close()

    def test_list_goals_fails_closed(self):
        from hermes_cli.goal_metrics import GoalSchemaNotInitializedError, list_goals

        conn, _ = _fresh_board_via_connect()
        try:
            with pytest.raises(GoalSchemaNotInitializedError):
                list_goals(conn)
        finally:
            conn.close()

    def test_create_goal_fails_closed(self):
        from hermes_cli.goal_metrics import (
            GoalSchemaNotInitializedError,
            create_goal,
        )

        conn, _ = _fresh_board_via_connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            with pytest.raises(GoalSchemaNotInitializedError):
                create_goal(
                    conn,
                    metric_id="test.metric",
                    source_id="test",
                )
            conn.execute("ROLLBACK")
        finally:
            conn.close()

    def test_activate_goal_fails_closed(self):
        from hermes_cli.goal_metrics import (
            GoalSchemaNotInitializedError,
            activate_goal,
        )

        conn, _ = _fresh_board_via_connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            with pytest.raises(GoalSchemaNotInitializedError):
                activate_goal(conn, "nonexistent", "evidence", time.time())
            conn.execute("ROLLBACK")
        finally:
            conn.close()

    def test_pause_goal_fails_closed(self):
        from hermes_cli.goal_metrics import (
            GoalSchemaNotInitializedError,
            pause_goal,
        )

        conn, _ = _fresh_board_via_connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            with pytest.raises(GoalSchemaNotInitializedError):
                pause_goal(conn, "nonexistent")
            conn.execute("ROLLBACK")
        finally:
            conn.close()

    def test_record_measurement_fails_closed(self):
        from hermes_cli.goal_metrics import (
            GoalSchemaNotInitializedError,
            record_measurement,
        )

        conn, _ = _fresh_board_via_connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            with pytest.raises(GoalSchemaNotInitializedError):
                record_measurement(
                    conn,
                    goal_id="nonexistent",
                    metric_id="test.metric",
                    value=1.0,
                    source_id="test",
                    source_timestamp=time.time(),
                    observed_at=time.time(),
                    source_ref="test",
                    idempotency_key="test-key",
                )
            conn.execute("ROLLBACK")
        finally:
            conn.close()


class TestP0aStatusEventsUnaffected:
    """P0a functionality (task status, events, event bus) is unaffected by the
    goal migration gate."""

    def test_task_create_and_status_work(self):
        """Creating a task, changing status, and recording events all work
        on a board with no goal tables."""
        conn, _ = _fresh_board_via_connect()
        try:
            tables = _table_names(conn)
            assert "tasks" in tables
            assert "task_events" in tables
            assert "hermes_events" in tables

            # Create a task.
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("t_test1", "Test task", "ready", int(time.time())),
            )

            # Update status.
            conn.execute(
                "UPDATE tasks SET status = 'running' WHERE id = 't_test1'"
            )

            # Record an event.
            conn.execute(
                "INSERT INTO task_events (task_id, kind, created_at) "
                "VALUES (?, ?, ?)",
                ("t_test1", "claimed", int(time.time())),
            )

            # Verify the task is readable.
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = 't_test1'"
            ).fetchone()
            assert row is not None
            assert row[0] == "running"

            # Verify the event is readable.
            evt = conn.execute(
                "SELECT kind FROM task_events WHERE task_id = 't_test1'"
            ).fetchone()
            assert evt is not None
            assert evt[0] == "claimed"
        finally:
            conn.close()

    def test_event_bus_works_without_goals(self):
        """The P0a event bus publishes and polls events on a board with no
        goal tables."""
        from hermes_cli.event_bus import publish, subscribe, poll

        conn, _ = _fresh_board_via_connect()
        try:
            # Subscribe.
            conn.execute("BEGIN IMMEDIATE")
            subscribe(conn, "test-sub")
            conn.execute("COMMIT")

            # Publish.
            conn.execute("BEGIN IMMEDIATE")
            publish(
                conn,
                event_type="test.event",
                source="test",
                payload={"status": "ok"},
                idempotency_key="test-idem-1",
            )
            conn.execute("COMMIT")

            # Poll.
            events = poll(conn, "test-sub", limit=10)
            assert len(events) == 1
            assert events[0].event_type == "test.event"
        finally:
            conn.close()


class TestGoalCrudWorksAfterExplicitMigration:
    """After ensure_goal_schema(), goal CRUD works normally."""

    def test_create_and_read_goal_after_migration(self):
        from hermes_cli.kanban_db_connect import ensure_goal_schema
        from hermes_cli.goal_metrics import create_goal, get_goal, list_goals

        conn, _ = _fresh_board_via_connect()
        try:
            ensure_goal_schema(conn)

            conn.execute("BEGIN IMMEDIATE")
            goal = create_goal(
                conn,
                metric_id="board.blocked_card_count",
                source_id="kanban_board_adapter",
                direction="lower_is_better",
                target_value=5.0,
                max_age_seconds=3600,
                title="Test goal",
            )
            conn.execute("COMMIT")

            # Read back.
            fetched = get_goal(conn, goal.id)
            assert fetched is not None
            assert fetched.metric_id == "board.blocked_card_count"
            assert fetched.state == "draft"

            goals = list_goals(conn)
            assert len(goals) == 1
        finally:
            conn.close()

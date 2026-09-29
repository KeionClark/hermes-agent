"""Tests for the P0a Kanban event publisher bridge.

Exercises every lifecycle path: create/block/complete/status transitions,
rollbacks, no-op/repeated changes, nested create, dispatcher exit/breaker,
dashboard drag-drop, swarm, and privacy markers.

Uses real disposable boards (temp DB) — no mocks for board state.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Fixture: isolated disposable board with event bus tables
# ---------------------------------------------------------------------------

@pytest.fixture()
def board(tmp_path, monkeypatch):
    """Disposable board: temp DB with all schema + event bus tables.

    Yields ``(conn, db_path)`` and sets ``HERMES_KANBAN_DB`` so
    ``kanban_db_connect.connect`` reaches it.  Import ``kanban_db`` inside
    the test body (after monkeypatch has run) to avoid module-level side effects.
    """
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    # Hermes's board home must also be writable:
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    # Avoid profile lookup noise:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))

    from hermes_cli.kanban_db_connect import connect
    conn = connect(db_path)
    # Bus tables should exist (migrated by connect):
    row = conn.execute(
        "SELECT version FROM hermes_event_schema_version WHERE id = 1"
    ).fetchone()
    assert row is not None, "event bus migration did not run"

    # Clear the bus probe cache so tests start fresh:
    from hermes_cli.kanban_event_publish import clear_bus_cache
    clear_bus_cache()

    yield conn, db_path

    conn.close()
    clear_bus_cache()


def _bus_events(conn: sqlite3.Connection, event_type: Optional[str] = None) -> list[dict]:
    """All hermes_events rows, optionally filtered by event_type."""
    if event_type:
        rows = conn.execute(
            "SELECT id, event_type, source, payload, created_at, idempotency_key "
            "FROM hermes_events WHERE event_type = ? ORDER BY id",
            (event_type,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, event_type, source, payload, created_at, idempotency_key "
            "FROM hermes_events ORDER BY id"
        ).fetchall()
    result = []
    for r in rows:
        result.append({
            "id": r[0],
            "event_type": r[1],
            "source": r[2],
            "payload": json.loads(r[3]),
            "created_at": r[4],
            "idempotency_key": r[5],
        })
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCardCreated:
    """card.created is emitted once on successful task INSERT."""

    def test_create_task_emits_card_created(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task
        task_id = create_task(conn, title="Test task", assignee="test-profile")
        events = _bus_events(conn, "card.created")
        assert len(events) == 1
        assert events[0]["payload"]["task_id"] == task_id
        assert events[0]["source"] == "kanban"

    def test_create_with_initial_blocked_emits_card_created_and_blocked(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task
        task_id = create_task(
            conn, title="Blocked from start", assignee="test",
            initial_status="blocked",
        )
        created = _bus_events(conn, "card.created")
        assert len(created) == 1
        assert created[0]["payload"]["task_id"] == task_id

        blocked = _bus_events(conn, "card.blocked")
        assert len(blocked) == 1
        assert blocked[0]["payload"]["task_id"] == task_id

    def test_create_task_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task
        task_id = create_task(conn, title="SC test", assignee="p")
        sc = _bus_events(conn, "card.status_changed")
        assert len(sc) == 1
        p = sc[0]["payload"]
        assert p["task_id"] == task_id
        assert p["to"] == "ready"
        assert p["from"] is None


class TestCardCompleted:
    """card.completed on successful done transition."""

    def test_complete_emits_card_completed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, complete_task
        task_id = create_task(conn, title="Complete me", assignee="x")
        assert complete_task(conn, task_id, result="done", fire_lifecycle_hook=False)
        completed = _bus_events(conn, "card.completed")
        assert len(completed) == 1
        assert completed[0]["payload"]["task_id"] == task_id

    def test_complete_emits_status_changed_to_done(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, complete_task
        task_id = create_task(conn, title="Done test", assignee="x")
        complete_task(conn, task_id, result="done", fire_lifecycle_hook=False)
        sc = [e for e in _bus_events(conn, "card.status_changed") if e["payload"]["to"] == "done"]
        assert len(sc) == 1
        assert sc[0]["payload"]["from"] == "ready"


class TestCardBlocked:
    """card.blocked only on actual entry to blocked status."""

    def test_block_task_emits_card_blocked(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, block_task
        task_id = create_task(conn, title="Block me", assignee="x")
        claimed = claim_task(conn, task_id)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert block_task(conn, task_id, reason="test", kind="needs_input", expected_run_id=run_id)
        blocked = _bus_events(conn, "card.blocked")
        # Only blocked from block_task, not from create (which was "ready").
        assert len(blocked) == 1
        assert blocked[0]["payload"]["task_id"] == task_id
        assert blocked[0]["payload"]["block_kind"] == "needs_input"

    def test_dependency_block_does_not_emit_card_blocked(self, board):
        """A dependency block goes to todo, not blocked. No card.blocked event."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, block_task
        parent_id = create_task(conn, title="Parent", assignee="x")
        child_id = create_task(conn, title="Child", assignee="x", parents=[parent_id])
        # Complete parent so child becomes ready
        from hermes_cli.kanban_db import complete_task
        complete_task(conn, parent_id, result="ok", fire_lifecycle_hook=False)
        # Now claim and dependency-block the child
        claimed = claim_task(conn, child_id)
        assert claimed is not None
        # Re-create parent dependency: link a new undone parent
        parent2 = create_task(conn, title="Parent2", assignee="x")
        from hermes_cli.kanban_db import link_tasks
        link_tasks(conn, parent2, child_id, expected_child_run_id=claimed.current_run_id)
        # Block as dependency
        block_task(conn, child_id, reason="waiting", kind="dependency", expected_run_id=claimed.current_run_id)
        blocked_events = _bus_events(conn, "card.blocked")
        # Should have NO card.blocked from the dependency block (goes to todo)
        child_blocked = [e for e in blocked_events if e["payload"]["task_id"] == child_id]
        assert len(child_blocked) == 0

    def test_in_place_reclassification_no_card_blocked(self, board):
        """Classifying an untyped blocked card in place should not emit card.blocked."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, block_task
        task_id = create_task(conn, title="Reclass", assignee="x")
        claimed = claim_task(conn, task_id)
        # Block it first (it enters blocked status)
        block_task(conn, task_id, reason="first", kind="needs_input", expected_run_id=claimed.current_run_id)
        initial_blocked = len(_bus_events(conn, "card.blocked"))

        # Now manually strip block_kind to simulate untyped breaker block
        from hermes_cli.kanban_db_connect import write_txn
        with write_txn(conn):
            conn.execute("UPDATE tasks SET block_kind = NULL, current_run_id = NULL WHERE id = ?", (task_id,))

        # Reclassify in place
        block_task(conn, task_id, kind="transient")
        after_blocked = len(_bus_events(conn, "card.blocked"))
        # No new card.blocked
        assert after_blocked == initial_blocked


class TestStatusChanged:
    """card.status_changed on every actual status transition."""

    def test_full_lifecycle_emits_transitions(self, board):
        """create -> claim -> complete produces the expected sequence."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, complete_task
        task_id = create_task(conn, title="Lifecycle", assignee="x")
        claim_task(conn, task_id)
        complete_task(conn, task_id, result="ok", fire_lifecycle_hook=False)
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id]
        transitions = [(p["from"], p["to"]) for p in sc]
        assert (None, "ready") in transitions       # create
        assert ("ready", "running") in transitions   # claim
        assert ("running", "done") in transitions    # complete

    def test_no_op_status_change_not_emitted(self, board):
        """A mutation that doesn't actually change status should not emit."""
        conn, _ = board
        from hermes_cli.kanban_event_publish import emit_status_changed
        from hermes_cli.kanban_db_connect import write_txn
        with write_txn(conn):
            # Same from and to => should be skipped
            emit_status_changed(conn, "t_fake", from_status="ready", to_status="ready")
        assert len(_bus_events(conn, "card.status_changed")) == 0

    def test_promote_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, promote_task
        task_id = create_task(conn, title="Promote", assignee="x", initial_status="blocked")
        ok, _ = promote_task(conn, task_id, actor="test")
        assert ok
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["to"] == "ready"]
        assert len(sc) == 1
        assert sc[0]["from"] == "blocked"

    def test_unblock_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, block_task, unblock_task
        task_id = create_task(conn, title="Unblock", assignee="x")
        claimed = claim_task(conn, task_id)
        block_task(conn, task_id, reason="r", kind="needs_input", expected_run_id=claimed.current_run_id)
        unblock_task(conn, task_id)
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["from"] == "blocked"]
        assert len(sc) == 1

    def test_archive_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, archive_task
        task_id = create_task(conn, title="Archive", assignee="x")
        archive_task(conn, task_id)
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["to"] == "archived"]
        assert len(sc) == 1

    def test_review_lifecycle(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, request_review, complete_task
        task_id = create_task(conn, title="Review cycle", assignee="builder")
        claimed = claim_task(conn, task_id)
        request_review(conn, task_id, summary="review plz",
                        expected_run_id=claimed.current_run_id,
                        reviewer="reviewer")
        complete_task(conn, task_id, result="approved", fire_lifecycle_hook=False, force=True)
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id]
        transitions = [(p["from"], p["to"]) for p in sc]
        assert ("running", "review") in transitions
        assert ("review", "done") in transitions

    def test_reopen_review_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, request_review, reopen_review_task
        task_id = create_task(conn, title="Reopen", assignee="impl")
        claimed = claim_task(conn, task_id)
        request_review(conn, task_id, summary="look",
                        expected_run_id=claimed.current_run_id,
                        reviewer="rev")
        reopen_review_task(conn, task_id)
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["from"] == "review"]
        assert len(sc) >= 1

    def test_specify_triage_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, specify_triage_task
        task_id = create_task(conn, title="Triage", assignee="x", triage=True)
        specify_triage_task(conn, task_id, author="specifier")
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["from"] == "triage"]
        assert len(sc) == 1
        assert sc[0]["to"] == "todo"


class TestRollback:
    """Publish is atomic with the board mutation — rollback discards it."""

    def test_failed_txn_rolls_back_event(self, board):
        conn, _ = board
        from hermes_cli.kanban_db_connect import write_txn
        from hermes_cli.kanban_event_publish import emit_card_created
        try:
            with write_txn(conn):
                emit_card_created(conn, "t_rollback_test")
                raise RuntimeError("force rollback")
        except RuntimeError:
            pass
        assert len(_bus_events(conn, "card.created")) == 0


class TestPrivacy:
    """No PII/PHI in raw hermes_events."""

    def test_no_reason_title_assignee_in_events(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, block_task
        task_id = create_task(
            conn, title="Patient John Doe", assignee="nurse",
            body="Patient notes...",
        )
        claimed = claim_task(conn, task_id)
        block_task(conn, task_id, reason="patient needs consult",
                   kind="needs_input", expected_run_id=claimed.current_run_id)
        for event in _bus_events(conn):
            payload = event["payload"]
            payload_str = json.dumps(payload)
            assert "Patient" not in payload_str
            assert "John" not in payload_str
            assert "Doe" not in payload_str
            assert "nurse" not in payload_str
            assert "patient needs consult" not in payload_str
            # Only allowed keys
            for key in payload:
                assert key in {"task_id", "from", "to", "run_id", "block_kind"}, \
                    f"unexpected key {key!r} in event payload"


class TestOldTaskEventsPreserved:
    """Existing task_events rows are never altered by the publisher bridge."""

    def test_task_events_preserved(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, complete_task
        task_id = create_task(conn, title="Events test", assignee="x")
        claim_task(conn, task_id)
        complete_task(conn, task_id, result="ok", fire_lifecycle_hook=False)
        task_events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
        kinds = [r[0] for r in task_events]
        assert "created" in kinds
        assert "claimed" in kinds
        assert "completed" in kinds


class TestLegacyDB:
    """Legacy DB without event bus tables still works."""

    def test_missing_bus_tables_does_not_crash(self, tmp_path, monkeypatch):
        db_path = tmp_path / "legacy.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))

        from hermes_cli.kanban_event_publish import clear_bus_cache
        clear_bus_cache()

        from hermes_cli.kanban_db_connect import connect
        conn = connect(db_path)
        # Drop the bus tables to simulate a legacy DB
        conn.execute("DROP TABLE IF EXISTS hermes_events")
        conn.execute("DROP TABLE IF EXISTS hermes_event_cursors")
        conn.execute("DROP TABLE IF EXISTS hermes_event_schema_version")
        conn.commit()

        clear_bus_cache()  # Force re-probe

        from hermes_cli.kanban_db import create_task, complete_task
        task_id = create_task(conn, title="Legacy", assignee="x")
        assert complete_task(conn, task_id, result="ok", fire_lifecycle_hook=False)
        conn.close()
        clear_bus_cache()


class TestDecomposerChild:
    """Decomposed children emit card.created."""

    def test_decompose_emits_child_created(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task
        from hermes_cli.kanban_db_graph import decompose_triage_task
        root = create_task(conn, title="Root", assignee="orch", triage=True)
        child_ids = decompose_triage_task(
            conn, root, root_assignee="orch",
            children=[
                {"title": "Child A", "assignee": "worker-a"},
                {"title": "Child B", "assignee": "worker-b"},
            ],
            author="test",
        )
        assert child_ids is not None
        created = _bus_events(conn, "card.created")
        created_ids = {e["payload"]["task_id"] for e in created}
        for cid in child_ids:
            assert cid in created_ids


class TestSwarmInlineCompletion:
    """Swarm inline root activation emits card.completed."""

    def test_swarm_root_completion(self, board):
        conn, _ = board
        from hermes_cli.kanban_swarm import create_swarm, SwarmWorkerSpec
        swarm = create_swarm(
            conn,
            goal="Test swarm",
            workers=[SwarmWorkerSpec(profile="w", title="Worker", body="do")],
            verifier_assignee="v",
            synthesizer_assignee="s",
            created_by="test",
        )
        completed = _bus_events(conn, "card.completed")
        root_completed = [e for e in completed if e["payload"]["task_id"] == swarm.root_id]
        assert len(root_completed) == 1


class TestDispatcherPaths:
    """Dispatcher crash/timeout/breaker paths emit proper events."""

    def test_record_task_failure_breaker_emits_blocked(self, board):
        """When the breaker trips, card.blocked is emitted."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task
        from hermes_cli.kanban_db_dispatch import _record_task_failure
        task_id = create_task(conn, title="Breaker test", assignee="x")
        claimed = claim_task(conn, task_id)
        assert claimed is not None
        # Force trip the breaker
        tripped = _record_task_failure(
            conn, task_id, "test error",
            outcome="crashed",
            force_trip=True,
            release_claim=True,
            end_run=True,
        )
        assert tripped
        blocked = _bus_events(conn, "card.blocked")
        task_blocked = [e for e in blocked if e["payload"]["task_id"] == task_id]
        assert len(task_blocked) == 1

    def test_record_task_failure_below_limit_emits_status_change(self, board):
        """Below the breaker limit with release_claim, running -> retry_status."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task
        from hermes_cli.kanban_db_dispatch import _record_task_failure
        task_id = create_task(conn, title="Below limit", assignee="x")
        claimed = claim_task(conn, task_id)
        assert claimed is not None
        tripped = _record_task_failure(
            conn, task_id, "spawn fail",
            outcome="spawn_failed",
            failure_limit=10,
            release_claim=True,
            end_run=True,
        )
        assert not tripped
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["from"] == "running"]
        # The last status_changed for this task from running should be to ready (retry)
        assert any(p["to"] == "ready" for p in sc)


class TestScheduleTaskEmit:
    """schedule_task must emit card.status_changed with actual prior status."""

    def test_schedule_from_ready_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, schedule_task
        task_id = create_task(conn, title="Schedule ready", assignee="x")
        ok = schedule_task(conn, task_id, reason="cron wait")
        assert ok
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["to"] == "scheduled"]
        assert len(sc) == 1
        assert sc[0]["from"] == "ready"

    def test_schedule_from_blocked_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, block_task, schedule_task
        task_id = create_task(conn, title="Schedule blocked", assignee="x")
        claimed = claim_task(conn, task_id)
        block_task(conn, task_id, reason="wait", kind="needs_input",
                   expected_run_id=claimed.current_run_id)
        ok = schedule_task(conn, task_id, reason="timed park")
        assert ok
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["to"] == "scheduled"]
        assert len(sc) == 1
        assert sc[0]["from"] == "blocked"

    def test_schedule_from_running_emits_status_changed(self, board):
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, schedule_task
        task_id = create_task(conn, title="Schedule running", assignee="x")
        claimed = claim_task(conn, task_id)
        ok = schedule_task(conn, task_id, reason="park", expected_run_id=claimed.current_run_id)
        assert ok
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["to"] == "scheduled"]
        assert len(sc) == 1
        assert sc[0]["from"] == "running"

    def test_schedule_carries_run_id(self, board):
        """The emitted event should carry the run_id from _end_or_synthesize_run."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, schedule_task
        task_id = create_task(conn, title="Schedule run_id", assignee="x")
        claimed = claim_task(conn, task_id)
        schedule_task(conn, task_id, reason="park", expected_run_id=claimed.current_run_id)
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["to"] == "scheduled"]
        assert len(sc) == 1
        assert "run_id" in sc[0]

    def test_schedule_noop_no_event(self, board):
        """Scheduling an already-scheduled task should not emit."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, schedule_task
        task_id = create_task(conn, title="Schedule noop", assignee="x")
        schedule_task(conn, task_id, reason="first")
        # Count events so far
        before = len(_bus_events(conn, "card.status_changed"))
        ok = schedule_task(conn, task_id, reason="second")
        assert not ok  # already scheduled, CAS fails
        after = len(_bus_events(conn, "card.status_changed"))
        assert after == before


class TestUnblockScheduledFromStatus:
    """unblock_task must report the actual prior status, not hardcoded 'blocked'."""

    def test_unblock_from_scheduled_reports_scheduled(self, board):
        """When unblocking a 'scheduled' task, from_status must be 'scheduled'."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, schedule_task, unblock_task
        task_id = create_task(conn, title="Unblock scheduled", assignee="x")
        schedule_task(conn, task_id, reason="cron wait")
        ok = unblock_task(conn, task_id)
        assert ok
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["from"] == "scheduled"]
        assert len(sc) == 1, (
            f"Expected one status_changed from 'scheduled', got {sc}"
        )

    def test_unblock_from_blocked_still_reports_blocked(self, board):
        """When unblocking a 'blocked' task, from_status must still be 'blocked'."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, claim_task, block_task, unblock_task
        task_id = create_task(conn, title="Unblock blocked", assignee="x")
        claimed = claim_task(conn, task_id)
        block_task(conn, task_id, reason="wait", kind="needs_input",
                   expected_run_id=claimed.current_run_id)
        ok = unblock_task(conn, task_id)
        assert ok
        sc = [e["payload"] for e in _bus_events(conn, "card.status_changed")
              if e["payload"]["task_id"] == task_id and e["payload"]["from"] == "blocked"
              and e["payload"]["to"] != "blocked"]  # skip the block transition itself
        assert len(sc) >= 1, (
            f"Expected at least one status_changed from 'blocked' (unblock), got {sc}"
        )

    def test_unblock_scheduled_no_false_blocked_event(self, board):
        """The bug: unblocking a 'scheduled' task must NOT emit from='blocked'."""
        conn, _ = board
        from hermes_cli.kanban_db import create_task, schedule_task, unblock_task
        task_id = create_task(conn, title="No false blocked", assignee="x")
        schedule_task(conn, task_id, reason="park")
        # Clear events up to this point so we only check the unblock emit
        events_before_unblock = len(_bus_events(conn, "card.status_changed"))
        ok = unblock_task(conn, task_id)
        assert ok
        all_sc = _bus_events(conn, "card.status_changed")
        unblock_events = all_sc[events_before_unblock:]
        for ev in unblock_events:
            if ev["payload"]["task_id"] == task_id:
                assert ev["payload"]["from"] != "blocked", (
                    f"Unblocking a 'scheduled' task must not report from='blocked': {ev['payload']}"
                )

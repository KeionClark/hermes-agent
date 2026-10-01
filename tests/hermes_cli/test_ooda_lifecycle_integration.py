"""Real-board integration: rights decision and terminal Kanban bus feedback."""
import json
from pathlib import Path

import pytest

from hermes_cli import event_bus, kanban_db as kb, kanban_db_connect as kbc
from hermes_cli.ooda_controller import _decide, _orient_run_completed


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def test_real_rights_authorize_bounded_internal_investigation():
    allowed, reason = _decide({"event_id": 17}, max_active_acts=2, active_count=0)
    assert (allowed, reason) == (True, "authorized")
    assert _decide({}, max_active_acts=2, active_count=0)[0] is False
    assert _decide({"event_id": 17}, max_active_acts=2, active_count=2)[0] is False


def test_terminal_run_bus_events_are_allowlisted_and_correlated(board):
    conn = board
    event_bus.subscribe(conn, "integration", event_types={"run.completed"}, start="latest")
    tid = kb.create_task(conn, title="synthetic investigation", assignee="test-worker",
                         idempotency_key="ooda:investigate:test:metric:17")
    assert kb.claim_task(conn, tid)
    run_id = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()[0]
    assert kb.complete_task(conn, tid, summary="PRIVATE patient content must not enter bus",
                            metadata={"private": "not for bus"})
    events = event_bus.poll(conn, "integration")
    assert len(events) == 1
    evt = events[0]
    assert evt.payload == {"task_id": tid, "run_id": run_id, "outcome": "completed"}
    assert _orient_run_completed(conn, evt)["task_id"] == tid
    assert _orient_run_completed(conn, evt.__class__(id=evt.id, event_type=evt.event_type,
            source=evt.source, payload={**evt.payload, "run_id": run_id + 100},
            created_at=evt.created_at, idempotency_key=evt.idempotency_key)) is None
    assert kb._end_run(conn, tid, outcome="completed") is None
    assert len(event_bus.poll(conn, "integration")) == 1


def test_never_claimed_completion_gets_one_synthetic_run_event(board):
    conn = board
    event_bus.subscribe(conn, "synthetic", event_types={"run.completed"}, start="latest")
    tid = kb.create_task(conn, title="unclaimed")
    assert kb.complete_task(conn, tid, summary="private")
    events = event_bus.poll(conn, "synthetic")
    assert len(events) == 1
    assert events[0].payload["outcome"] == "completed"
    assert events[0].payload["run_id"] is not None


def test_failed_run_event_has_outcome_not_error(board):
    conn = board
    event_bus.subscribe(conn, "failure", event_types={"run.completed"}, start="latest")
    tid = kb.create_task(conn, title="synthetic failure", assignee="worker")
    assert kb.claim_task(conn, tid)
    with kbc.write_txn(conn):
        run_id = kb._end_run(conn, tid, outcome="failed", error="private exception")
    events = event_bus.poll(conn, "failure")
    assert len(events) == 1
    assert events[0].payload == {"task_id": tid, "run_id": run_id, "outcome": "failed"}

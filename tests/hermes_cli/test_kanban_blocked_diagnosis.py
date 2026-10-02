"""Event-stream diagnosis contract on a disposable board."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_blocked_diagnosis import consume, migrate, IDLE_SECONDS


@pytest.fixture
def board(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def block(conn, kind="needs_input", parents=()):
    task = kb.create_task(conn, title="synthetic gate", assignee="owner", parents=list(parents))
    assert kb.claim_task(conn, task)
    run = kb.get_task(conn, task).current_run_id
    assert kb.block_task(conn, task, kind=kind, reason="synthetic", expected_run_id=run)
    return task


def test_opt_in_backfill_replay_and_original_receipt(board):
    task = block(board)
    with pytest.raises((RuntimeError, __import__('sqlite3').OperationalError)):
        consume(board)
    before = len(kb.list_tasks(board))
    migrate(board)
    first = consume(board)
    assert [d["task_id"] for d in first["diagnoses"]] == [task]
    assert first["diagnoses"][0]["historical_sweep_at"] is None
    assert first["diagnoses"][0]["dispatch_mode"] == "bounded_poll_not_immediate"
    assert not consume(board)["diagnoses"]
    assert len(kb.list_comments(board, task)) == 1
    assert len(kb.list_tasks(board)) == before
    assert kb.get_task(board, task).status == "blocked"


@pytest.mark.parametrize("kind", ["needs_input", "capability", "transient"])
def test_sticky_kinds_do_not_auto_resolve(board, kind):
    task = block(board, kind)
    migrate(board)
    diagnosis = consume(board)["diagnoses"][0]
    assert diagnosis["kind"] == kind
    assert kb.get_task(board, task).status == "blocked"
    assert not consume(board)["outcomes"]


def test_dependency_parent_states_and_archived_predecessor(board):
    parent = kb.create_task(board, title="prerequisite", assignee="owner")
    child = kb.create_task(board, title="synthetic dependent", assignee="owner")
    assert kb.claim_task(board, child)
    board.execute("INSERT INTO task_links VALUES (?, ?)", (parent, child))
    board.commit()
    assert kb.block_task(board, child, kind="dependency", reason="waiting", expected_run_id=kb.get_task(board, child).current_run_id)
    assert kb.get_task(board, child).status == "todo"
    assert kb.get_task(board, parent).status == "ready"
    assert kb.claim_task(board, parent)
    assert kb.get_task(board, parent).status == "running"
    assert kb.block_task(board, parent, kind="capability", reason="blocked")
    assert kb.get_task(board, parent).status == "blocked"
    migrate(board)
    consume(board)
    assert kb.archive_task(board, parent)
    assert kb.get_task(board, child).status == "todo"
    assert kb.unsatisfied_parents(board, child) == [(parent, "archived")]
    assert not kb.claim_task(board, child)
    assert not kb.recompute_ready(board)
    # Diagnosis agrees with the kernel: an unfinished archived parent still gates.
    from hermes_cli.kanban_blocked_diagnosis import _parents
    assert _parents(board, child)[0]["status"] == "archived"


def test_unfulfilled_archived_parent_gates_creation_and_link_but_completed_archive_does_not(board):
    abandoned = kb.create_task(board, title="unfinished predecessor")
    assert kb.archive_task(board, abandoned)
    waiting = kb.create_task(board, title="dependent", parents=[abandoned])
    assert kb.get_task(board, waiting).status == "todo"
    assert kb.unsatisfied_parents(board, waiting) == [(abandoned, "archived")]
    ready = kb.create_task(board, title="ready")
    assert kb.link_tasks(board, abandoned, ready)
    assert kb.get_task(board, ready).status == "todo"

    fulfilled = kb.create_task(board, title="finished predecessor")
    assert kb.complete_task(board, fulfilled, result="done")
    assert kb.archive_task(board, fulfilled)
    child = kb.create_task(board, title="fulfilled child", parents=[fulfilled])
    assert kb.get_task(board, child).status == "ready"
    assert kb.claim_task(board, child)


def test_cycle_and_retry_cap_are_diagnostic_only(board):
    task = block(board, "transient")
    parent = kb.create_task(board, title="cycle peer", assignee="owner", parents=[task], initial_status="blocked")
    board.execute("INSERT INTO task_links VALUES (?, ?)", (parent, task))
    board.execute("UPDATE tasks SET block_recurrences=3 WHERE id=?", (task,))
    board.commit()
    migrate(board)
    diagnosed = consume(board)["diagnoses"][0]
    assert diagnosed["cycle_edges"] == [[task, parent, task]]
    assert diagnosed["route"] == "owner_review_cycle_edges_no_auto_unlink"
    assert board.execute("SELECT count(*) FROM task_links").fetchone()[0] == 2


def test_idle_does_not_reset_on_controller_comment_or_arbitrary_comment(board):
    task = block(board)
    migrate(board)
    first = consume(board)["diagnoses"][0]
    kb.add_comment(board, task, "blocker-controller", "audit")
    second = consume(board, now=first["blocked_at"] + IDLE_SECONDS)
    assert [r["task_id"] for r in second["idle_alerts"]] == [task]
    assert not consume(board, now=first["blocked_at"] + IDLE_SECONDS + 10)["idle_alerts"]
    kb.add_comment(board, task, "other", "approved")
    assert not consume(board)["diagnoses"]
    assert kb.get_task(board, task).status == "blocked"


def test_outcome_only_after_real_ready_or_running_transition(board):
    task = block(board)
    migrate(board)
    first = consume(board)["diagnoses"][0]
    board.execute("UPDATE tasks SET status='todo' WHERE id=?", (task,))
    kb._append_event(board, task, "status", {"to_status": "todo"})
    board.commit()
    assert not consume(board)["outcomes"]
    board.execute("UPDATE tasks SET status='ready' WHERE id=?", (task,))
    kb._append_event(board, task, "promoted", {})
    board.commit()
    result = consume(board)["outcomes"][0]
    assert result["blocked_at"] == first["blocked_at"]
    assert result["resolved_at"] >= result["blocked_at"]
    assert not consume(board)["outcomes"]


def test_real_unblock_ready_emits_outcome_without_approving_block(board):
    task = block(board)
    migrate(board)
    first = consume(board)["diagnoses"][0]
    assert kb.unblock_task(board, task)
    result = consume(board)["outcomes"]
    assert len(result) == 1
    assert result[0]["blocked_at"] == first["blocked_at"]
    assert result[0]["route"] == first["route"]
    assert not consume(board)["outcomes"]

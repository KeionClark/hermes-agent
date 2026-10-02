"""Event-stream diagnosis contract on a disposable board."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_blocked_diagnosis import consume, migrate, IDLE_SECONDS, RETENTION_SECONDS


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
    assert len(kb.list_comments(board, child)) == 1
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
    kb._append_event(board, task, "promoted", {"status": "review"})
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


def test_persistent_oldest_blocks_do_not_starve_new_eligible_outcome(board):
    tasks = [block(board) for _ in range(105)]
    migrate(board)
    consume(board)
    consume(board)
    target = tasks[-1]
    assert kb.unblock_task(board, target)
    outcome = consume(board)["outcomes"] + consume(board)["outcomes"]
    assert [row["task_id"] for row in outcome] == [target]
    assert kb.get_task(board, tasks[0]).status == "blocked"
    assert not consume(board)["outcomes"]
    persisted = board.execute(
        "SELECT blocked_at, resolved_at, transition_event_id FROM blocked_diagnosis_receipts "
        "WHERE task_id=? AND action='resolved'", (target,)).fetchone()
    assert persisted["blocked_at"] == outcome[0]["blocked_at"]
    assert persisted["resolved_at"] == outcome[0]["resolved_at"]
    assert persisted["transition_event_id"] == outcome[0]["transition_event_id"]
    assert not board.execute("SELECT 1 FROM blocked_diagnosis_receipts WHERE action='resolved' "
                             "AND task_id=?", (tasks[0],)).fetchone()


def test_retention_restart_and_schema_rollback(board):
    unresolved, resolved = block(board), block(board)
    migrate(board)
    consume(board)
    assert kb.unblock_task(board, resolved)
    outcome = consume(board)["outcomes"][0]
    db_path = board.execute("PRAGMA database_list").fetchone()["file"]
    # A new connection represents an interrupted watcher restart.
    import sqlite3
    reopened = sqlite3.connect(db_path)
    reopened.row_factory = sqlite3.Row
    try:
        migrate(reopened)
        assert not consume(reopened)["outcomes"]
        consume(reopened, now=outcome["blocked_at"] + RETENTION_SECONDS + 2)
        assert not reopened.execute("SELECT 1 FROM blocked_diagnosis_receipts WHERE task_id=?",
                                    (unresolved,)).fetchone()
        assert not reopened.execute("SELECT 1 FROM blocked_diagnosis_receipts WHERE task_id=?",
                                    (resolved,)).fetchone()
        before = reopened.execute("SELECT status FROM tasks WHERE id=?", (unresolved,)).fetchone()[0]
        for table in ("blocked_diagnosis_receipts", "blocked_diagnosis_cursor"):
            reopened.execute(f"DROP TABLE {table}")
        reopened.commit()
        assert reopened.execute("SELECT status FROM tasks WHERE id=?", (unresolved,)).fetchone()[0] == before
        with pytest.raises(sqlite3.OperationalError):
            consume(reopened)
    finally:
        reopened.close()


def test_resolved_metadata_retained_from_transition_and_legacy_migration(board):
    task = block(board)
    blocked = board.execute("SELECT id, created_at FROM task_events WHERE task_id=? AND kind='blocked'",
                            (task,)).fetchone()
    # Simulate an already-enabled old schema: migration is additive and backfills
    # the actual event time, not the time the old receipt was inserted.
    board.execute("CREATE TABLE blocked_diagnosis_cursor (singleton INTEGER PRIMARY KEY, event_id INTEGER NOT NULL)")
    board.execute("INSERT INTO blocked_diagnosis_cursor VALUES (1, ?)", (blocked["id"],))
    board.execute("CREATE TABLE blocked_diagnosis_receipts (task_id TEXT NOT NULL, blocked_event_id INTEGER NOT NULL, "
                  "action TEXT NOT NULL, created_at INTEGER NOT NULL, PRIMARY KEY(task_id, blocked_event_id, action))")
    board.execute("INSERT INTO blocked_diagnosis_receipts VALUES (?, ?, 'diagnose', ?)",
                  (task, blocked["id"], blocked["created_at"] + 100))
    board.commit()
    migrate(board)
    migrate(board)
    assert board.execute("SELECT blocked_at FROM blocked_diagnosis_receipts WHERE task_id=?",
                         (task,)).fetchone()[0] == blocked["created_at"]
    assert kb.unblock_task(board, task)
    outcome = consume(board)["outcomes"][0]
    assert outcome["transition_event_id"] > blocked["id"]
    # Age the block, but not the transition; the outcome remains readable.
    board.execute("UPDATE blocked_diagnosis_receipts SET blocked_at=? WHERE task_id=?",
                  (blocked["created_at"] - RETENTION_SECONDS, task))
    board.commit()
    consume(board, now=outcome["resolved_at"] + RETENTION_SECONDS - 1)
    assert board.execute("SELECT transition_event_id FROM blocked_diagnosis_receipts "
                         "WHERE task_id=? AND action='resolved'", (task,)).fetchone()[0] == outcome["transition_event_id"]
    consume(board, now=outcome["resolved_at"] + RETENTION_SECONDS + 1)
    assert not board.execute("SELECT 1 FROM blocked_diagnosis_receipts WHERE task_id=?", (task,)).fetchone()


def test_interrupted_tick_rolls_back_cursor_and_receipts(board, monkeypatch):
    task = block(board)
    migrate(board)
    import hermes_cli.kanban_blocked_diagnosis as diagnosis
    original = diagnosis._outcome
    def interrupted(*args):
        raise RuntimeError("interrupted before commit")
    monkeypatch.setattr(diagnosis, "_outcome", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        consume(board)
    assert board.execute("SELECT event_id FROM blocked_diagnosis_cursor").fetchone()[0] == 0
    assert not board.execute("SELECT 1 FROM blocked_diagnosis_receipts WHERE task_id=?", (task,)).fetchone()
    monkeypatch.setattr(diagnosis, "_outcome", original)
    assert [item["task_id"] for item in consume(board)["diagnoses"]] == [task]
    assert len(kb.list_comments(board, task)) == 1

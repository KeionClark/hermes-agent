"""Dark fresh-process Kanban checkpoint pilot contracts."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _repo(path: Path) -> Path:
    path.mkdir()
    for args in (("init", "-b", "main"), ("config", "user.email", "test@example.invalid"),
                 ("config", "user.name", "Test")):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)
    (path / "file.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(path), "commit", "-m", "base"], check=True, capture_output=True)
    return path


def _claimed(conn, repo: Path) -> tuple[str, int]:
    tid = kb.create_task(conn, title="pilot", body="original brief", assignee="worker",
                         workspace_kind="dir", workspace_path=str(repo))
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
    task = kb.claim_task(conn, tid, claimer=f"{kb._host_prefix()}pilot")
    assert task and task.current_run_id
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = 424242, worker_started_at = 1, started_at = 0, claim_expires = ? WHERE id = ?",
            (int(time.time()) + 60, tid),
        )
        conn.execute("UPDATE task_runs SET worker_pid = 424242, worker_started_at = 1 WHERE id = ?",
                     (task.current_run_id,))
    return tid, task.current_run_id


def _clean_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kbd, "_worker_alive", lambda *_: False)
    monkeypatch.setattr(kbd, "_classify_dead_worker", lambda *a, **k: kbd._DeadWorker(
        "clean_exit", 0, "", "protocol_violation", {}, protocol_violation=True))


def test_fresh_process_checkpoint_settles_same_card_into_new_run_and_narrow_context(kanban_home, tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    with kbc.connect() as conn:
        tid, run_id = _claimed(conn, repo)
        handoff = {"summary": "implemented parser", "next_step": "run focused tests", "external_actions": []}
        assert kb.stage_fresh_process_checkpoint(conn, tid, handoff=handoff, delay_seconds=1,
                                                 expected_run_id=run_id, workspace=str(repo))
        _clean_exit(monkeypatch)
        assert kbd.detect_crashed_workers(conn) == []
        parked = kb.get_task(conn, tid)
        assert parked.status == "scheduled" and parked.continuation_count == 1
        assert kb.resume_due_checkpoints(conn, now=parked.resume_after) == [tid]
        assert kb.claim_task(conn, tid, claimer=f"{kb._host_prefix()}next")
        context = kb.build_worker_context(conn, tid)
        assert "original brief" in context and "implemented parser" in context and "run focused tests" in context
        assert "Prior attempts" not in context and "Comments" not in context


def test_fresh_process_checkpoint_holds_changed_workspace_and_rejects_external_actions(kanban_home, tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    with kbc.connect() as conn:
        tid, run_id = _claimed(conn, repo)
        with pytest.raises(ValueError, match="external_actions"):
            kb.stage_fresh_process_checkpoint(conn, tid, handoff={
                "summary": "x", "next_step": "y", "external_actions": [{"send": "customer"}],
            }, delay_seconds=1, expected_run_id=run_id, workspace=str(repo))
        handoff = {"summary": "x", "next_step": "y", "external_actions": []}
        assert kb.stage_fresh_process_checkpoint(conn, tid, handoff=handoff, delay_seconds=1,
                                                 expected_run_id=run_id, workspace=str(repo))
        (repo / "file.txt").write_text("changed\n", encoding="utf-8")
        _clean_exit(monkeypatch)
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.latest_run(conn, tid).outcome == "fresh_process_checkpoint_rejected"


def test_fresh_process_checkpoint_respects_finite_continuation_cap(kanban_home, tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    with kbc.connect() as conn:
        tid, run_id = _claimed(conn, repo)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET continuation_count = ? WHERE id = ?", (kb.DEFAULT_CONTINUATION_LIMIT, tid))
        assert kb.stage_fresh_process_checkpoint(conn, tid, handoff={
            "summary": "cap", "next_step": "operator", "external_actions": [],
        }, delay_seconds=1, expected_run_id=run_id, workspace=str(repo))
        _clean_exit(monkeypatch)
        kbd.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked" and task.continuation_count == kb.DEFAULT_CONTINUATION_LIMIT + 1


def test_stale_fresh_checkpoint_trailer_cannot_mutate_a_successor_or_legacy_checkpoint(kanban_home, tmp_path):
    repo = _repo(tmp_path / "repo")
    with kbc.connect() as conn:
        tid, run_id = _claimed(conn, repo)
        # The ordinary summary path remains the canonical immediate checkpoint.
        assert kb.checkpoint_task(conn, tid, summary="legacy summary", delay_seconds=1,
                                  expected_run_id=run_id) == (True, False)
        assert kb.latest_run(conn, tid).summary == "legacy summary"
        assert kb.resume_due_checkpoints(conn, now=kb.get_task(conn, tid).resume_after) == [tid]
        successor = kb.claim_task(conn, tid, claimer=f"{kb._host_prefix()}successor")
        assert successor and successor.current_run_id
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO fresh_process_checkpoints "
                "(task_id, run_id, claim_lock, worker_pid, workspace_fingerprint, handoff, delay_seconds, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tid, run_id, "old", 7, "old", '{"summary":"x","next_step":"y","external_actions":[]}', 1, 0),
            )
            kb.reject_fresh_process_checkpoint_locked(conn, tid, run_id=run_id, reason="stale trailer")
        assert kb.get_task(conn, tid).current_run_id == successor.current_run_id
        assert conn.execute("SELECT 1 FROM fresh_process_checkpoints WHERE task_id = ?", (tid,)).fetchone() is None

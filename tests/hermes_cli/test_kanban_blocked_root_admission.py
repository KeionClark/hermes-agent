"""Admission guard: blocked human-input roots cannot grow worker fan-out."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _blocked_root(conn, *, kind: str = "needs_input") -> str:
    root = kb.create_task(conn, title="human gate", assignee="worker")
    assert kb.claim_task(conn, root)
    assert kb.block_task(
        conn, root, reason="awaiting a changed external condition", kind=kind,
        expected_run_id=kb.get_task(conn, root).current_run_id,
    )
    return root


def test_worker_fanout_below_needs_input_root_is_rejected(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = _blocked_root(conn)
        with pytest.raises(ValueError, match="blocked needs_input/capability ancestor"):
            kb.create_task(
                conn, title="speculative child", assignee="worker", parents=[root],
                creator_task_id=root, created_by="worker",
            )
        assert conn.execute("SELECT count(*) FROM task_links WHERE parent_id = ?", (root,)).fetchone()[0] == 0


def test_capability_root_and_creator_lineage_are_also_rejected(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = _blocked_root(conn, kind="capability")
        with pytest.raises(ValueError, match=root):
            kb.create_task(
                conn, title="unlinked speculative child", assignee="worker",
                creator_task_id=root, created_by="worker",
            )


def test_known_automation_identity_cannot_fan_out_below_blocked_root(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = _blocked_root(conn)
        with pytest.raises(ValueError, match="blocked needs_input/capability ancestor"):
            kb.create_task(
                conn, title="automatic child", assignee="worker", parents=[root],
                created_by="decomposer",
            )


def test_decomposer_cannot_fan_out_a_triage_child_below_blocked_root(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = _blocked_root(conn)
        triage = kb.create_task(conn, title="triage child", assignee="worker", parents=[root], triage=True)
        with pytest.raises(ValueError, match="blocked needs_input/capability ancestor"):
            decompose_triage_task(
                conn, triage, root_assignee="worker",
                children=[{"title": "automatic child", "assignee": "worker"}], author="decomposer",
            )


def test_explicit_changed_condition_resolver_is_replay_safe(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = _blocked_root(conn)
        purpose = "verify SSH route is now available"
        key = kb.blocked_root_resolver_idempotency_key(root, purpose)
        with pytest.raises(ValueError, match="requires idempotency_key"):
            kb.create_task(
                conn, title="bad resolver", assignee="worker",
                creator_task_id=root, created_by="worker", blocked_root_resolver_for=root,
                blocked_root_resolver_purpose=purpose, idempotency_key="wrong-key",
            )
        with pytest.raises(ValueError, match="must not depend on its blocked root"):
            kb.create_task(
                conn, title="deadlocked resolver", assignee="worker", parents=[root],
                creator_task_id=root, created_by="worker", blocked_root_resolver_for=root,
                blocked_root_resolver_purpose=purpose, idempotency_key=key,
            )
        first = kb.create_task(
            conn, title="Verify recovered SSH route", assignee="worker",
            creator_task_id=root, created_by="worker", blocked_root_resolver_for=root,
            blocked_root_resolver_purpose=purpose, idempotency_key=key,
        )
        second = kb.create_task(
            conn, title="duplicate retry", assignee="worker",
            creator_task_id=root, created_by="worker", blocked_root_resolver_for=root,
            blocked_root_resolver_purpose=purpose, idempotency_key=key,
        )
        assert second == first
        assert kb.get_task(conn, first).status == "ready"
        assert kb.claim_task(conn, first)
        assert conn.execute("SELECT count(*) FROM tasks WHERE idempotency_key = ?", (key,)).fetchone()[0] == 1
        created = [event for event in kb.list_events(conn, first) if event.kind == "created"]
        assert created[-1].payload["blocked_root_resolver"] == {"root": root, "purpose": purpose}


def test_manual_child_and_normal_worker_path_remain_allowed(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = _blocked_root(conn)
        manual = kb.create_task(conn, title="operator follow-up", assignee="worker", parents=[root])
        assert manual
        assert kb.get_task(conn, manual).status == "todo"

        assert kb.unblock_task(conn, root)
        assert kb.claim_task(conn, root)
        assert kb.complete_task(conn, root, result="external condition changed")
        child = kb.create_task(
            conn, title="ordinary worker child", assignee="worker", parents=[root],
            creator_task_id=root, created_by="worker",
        )
        assert kb.get_task(conn, child).status == "ready"

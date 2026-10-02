"""Regression coverage for the read-only blocked-root controller tick."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_blocker_controller import apply_tick, controller_tick


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _block(conn, task_id: str, *, kind: str = "needs_input") -> int:
    assert kb.claim_task(conn, task_id)
    run = kb.get_task(conn, task_id).current_run_id
    assert kb.block_task(conn, task_id, reason="awaiting external access", kind=kind, expected_run_id=run)
    events = kb.list_events(conn, task_id)
    return max(event.created_at for event in events if event.kind == "blocked")


def test_tick_groups_blocked_descendants_at_exact_root_and_is_read_only(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="access gate", assignee="chief", priority=100)
        blocked_at = _block(conn, root)
        child = kb.create_task(
            conn, title="blocked descendant", assignee="builder", parents=[root], initial_status="blocked",
        )
        ready = kb.create_task(conn, title="unclaimed urgent", assignee="release", priority=95)
        before = tuple(conn.execute("SELECT count(*) FROM " + table).fetchone()[0]
                       for table in ("tasks", "task_events", "task_links"))

        report = controller_tick(
            conn, now=blocked_at + 25 * 60 * 60,
            stale_gate_seconds=24 * 60 * 60, ready_starvation_seconds=60 * 60,
        )
        after = tuple(conn.execute("SELECT count(*) FROM " + table).fetchone()[0]
                      for table in ("tasks", "task_events", "task_links"))

    assert after == before
    assert report["ledger"] == [{
        "root_gate_id": root,
        "owner": "chief",
        "reason": "awaiting external access",
        "block_kind": "needs_input",
        "effective_priority": 100,
        "last_check_at": blocked_at,
        "next_check_at": blocked_at + 60 * 60,
        "blocked_task_ids": [root, child],
        "blocked_count": 2,
        "next_action": "verify the named changed condition; resolve only with evidence",
    }]
    due = next(item for item in report["escalations"] if item["kind"] == "gate_check_due")
    assert due["command"] == f"hermes kanban show {root}"
    assert due["escalation_key"] == f"gate-check-due:{root}:{blocked_at}"
    stale = next(item for item in report["escalations"] if item["kind"] == "stale_gate")
    assert stale["escalation_key"] == f"stale-gate:{root}:{blocked_at}"
    starving = next(item for item in report["escalations"] if item["kind"] == "ready_starvation")
    assert starving["task_id"] == ready
    assert starving["escalation_key"].startswith(f"ready-starvation:{ready}:")


def test_recent_gate_does_not_escalate_and_intervals_must_be_positive(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="recent gate", assignee="chief")
        blocked_at = _block(conn, root, kind="capability")
        report = controller_tick(conn, now=blocked_at + 10, stale_gate_seconds=60)
        assert report["ledger"][0]["root_gate_id"] == root
        assert report["escalations"] == []
        with pytest.raises(ValueError, match="intervals must be positive"):
            controller_tick(conn, now=blocked_at, stale_gate_seconds=0)


def test_apply_creates_one_runnable_resolver_and_targets_ready_alert(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="access gate", assignee="chief", priority=100)
        blocked_at = _block(conn, root)
        ready = kb.create_task(conn, title="unclaimed urgent", assignee="release", priority=95)
        report = controller_tick(
            conn, now=blocked_at + 25 * 60 * 60,
            stale_gate_seconds=24 * 60 * 60, ready_starvation_seconds=60 * 60,
        )
        first = apply_tick(conn, report)
        event_count = conn.execute("SELECT count(*) FROM task_events").fetchone()[0]
        task_count = conn.execute("SELECT count(*) FROM tasks").fetchone()[0]
        second = apply_tick(conn, report)

        assert len(first["resolver_task_ids"]) == 1
        resolver = kb.get_task(conn, first["resolver_task_ids"][0])
        assert resolver.status == "ready"
        assert resolver.assignee == "chief"
        assert "@chief must post `[changed-condition] <specific evidence>`" in resolver.body
        assert first["ready_alert_task_ids"] == [ready]
        assert second["resolver_task_ids"] == []
        assert second["ready_alert_task_ids"] == []
        assert conn.execute("SELECT count(*) FROM task_events").fetchone()[0] == event_count
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == task_count
        assert kb.get_task(conn, root).status == "blocked"
        assert len(kb.list_comments(conn, ready)) == 1


def test_outer_gate_and_descendant_priority_win(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        outer = kb.create_task(conn, title="outer", assignee="chief", priority=10)
        blocked_at = _block(conn, outer)
        inner = kb.create_task(conn, title="inner", assignee="chief", parents=[outer], initial_status="blocked")
        assert kb.block_task(conn, inner, reason="inner capability", kind="capability")
        child = kb.create_task(conn, title="urgent child", assignee="chief", parents=[inner], initial_status="blocked", priority=100)
        report = controller_tick(conn, now=blocked_at + 61 * 60)
        entry = next(item for item in report["ledger"] if child in item["blocked_task_ids"])
        assert entry["root_gate_id"] == outer
        assert entry["effective_priority"] == 100
        applied = apply_tick(conn, report)
        resolver = kb.get_task(conn, applied["resolver_task_ids"][0])
        assert resolver.title == "Resolve gate: outer"


def test_diamond_graph_is_cycle_safe_and_groups_at_outer_gate(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="root", assignee="chief", priority=10)
        blocked_at = _block(conn, root)
        left = kb.create_task(conn, title="left", assignee="builder", parents=[root], initial_status="blocked")
        right = kb.create_task(conn, title="right", assignee="builder", parents=[root], initial_status="blocked")
        leaf = kb.create_task(
            conn, title="P100 diamond leaf", assignee="builder", parents=[left, right],
            initial_status="blocked", priority=100,
        )
        report = controller_tick(conn, now=blocked_at + 61 * 60)
        entry = next(item for item in report["ledger"] if leaf in item["blocked_task_ids"])
        assert entry["root_gate_id"] == root
        assert entry["effective_priority"] == 100



def test_merged_blocked_roots_choose_one_resolver_scope(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        left = kb.create_task(conn, title="left root", assignee="left", priority=90)
        right = kb.create_task(conn, title="right root", assignee="right", priority=90)
        left_at, right_at = _block(conn, left), _block(conn, right)
        leaf = kb.create_task(
            conn, title="merged P100 leaf", assignee="builder", parents=[left, right],
            initial_status="blocked", priority=100,
        )
        report = controller_tick(conn, now=max(left_at, right_at) + 61 * 60)
        entry = next(item for item in report["ledger"] if leaf in item["blocked_task_ids"])
        assert entry["root_gate_id"] == min(left, right)
        applied = apply_tick(conn, report)
        assert len(applied["resolver_task_ids"]) == 1
        resolver = kb.get_task(conn, applied["resolver_task_ids"][0])
        assert resolver.title == f"Resolve gate: {kb.get_task(conn, min(left, right)).title}"
        assert resolver.status == "ready"


def test_systemd_text_is_bounded_and_not_installed(kanban_home: Path) -> None:
    from hermes_cli.kanban_blocker_controller import systemd_units
    units = systemd_units()
    assert "OnUnitActiveSec=15m" in units["timer"]
    assert "hermes --run-module hermes_cli.kanban_blocker_controller --apply --json" in units["service"]


def test_workspace_crash_root_is_classified_before_one_resolver(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="missing checkout", assignee="release", initial_status="blocked")
        conn.execute("UPDATE tasks SET last_failure_error = ? WHERE id = ?", ("workspace checkout route missing", root))
        conn.commit()
        blocked_at = max(event.created_at for event in kb.list_events(conn, root) if event.kind == "blocked")
        report = controller_tick(conn, now=blocked_at + 25 * 60 * 60)
        applied = apply_tick(conn, report)
        assert kb.get_task(conn, root).block_kind == "capability"
        assert len(applied["resolver_task_ids"]) == 1
        assert kb.get_task(conn, applied["resolver_task_ids"][0]).status == "ready"


def test_terminal_resolver_waits_for_explicit_changed_condition_marker(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="access gate", assignee="chief", priority=100)
        blocked_at = _block(conn, root)
        report = controller_tick(conn, now=blocked_at + 25 * 60 * 60)
        first = apply_tick(conn, report)
        resolver_id = first["resolver_task_ids"][0]
        assert kb.claim_task(conn, resolver_id)
        assert kb.complete_task(conn, resolver_id, result="condition still unavailable")
        assert kb.get_task(conn, root).status == "blocked"
        assert apply_tick(conn, report)["resolver_task_ids"] == []

        kb.add_comment(conn, root, "other", "[changed-condition] repository route has been restored")
        assert apply_tick(conn, report)["resolver_task_ids"] == []
        kb.add_comment(conn, root, "chief", "[changed-condition] repository route has been restored")
        changed = apply_tick(conn, report)
        assert len(changed["resolver_task_ids"]) == 1
        assert changed["resolver_task_ids"][0] != resolver_id
        assert kb.get_task(conn, changed["resolver_task_ids"][0]).status == "ready"


def test_apply_selects_highest_priority_critical_root_first(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        low = kb.create_task(conn, title="P90 root", assignee="chief", priority=90)
        high = kb.create_task(conn, title="P100 root", assignee="chief", priority=100)
        low_at, high_at = _block(conn, low), _block(conn, high)
        report = controller_tick(conn, now=max(low_at, high_at) + 25 * 60 * 60)
        applied = apply_tick(conn, report, max_resolvers=1)
        resolver = kb.get_task(conn, applied["resolver_task_ids"][0])
        assert "P100 root" in resolver.title


def test_changed_condition_does_not_duplicate_an_active_resolver(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="access gate", assignee="chief", priority=100)
        blocked_at = _block(conn, root)
        report = controller_tick(conn, now=blocked_at + 25 * 60 * 60)
        first = apply_tick(conn, report)
        kb.add_comment(conn, root, "chief", "[changed-condition] route changed")
        assert apply_tick(conn, report)["resolver_task_ids"] == []
        assert len([task for task in kb.list_tasks(conn) if task.title.startswith("Resolve gate:")]) == 1
        resolver = first["resolver_task_ids"][0]
        assert kb.claim_task(conn, resolver)
        assert kb.complete_task(conn, resolver, result="old verification finished")
        second = apply_tick(conn, report)
        assert len(second["resolver_task_ids"]) == 1
        assert second["resolver_task_ids"][0] != resolver

"""Regression coverage for evidenced blocked-card resolution."""

from __future__ import annotations

from pathlib import Path
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban as kc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _blocked(conn, title="blocked task", *, parents=()):
    task_id = kb.create_task(conn, title=title, parents=list(parents), assignee="builder")
    assert kb.block_task(conn, task_id, reason="missing scoped credential", kind="capability")
    return task_id


def test_resolve_block_records_evidence_and_preserves_block_history(kanban_home):
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        before = kb.get_task(conn, task_id)
        assert kb.resolve_block_and_requeue(
            conn, task_id, actor="chief-of-staff", evidence_ref="ref:0123456789abcdef",
        )
        after = kb.get_task(conn, task_id)
        assert after.status == "ready"
        assert after.block_kind == before.block_kind == "capability"
        assert after.block_recurrences == before.block_recurrences == 1
        comments = kb.list_comments(conn, task_id)
        assert comments == []
        events = kb.list_events(conn, task_id)
        resolution = next(e for e in events if e.kind == "block_resolved")
        assert resolution.payload["actor"] == "chief-of-staff"
        assert resolution.payload["evidence_ref"] == "ref:0123456789abcdef"
        assert "evidence" not in resolution.payload
        assert resolution.payload["status"] == "ready"
        original_block = next(e for e in events if e.kind == "blocked")
        assert original_block.payload["reason"] == "missing scoped credential"


def test_resolve_block_parent_gate_and_repeat_are_safe(kanban_home):
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="unfinished parent", assignee="builder")
        task_id = _blocked(conn)
        conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", (parent, task_id))
        conn.commit()
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.resolve_block_and_requeue(conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef")
        assert kb.get_task(conn, task_id).status == "todo"
        assert not kb.resolve_block_and_requeue(conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef")
        assert kb.list_comments(conn, task_id) == []


def test_resolve_block_rejects_blank_audit_fields(kanban_home):
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        before_events = kb.list_events(conn, task_id)
        before_comments = kb.list_comments(conn, task_id)
        with pytest.raises(ValueError, match="actor"):
            kb.resolve_block_and_requeue(conn, task_id, actor=" ", evidence_ref="run:0123456789abcdef")
        with pytest.raises(ValueError, match="evidence reference"):
            kb.resolve_block_and_requeue(conn, task_id, actor="ops", evidence_ref="human sentence")
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.list_events(conn, task_id) == before_events
        assert kb.list_comments(conn, task_id) == before_comments


def test_resolve_block_rejects_invalid_parent_atomically(kanban_home):
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", ("missing-parent", task_id))
        conn.commit()
        before_events = kb.list_events(conn, task_id)
        before_comments = kb.list_comments(conn, task_id)
        with pytest.raises(ValueError, match="invalid parent"):
            kb.resolve_block_and_requeue(
                conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef",
            )
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.list_events(conn, task_id) == before_events
        assert kb.list_comments(conn, task_id) == before_comments


def test_resolve_block_rejects_invalid_parent_atomically(kanban_home):
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", ("missing-parent", task_id))
        conn.commit()
        before_events = kb.list_events(conn, task_id)
        before_comments = kb.list_comments(conn, task_id)
        with pytest.raises(ValueError, match="invalid parent"):
            kb.resolve_block_and_requeue(
                conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef",
            )
        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.list_events(conn, task_id) == before_events
        assert kb.list_comments(conn, task_id) == before_comments


def test_latest_activity_prefers_new_comment_over_old_run_summary(kanban_home):
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        kb.add_comment(conn, task_id, "chief-of-staff", "New credential is available")
        preview = kb.latest_activity_previews(conn, [task_id])[task_id]
        assert preview["kind"] == "comment"
        assert preview["author"] == "chief-of-staff"
        assert preview["text"] == "New credential is available"


def test_resolution_metadata_expires_after_30_days_while_task_is_active(kanban_home):
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        assert kb.resolve_block_and_requeue(
            conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef",
        )
        conn.execute("UPDATE task_events SET created_at = 0 WHERE task_id = ?", (task_id,))
        conn.commit()
        assert kb.gc_events(conn, older_than_seconds=30 * 24 * 3600) == 2
        assert not any(e.kind == "block_resolved" for e in kb.list_events(conn, task_id))


def test_resolution_metadata_gc_keeps_generic_history_when_disabled(kanban_home):
    """The daily safety sweep cannot be disabled with the generic CLI flag."""
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        assert kb.resolve_block_and_requeue(
            conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef",
        )
        conn.execute("UPDATE task_events SET created_at = 0 WHERE task_id = ?", (task_id,))
        conn.commit()
        assert kb.gc_events(conn, older_than_seconds=0) == 2
        events = kb.list_events(conn, task_id)
        assert {event.kind for event in events} == {"created", "blocked"}


def test_resolution_metadata_is_hidden_at_the_30_day_boundary_before_gc(kanban_home):
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        assert kb.resolve_block_and_requeue(
            conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef",
        )
        boundary = int(time.time()) - kb.BLOCK_RESOLUTION_METADATA_MAX_AGE_SECONDS
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE task_id = ? AND "
            "(kind = 'block_resolved' OR (kind = 'unblocked' AND payload LIKE '%block_resolved%'))",
            (boundary, task_id),
        )
        conn.commit()
        assert {event.kind for event in kb.list_events(conn, task_id)} == {"created", "blocked"}
        assert conn.execute("SELECT count(*) FROM task_events WHERE task_id = ?", (task_id,)).fetchone()[0] == 4


def test_resolution_metadata_is_physically_purged_before_the_30_day_limit(kanban_home):
    with kbc.connect() as conn:
        task_id = _blocked(conn)
        assert kb.resolve_block_and_requeue(
            conn, task_id, actor="ops", evidence_ref="case:0123456789abcdef",
        )
        purge_boundary = int(time.time()) - kb.BLOCK_RESOLUTION_METADATA_PURGE_AGE_SECONDS
        conn.execute(
            "UPDATE task_events SET created_at = ? WHERE task_id = ? AND "
            "(kind = 'block_resolved' OR (kind = 'unblocked' AND payload LIKE '%block_resolved%'))",
            (purge_boundary, task_id),
        )
        conn.commit()
        assert kb.gc_events(conn, older_than_seconds=0) == 2
        assert {event.kind for event in kb.list_events(conn, task_id)} == {"created", "blocked"}


def test_cli_resolve_block_records_opaque_reference_and_profile_actor(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "release-engineer")
    with kbc.connect() as conn:
        task_id = _blocked(conn)
    output = kc.run_slash(f"resolve-block {task_id} run:0123456789abcdef")
    assert "Resolved" in output
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
        event = next(e for e in kb.list_events(conn, task_id) if e.kind == "block_resolved")
        assert event.payload["actor"] == "release-engineer"
        assert event.payload["evidence_ref"] == "run:0123456789abcdef"

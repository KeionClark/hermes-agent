"""Atomic persisted Kanban receipt identity on real SQLite boards."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_output import _task_to_dict
from tools import kanban_tools


def test_concurrent_replay_board_scope_and_supported_readback(tmp_path):
    path = tmp_path / "board-a.db"
    kbc.init_db(path)
    barrier = Barrier(8)

    def create(i):
        with kbc.connect(path) as conn:
            barrier.wait()
            return kb.create_task(conn, title=f"candidate {i}", assignee="builder", idempotency_key="exact:key")

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(create, range(8)))
    assert len(set(ids)) == 1
    with kbc.connect(path) as conn:
        task = kb.get_task(conn, ids[0])
        assert task.assignee == "builder"
        assert _task_to_dict(task)["idempotency_key"] == "exact:key"
        assert kanban_tools._fields(task, kanban_tools._TASK_FIELDS)["idempotency_key"] == "exact:key"
        assert conn.execute("SELECT count(*) FROM tasks WHERE idempotency_key = ?", ("exact:key",)).fetchone()[0] == 1
        conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (ids[0],))
        conn.commit()
        assert kb.create_task(conn, title="replayed archived", idempotency_key="exact:key") == ids[0]
        assert kb.create_task(conn, title="distinct", idempotency_key="other") != ids[0]
        assert kb.create_task(conn, title="unkeyed") != kb.create_task(conn, title="unkeyed")
    with kbc.connect(tmp_path / "board-b.db") as conn:
        assert kb.create_task(conn, title="independent board", idempotency_key="exact:key") != ids[0]


def test_duplicate_legacy_keys_fail_closed_without_losing_rows(tmp_path):
    path = tmp_path / "duplicate.db"
    with kbc.connect(path) as conn:
        a = kb.create_task(conn, title="one", idempotency_key="same")
        b = kb.create_task(conn, title="two", idempotency_key="different")
        conn.execute("DROP INDEX idx_tasks_idempotency_unique")
        conn.execute("UPDATE tasks SET idempotency_key = 'same' WHERE id = ?", (b,))
        conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        kbc.init_db(path)
    with sqlite3.connect(path) as raw:
        assert {row[0] for row in raw.execute("SELECT id FROM tasks WHERE idempotency_key = 'same'")} == {a, b}

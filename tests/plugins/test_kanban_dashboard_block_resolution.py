"""Dashboard coverage for the evidenced blocked-card resolution endpoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def _router():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_block_resolution", repo_root / "plugins/kanban/dashboard/plugin_api.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.include_router(_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def test_resolve_block_endpoint_requires_audit_fields_and_requeues(client):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="blocked", assignee="builder")
        assert kb.block_task(conn, task_id, reason="need read-only access", kind="capability")

    missing = client.post(f"/api/plugins/kanban/tasks/{task_id}/resolve-block", json={"actor": "ops", "evidence": " "})
    assert missing.status_code == 400
    resolved = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/resolve-block",
        json={"actor": "ops", "evidence": "read-only path tested"},
    )
    assert resolved.status_code == 200
    assert resolved.json()["task"]["status"] == "ready"
    board = client.get("/api/plugins/kanban/board").json()
    ready = next(c for c in board["columns"] if c["name"] == "ready")["tasks"]
    row = next(t for t in ready if t["id"] == task_id)
    assert row["latest_activity"]["kind"] == "comment"
    assert row["latest_activity"]["text"] == "Block resolved: read-only path tested"


def test_board_activity_preview_is_bounded(client):
    long_evidence = "x" * 260
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="blocked", assignee="builder")
        assert kb.block_task(conn, task_id, reason="need access", kind="capability")
    response = client.post(
        f"/api/plugins/kanban/tasks/{task_id}/resolve-block",
        json={"actor": "ops", "evidence": long_evidence},
    )
    assert response.status_code == 200
    board = client.get("/api/plugins/kanban/board").json()
    ready = next(c for c in board["columns"] if c["name"] == "ready")["tasks"]
    row = next(t for t in ready if t["id"] == task_id)
    assert len(row["latest_activity"]["text"]) == 200

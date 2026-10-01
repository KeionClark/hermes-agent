"""Evidence, capacity, scoring and replay contracts for the dark proposal queue."""
from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli.event_bus import migrate_event_bus
from hermes_cli.goal_metrics import (
    activate_goal, create_goal, migrate_goal_schema, record_measurement,
)
from hermes_cli.proposal_queue import (
    apply_evaluated_outcome, list_proposals, migrate_proposal_schema,
    propose_for_gap, revert_evaluated_outcome,
)


def _open(path):
    conn = sqlite3.connect(path, isolation_level=None, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@pytest.fixture
def board(tmp_path):
    conn = _open(tmp_path / "board.db")
    migrate_event_bus(conn)
    migrate_goal_schema(conn)
    migrate_proposal_schema(conn)
    migrate_proposal_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    create_goal(conn, goal_id="g", metric_id="metric", source_id="source",
                target_value=100, max_age_seconds=8 * 86400)
    activate_goal(conn, "g", "approval", time.time())
    now = time.time()
    measurement = record_measurement(
        conn, goal_id="g", metric_id="metric", value=50, source_id="source",
        source_timestamp=now - 2, observed_at=now - 1,
        source_ref="fixture-provenance", idempotency_key="reading-1",
    ).measurement
    conn.execute("COMMIT")
    yield conn, measurement.id, now, tmp_path / "board.db"
    conn.close()


def _propose(conn, measurement_id, now, **overrides):
    data = dict(goal_id="g", metric_id="metric", gap_measurement_id=measurement_id,
                action_type="investigate", hypothesis="Missing signal",
                proposed_action="Review evidence", impact=0.8, confidence=0.7,
                cost=0.1, decision_tier="agent", idempotency_key="one", now=now)
    data.update(overrides)
    return propose_for_gap(conn, **data)


@pytest.mark.parametrize("cost", [0, -1, float("nan"), float("inf")])
def test_cost_rejected_without_side_effects(board, cost):
    conn, mid, now, _ = board
    with pytest.raises(ValueError):
        _propose(conn, mid, now, cost=cost)
    assert not list_proposals(conn, now=now)


def test_normalized_identity_and_retry_do_not_fuzzy_merge(board):
    conn, mid, now, _ = board
    first = _propose(conn, mid, now)
    assert first == _propose(conn, mid, now)
    assert first.id == _propose(conn, mid, now, hypothesis="  MISSING  signal  ",
                                idempotency_key="two").id
    different = _propose(conn, mid, now, hypothesis="Missing signals",
                         idempotency_key="three")
    assert different.id != first.id
    with pytest.raises(ValueError, match="conflicting"):
        _propose(conn, mid, now, impact=0.3)
    assert all(0 <= p.score <= 10 for p in list_proposals(conn, now=now))
    assert first.confidence == 0.3  # uncalibrated action type prior


def test_concurrent_connections_one_identity_and_capacity(board):
    conn, mid, now, path = board
    def worker(key):
        other = _open(path)
        try:
            return _propose(other, mid, now, idempotency_key=key).id
        finally:
            other.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(worker, ("parallel-1", "parallel-2")))
    assert ids[0] == ids[1]
    for i in range(1, 10):
        _propose(conn, mid, now, hypothesis=f"hypothesis {i}",
                 idempotency_key=f"capacity-{i}")
    with pytest.raises(ValueError, match="capacity"):
        _propose(conn, mid, now, hypothesis="eleventh", idempotency_key="eleven")
    assert len(list_proposals(conn, now=now)) == 10
    assert not list_proposals(conn, now=now + 7 * 86400)
    newer = _propose(conn, mid, now + 7 * 86400,
                     hypothesis="eleventh", idempotency_key="eleven")
    assert newer.id not in ids
    assert len(list_proposals(conn, now=now + 7 * 86400)) == 1
    assert _propose(conn, mid, now, idempotency_key="parallel-1").id in ids


def test_rejects_unverified_or_old_gap(board):
    conn, mid, now, _ = board
    with pytest.raises(ValueError, match="gap"):
        _propose(conn, mid, now + 9 * 86400)
    conn.execute("BEGIN IMMEDIATE")
    record_measurement(conn, goal_id="g", metric_id="metric", value=110,
                       source_id="source", source_timestamp=now,
                       observed_at=now, source_ref="new-reading",
                       idempotency_key="reading-2")
    conn.execute("COMMIT")
    with pytest.raises(ValueError, match="gap"):
        _propose(conn, mid, now)
    conn.execute("UPDATE goals SET metric_version = metric_version + 1 WHERE id = 'g'")
    with pytest.raises(ValueError, match="version|gap"):
        _propose(conn, mid, now)


def test_evaluated_comparable_only_and_exactly_once_revert(board):
    conn, before, now, _ = board
    proposal = _propose(conn, before, now)
    conn.execute("BEGIN IMMEDIATE")
    after = record_measurement(
        conn, goal_id="g", metric_id="metric", value=70, source_id="source",
        source_timestamp=now, observed_at=now, source_ref="after-evidence",
        idempotency_key="after",
    ).measurement.id
    conn.execute("COMMIT")
    # The outcome ledger is built in an independent branch. This fixture
    # exercises its agreed persisted columns; integration tests use the real API.
    conn.execute("""CREATE TABLE outcomes (
        id TEXT PRIMARY KEY, status TEXT, goal_id TEXT, goal_metric_version INTEGER,
        expected_direction TEXT, action_type TEXT, hypothesis TEXT, before_measurement_id INTEGER,
        after_measurement_id INTEGER, before_value REAL, after_value REAL,
        before_source_timestamp REAL, after_source_timestamp REAL,
        before_source_ref TEXT, after_source_ref TEXT,
        before_observed_at REAL, after_observed_at REAL,
        started_at REAL, completed_at REAL, settling_seconds INTEGER,
        evaluated_at REAL, confidence_delta REAL)""")
    first = conn.execute("SELECT * FROM goal_measurements WHERE id = ?", (before,)).fetchone()
    second = conn.execute("SELECT * FROM goal_measurements WHERE id = ?", (after,)).fetchone()
    conn.execute("""INSERT INTO outcomes VALUES (
        'o', 'confirmed', 'g', 1, 'higher', 'investigate', 'Missing signal',
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0.2)""",
        (before, after, first["value"], second["value"],
         first["source_timestamp"], second["source_timestamp"],
         first["source_ref"], second["source_ref"],
         first["observed_at"], second["observed_at"], now - 1, now - 0.5, now + 1))
    assert apply_evaluated_outcome(conn, "absent") is None
    conn.execute("UPDATE outcomes SET evaluated_at = ? WHERE id = 'o'",
                 (now + 9 * 86400,))
    assert apply_evaluated_outcome(conn, "o") is None  # stale after
    conn.execute("UPDATE outcomes SET evaluated_at = ? WHERE id = 'o'", (now + 1,))
    assert apply_evaluated_outcome(conn, "o") == pytest.approx(0.4)
    assert apply_evaluated_outcome(conn, "o") == pytest.approx(0.4)
    assert list_proposals(conn, now=now)[0].score > proposal.score
    conn.execute("UPDATE outcomes SET status = 'reverted' WHERE id = 'o'")
    assert revert_evaluated_outcome(conn, "o") == pytest.approx(0.3)
    assert revert_evaluated_outcome(conn, "o") == pytest.approx(0.3)
    assert list_proposals(conn, now=now)[0].score == pytest.approx(proposal.score)
    assert apply_evaluated_outcome(conn, "o") is None
    assert not list_proposals(conn, now=now + 7 * 86400)


def test_no_confidence_from_missing_or_wrong_version_evidence(board):
    conn, before, now, _ = board
    proposal = _propose(conn, before, now)
    conn.execute("""CREATE TABLE outcomes (
        id TEXT PRIMARY KEY, status TEXT, goal_id TEXT, goal_metric_version INTEGER,
        expected_direction TEXT, action_type TEXT, hypothesis TEXT, before_measurement_id INTEGER,
        after_measurement_id INTEGER)""")
    conn.execute("INSERT INTO outcomes VALUES ('missing', 'confirmed', 'g', 1, 'higher', "
                 "'investigate', 'Missing signal', ?, NULL)", (before,))
    assert apply_evaluated_outcome(conn, "missing") is None
    conn.execute("UPDATE goals SET metric_version = 2 WHERE id = 'g'")
    conn.execute("UPDATE outcomes SET after_measurement_id = ? WHERE id = 'missing'", (before,))
    assert apply_evaluated_outcome(conn, "missing") is None
    assert list_proposals(conn, now=now)[0].score == proposal.score

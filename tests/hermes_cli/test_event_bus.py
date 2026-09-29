"""Tests for hermes_cli.event_bus — P0a event substrate.

Run via ``scripts/run_tests.sh tests/hermes_cli/test_event_bus.py``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Ensure the repo root is importable.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hermes_cli.event_bus import (
    Event,
    _ALLOWED_PAYLOAD_KEYS,
    _FORBIDDEN_PAYLOAD_KEYS,
    _MAX_FIELD_LEN,
    _validate_payload,
    ack,
    migrate_event_bus,
    poll,
    publish,
    subscribe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_db() -> sqlite3.Connection:
    """Return a WAL-mode temp-file DB with event-bus schema.

    Uses autocommit for migration DDL, then switches to manual transactions
    for test code.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    conn = sqlite3.connect(tmp.name, autocommit=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    migrate_event_bus(conn)
    # Switch to manual transaction management for test code.
    conn.autocommit = False
    conn.execute("COMMIT")  # End any implicit txn from migration.
    return conn


def _begin(conn: sqlite3.Connection) -> None:
    """Start a write transaction for publish/ack calls."""
    # Ensure no implicit transaction is open.
    if conn.in_transaction:
        conn.execute("COMMIT")
    conn.execute("BEGIN IMMEDIATE")


def _commit(conn: sqlite3.Connection) -> None:
    conn.execute("COMMIT")


def _rollback(conn: sqlite3.Connection) -> None:
    conn.execute("ROLLBACK")


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------

class TestMigration:
    """Schema migration: idempotent, atomic, independent of task_events."""

    def test_fresh_db_creates_all_tables(self):
        conn = _fresh_db()
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "hermes_events" in tables
        assert "hermes_event_cursors" in tables
        assert "hermes_event_schema_version" in tables
        conn.close()

    def test_idempotent_migration(self):
        conn = _fresh_db()
        # Run migration a second time — should be a no-op.
        migrate_event_bus(conn)
        row = conn.execute(
            "SELECT version FROM hermes_event_schema_version WHERE id = 1"
        ).fetchone()
        assert row[0] == 1
        conn.close()

    def test_unknown_future_version_raises(self):
        conn = _fresh_db()
        conn.execute(
            "UPDATE hermes_event_schema_version SET version = 999 WHERE id = 1"
        )
        with pytest.raises(RuntimeError, match="too old"):
            migrate_event_bus(conn)
        conn.close()

    def test_migration_independent_of_task_events(self):
        """Event bus tables coexist with task_events without conflict."""
        conn = _fresh_db()
        # Create a task_events table (simulating pre-existing kanban schema).
        conn.execute(
            "CREATE TABLE IF NOT EXISTS task_events "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, "
            "run_id INTEGER, kind TEXT NOT NULL, payload TEXT, "
            "created_at INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES ('t_test', 'created', 1000)"
        )

        # Re-run migration — no interference.
        migrate_event_bus(conn)

        # Original task_events row intact.
        row = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = 't_test'"
        ).fetchone()
        assert row[0] == "created"
        conn.close()

    def test_legacy_board_with_task_events_preserved(self):
        """Pre-existing task_events rows remain after migration."""
        conn = _fresh_db()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS task_events "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, "
            "kind TEXT, payload TEXT, created_at INTEGER)"
        )
        # Insert 5 legacy events.
        for i in range(5):
            conn.execute(
                "INSERT INTO task_events (task_id, kind, created_at) "
                "VALUES (?, ?, ?)",
                (f"t_{i}", "status_change", 1000 + i),
            )
        migrate_event_bus(conn)
        count = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        assert count == 5
        conn.close()

    def test_schema_version_table_indexes(self):
        """Verify expected indexes exist."""
        conn = _fresh_db()
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_events_idempotency" in indexes
        assert "idx_events_type_id" in indexes
        conn.close()

    def test_cursor_table_check_constraint(self):
        """last_event_id CHECK(last_event_id >= 0) enforced."""
        conn = _fresh_db()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO hermes_event_cursors "
                "(subscriber_id, event_types, last_event_id, created_at, updated_at) "
                "VALUES ('test', 'null', -1, 0, 0)"
            )
        conn.close()

    def test_migration_atomic_on_fresh_db(self):
        """A failure mid-DDL leaves NO partial schema on a fresh DB.

        Inject a SQLite authorizer that denies CREATE INDEX (the third DDL
        statement).  After the expected DatabaseError, no event-bus tables
        or version row should persist — the migration must roll back
        entirely.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")

        def deny_create_index(action, arg1, arg2, db, trigger):
            if action == sqlite3.SQLITE_CREATE_INDEX:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(deny_create_index)

        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            migrate_event_bus(conn)

        # Remove authorizer so we can inspect freely.
        conn.set_authorizer(None)

        # No bus tables should exist.
        bus_tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'hermes_event%' ORDER BY name"
            ).fetchall()
        ]
        assert bus_tables == [], (
            f"partial schema persisted after failed migration: {bus_tables}"
        )

        # No version row should exist.
        version_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name = 'hermes_event_schema_version'"
        ).fetchall()
        assert version_tables == [], "version table persisted after rollback"

        conn.close()
        os.unlink(db_path)

    def test_migration_atomic_on_legacy_db(self):
        """A failure mid-DDL on a legacy board preserves existing tables.

        Simulate a legacy board that already has task_events and a notifier
        table.  After the failed migration, those tables must still be
        intact and no event-bus tables should appear.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")

        # Create pre-existing legacy tables.
        conn.execute(
            "CREATE TABLE task_events "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, "
            "kind TEXT, payload TEXT, created_at INTEGER)"
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES ('t_legacy', 'created', 1000)"
        )
        conn.execute(
            "CREATE TABLE notifier_state "
            "(key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT INTO notifier_state (key, value) "
            "VALUES ('last_seen', '42')"
        )

        def deny_create_index(action, arg1, arg2, db, trigger):
            if action == sqlite3.SQLITE_CREATE_INDEX:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(deny_create_index)

        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            migrate_event_bus(conn)

        conn.set_authorizer(None)

        # No bus tables should have leaked.
        bus_tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name LIKE 'hermes_event%' ORDER BY name"
            ).fetchall()
        ]
        assert bus_tables == [], (
            f"partial schema persisted after failed migration: {bus_tables}"
        )

        # Legacy tables intact.
        row = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = 't_legacy'"
        ).fetchone()
        assert row[0] == "created", "legacy task_events row corrupted"

        notifier = conn.execute(
            "SELECT value FROM notifier_state WHERE key = 'last_seen'"
        ).fetchone()
        assert notifier[0] == "42", "legacy notifier_state row corrupted"

        conn.close()
        os.unlink(db_path)

    def test_migration_succeeds_after_prior_failure(self):
        """After a failed migration, a retry without the fault succeeds."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")

        def deny_create_index(action, arg1, arg2, db, trigger):
            if action == sqlite3.SQLITE_CREATE_INDEX:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(deny_create_index)

        with pytest.raises(sqlite3.DatabaseError):
            migrate_event_bus(conn)

        # Remove fault and retry.
        conn.set_authorizer(None)
        migrate_event_bus(conn)

        # Full schema present.
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "hermes_events" in tables
        assert "hermes_event_cursors" in tables
        assert "hermes_event_schema_version" in tables

        row = conn.execute(
            "SELECT version FROM hermes_event_schema_version WHERE id = 1"
        ).fetchone()
        assert row[0] == 1

        conn.close()
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Publish tests
# ---------------------------------------------------------------------------

class TestPublish:
    """publish() contract."""

    def test_basic_publish(self):
        conn = _fresh_db()
        _begin(conn)
        evt = publish(
            conn,
            event_type="task.status_change",
            source="kanban",
            payload={"task_id": "t_123", "from": "todo", "to": "running"},
        )
        _commit(conn)
        assert isinstance(evt, Event)
        assert evt.id >= 1
        assert evt.event_type == "task.status_change"
        assert evt.source == "kanban"
        assert evt.payload == {"task_id": "t_123", "from": "todo", "to": "running"}
        assert evt.idempotency_key is None
        assert evt.created_at > 0
        conn.close()

    def test_publish_requires_transaction(self):
        conn = _fresh_db()
        with pytest.raises(RuntimeError, match="write transaction"):
            publish(
                conn,
                event_type="test",
                source="test",
                payload={"task_id": "t_1"},
            )
        conn.close()

    def test_publish_empty_event_type_rejected(self):
        conn = _fresh_db()
        _begin(conn)
        with pytest.raises(ValueError, match="event_type"):
            publish(conn, event_type="", source="test", payload={"task_id": "t_1"})
        _rollback(conn)
        conn.close()

    def test_publish_empty_source_rejected(self):
        conn = _fresh_db()
        _begin(conn)
        with pytest.raises(ValueError, match="source"):
            publish(conn, event_type="test", source="", payload={"task_id": "t_1"})
        _rollback(conn)
        conn.close()

    def test_publish_long_event_type_rejected(self):
        conn = _fresh_db()
        _begin(conn)
        with pytest.raises(ValueError, match="exceeds"):
            publish(
                conn,
                event_type="x" * (_MAX_FIELD_LEN + 1),
                source="test",
                payload={"task_id": "t_1"},
            )
        _rollback(conn)
        conn.close()

    def test_idempotent_publish_same_key(self):
        conn = _fresh_db()
        _begin(conn)
        evt1 = publish(
            conn,
            event_type="task.created",
            source="kanban",
            payload={"task_id": "t_1"},
            idempotency_key="create-t_1",
        )
        # Second publish with same key — returns original, even with different payload.
        evt2 = publish(
            conn,
            event_type="task.created",
            source="kanban",
            payload={"task_id": "t_2"},  # Different payload.
            idempotency_key="create-t_1",
        )
        _commit(conn)
        assert evt1.id == evt2.id
        assert evt1.payload == evt2.payload  # Original payload preserved.
        assert evt2.payload == {"task_id": "t_1"}
        conn.close()

    def test_idempotent_publish_different_source(self):
        """Same key, different source → two distinct events."""
        conn = _fresh_db()
        _begin(conn)
        evt1 = publish(
            conn,
            event_type="task.created",
            source="source_a",
            payload={"task_id": "t_1"},
            idempotency_key="create-t_1",
        )
        evt2 = publish(
            conn,
            event_type="task.created",
            source="source_b",
            payload={"task_id": "t_1"},
            idempotency_key="create-t_1",
        )
        _commit(conn)
        assert evt1.id != evt2.id
        conn.close()

    def test_publish_canonical_payload_serialization(self):
        """Payload is canonically serialized (sorted keys, no spaces)."""
        conn = _fresh_db()
        _begin(conn)
        evt = publish(
            conn,
            event_type="test",
            source="test",
            payload={"to": "running", "from": "todo", "task_id": "t_1"},
        )
        _commit(conn)
        raw = conn.execute(
            "SELECT payload FROM hermes_events WHERE id = ?", (evt.id,)
        ).fetchone()[0]
        assert raw == '{"from":"todo","task_id":"t_1","to":"running"}'
        conn.close()

    def test_publish_events_ordered_by_id(self):
        """Multiple publishes ordered by monotonically increasing id."""
        conn = _fresh_db()
        _begin(conn)
        ids = []
        for i in range(10):
            evt = publish(
                conn,
                event_type="test",
                source="test",
                payload={"task_id": f"t_{i}"},
            )
            ids.append(evt.id)
        _commit(conn)
        assert ids == sorted(ids)
        assert len(set(ids)) == 10
        conn.close()

    def test_publish_delayed_created_at_still_ordered_by_id(self):
        """Even if created_at is not monotonic, ordering is by id."""
        conn = _fresh_db()
        _begin(conn)
        # Publish with manipulated created_at (future).
        evt1 = publish(
            conn,
            event_type="test",
            source="test",
            payload={"task_id": "t_1"},
        )
        # Manually backdate the second event's created_at.
        evt2 = publish(
            conn,
            event_type="test",
            source="test",
            payload={"task_id": "t_2"},
        )
        conn.execute(
            "UPDATE hermes_events SET created_at = ? WHERE id = ?",
            (evt1.created_at - 100, evt2.id),
        )
        _commit(conn)
        assert evt2.id > evt1.id  # id ordering independent of created_at.
        conn.close()


# ---------------------------------------------------------------------------
# Payload validation tests
# ---------------------------------------------------------------------------

class TestPayloadValidation:
    """Fail-closed payload allowlist with nested forbidden values."""

    def test_allowed_keys_accepted(self):
        result = _validate_payload({"task_id": "t_1", "from": "todo", "to": "running"})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["task_id"] == "t_1"

    def test_forbidden_key_content_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"content": "some text"})

    def test_forbidden_key_body_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"body": "text"})

    def test_forbidden_key_text_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"text": "data"})

    def test_forbidden_key_message_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"message": "hello"})

    def test_forbidden_key_patient_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"patient": "John"})

    def test_forbidden_key_contact_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"contact": "info"})

    def test_forbidden_key_name_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"name": "Alice"})

    def test_forbidden_key_email_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"email": "a@b.com"})

    def test_forbidden_key_phone_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"phone": "555"})

    def test_forbidden_key_address_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"address": "123 St"})

    def test_forbidden_key_token_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"token": "abc"})

    def test_forbidden_key_credentials_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"credentials": "xyz"})

    def test_forbidden_key_auth_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"auth": "bearer"})

    def test_forbidden_key_traceback_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"traceback": "..."})

    def test_forbidden_key_raw_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"raw": "data"})

    def test_forbidden_key_metadata_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"metadata": {}})

    def test_nested_forbidden_key_rejected(self):
        """Forbidden keys rejected at any nesting depth."""
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"task_id": "t_1", "kind": {"patient": "data"}})

    def test_nested_forbidden_token_rejected(self):
        with pytest.raises(ValueError, match="forbidden"):
            _validate_payload({"task_id": "t_1", "kind": {"token": "secret"}})

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="allowlist"):
            _validate_payload({"unknown_field": "value"})

    def test_non_string_key_rejected(self):
        with pytest.raises(ValueError, match="string"):
            _validate_payload({123: "value"})  # type: ignore[dict-item]

    def test_non_mapping_rejected(self):
        with pytest.raises(ValueError, match="Mapping"):
            _validate_payload("string")  # type: ignore[arg-type]

    def test_empty_payload_accepted(self):
        result = _validate_payload({})
        assert result == "{}"

    def test_nested_allowed_keys_accepted(self):
        result = _validate_payload({
            "task_id": "t_1",
            "kind": {"status": "blocked", "reason_code": "dependency"},
        })
        parsed = json.loads(result)
        assert parsed["kind"]["status"] == "blocked"

    def test_list_values_accepted(self):
        result = _validate_payload({"task_id": "t_1", "count": [1, 2, 3]})
        parsed = json.loads(result)
        assert parsed["count"] == [1, 2, 3]

    def test_unsupported_type_in_value_rejected(self):
        with pytest.raises(ValueError, match="unsupported type"):
            _validate_payload({"task_id": set()})  # type: ignore[dict-item]

    def test_oversized_payload_rejected(self):
        # Build a payload that exceeds the 8192-byte limit.
        # Use many keys each with max-length string values.
        big_payload = {k: "v" * 250 for k in list(_ALLOWED_PAYLOAD_KEYS)}
        with pytest.raises(ValueError, match="exceeds"):
            _validate_payload(big_payload)

    def test_forbidden_markers_absent_from_db(self):
        """After publish, forbidden markers must not appear in raw DB."""
        conn = _fresh_db()
        _begin(conn)
        evt = publish(
            conn,
            event_type="task.status_change",
            source="kanban",
            payload={"task_id": "t_1", "from": "todo", "to": "running"},
        )
        _commit(conn)
        raw = conn.execute(
            "SELECT payload FROM hermes_events WHERE id = ?", (evt.id,)
        ).fetchone()[0]
        for forbidden in _FORBIDDEN_PAYLOAD_KEYS:
            assert forbidden not in raw, f"forbidden key '{forbidden}' found in raw DB"
        conn.close()

    def test_forbidden_markers_absent_from_error_messages(self):
        """Error messages must not leak the forbidden value."""
        try:
            _validate_payload({"patient": "John Doe"})
        except ValueError as exc:
            # Error message should say the key is forbidden, but not leak the value.
            msg = str(exc)
            assert "John Doe" not in msg
            assert "patient" in msg


# ---------------------------------------------------------------------------
# Subscribe tests
# ---------------------------------------------------------------------------

class TestSubscribe:
    """subscribe() contract."""

    def test_basic_subscribe_latest(self):
        conn = _fresh_db()
        # Publish two events first.
        _begin(conn)
        publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        publish(conn, event_type="test", source="s", payload={"task_id": "t_2"})
        _commit(conn)

        cursor = subscribe(conn, "sub_a", start="latest")
        assert cursor == 2  # MAX(id) = 2

        conn.close()

    def test_subscribe_beginning(self):
        conn = _fresh_db()
        _begin(conn)
        publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        _commit(conn)

        cursor = subscribe(conn, "sub_a", start="beginning")
        assert cursor == 0
        conn.close()

    def test_repeated_subscribe_does_not_reset_cursor(self):
        conn = _fresh_db()
        cursor1 = subscribe(conn, "sub_a", start="beginning")
        assert cursor1 == 0

        # Publish events and ack to advance cursor.
        _begin(conn)
        evt = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        ack(conn, "sub_a", evt.id)
        _commit(conn)

        # Re-subscribe — must NOT reset cursor.
        cursor2 = subscribe(conn, "sub_a", start="beginning")
        assert cursor2 == evt.id  # Advanced, not reset.
        conn.close()

    def test_subscribe_with_event_types(self):
        conn = _fresh_db()
        cursor = subscribe(
            conn, "sub_typed", event_types=["task.created", "task.completed"]
        )
        assert cursor == 0  # No events yet, latest = 0.

        # Verify interests stored.
        row = conn.execute(
            "SELECT event_types FROM hermes_event_cursors WHERE subscriber_id = ?",
            ("sub_typed",),
        ).fetchone()
        assert json.loads(row[0]) == ["task.completed", "task.created"]  # Sorted.
        conn.close()

    def test_subscribe_empty_event_types_rejected(self):
        conn = _fresh_db()
        with pytest.raises(ValueError, match="non-empty"):
            subscribe(conn, "sub_bad", event_types=[])
        conn.close()

    def test_subscribe_empty_subscriber_id_rejected(self):
        conn = _fresh_db()
        with pytest.raises(ValueError, match="non-empty"):
            subscribe(conn, "")
        conn.close()


# ---------------------------------------------------------------------------
# Poll tests
# ---------------------------------------------------------------------------

class TestPoll:
    """poll() contract."""

    def test_poll_returns_matching_events(self):
        conn = _fresh_db()
        subscribe(conn, "sub_all", start="beginning")

        _begin(conn)
        publish(conn, event_type="a", source="s", payload={"task_id": "t_1"})
        publish(conn, event_type="b", source="s", payload={"task_id": "t_2"})
        publish(conn, event_type="a", source="s", payload={"task_id": "t_3"})
        _commit(conn)

        events = poll(conn, "sub_all")
        assert len(events) == 3
        assert all(isinstance(e, Event) for e in events)
        assert [e.event_type for e in events] == ["a", "b", "a"]
        conn.close()

    def test_poll_with_type_filter(self):
        conn = _fresh_db()
        subscribe(conn, "sub_a", event_types=["a"], start="beginning")

        _begin(conn)
        publish(conn, event_type="a", source="s", payload={"task_id": "t_1"})
        publish(conn, event_type="b", source="s", payload={"task_id": "t_2"})
        publish(conn, event_type="a", source="s", payload={"task_id": "t_3"})
        _commit(conn)

        events = poll(conn, "sub_a")
        assert len(events) == 2
        assert all(e.event_type == "a" for e in events)
        conn.close()

    def test_poll_bounded_by_limit(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        for i in range(10):
            publish(conn, event_type="test", source="s", payload={"task_id": f"t_{i}"})
        _commit(conn)

        events = poll(conn, "sub", limit=3)
        assert len(events) == 3
        conn.close()

    def test_poll_limit_bounds(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")
        with pytest.raises(ValueError, match="between 1 and 1000"):
            poll(conn, "sub", limit=0)
        with pytest.raises(ValueError, match="between 1 and 1000"):
            poll(conn, "sub", limit=1001)
        conn.close()

    def test_poll_unregistered_subscriber_raises(self):
        conn = _fresh_db()
        with pytest.raises(ValueError, match="not registered"):
            poll(conn, "nonexistent")
        conn.close()

    def test_poll_does_not_advance_cursor(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        _commit(conn)

        events1 = poll(conn, "sub")
        events2 = poll(conn, "sub")
        assert len(events1) == len(events2) == 1
        assert events1[0].id == events2[0].id
        conn.close()

    def test_poll_ascending_id_order(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        for i in range(5):
            publish(conn, event_type="test", source="s", payload={"task_id": f"t_{i}"})
        _commit(conn)

        events = poll(conn, "sub")
        ids = [e.id for e in events]
        assert ids == sorted(ids)
        conn.close()

    def test_poll_uses_registered_interests(self):
        """poll() uses the interests from subscribe(), not a separate filter."""
        conn = _fresh_db()
        subscribe(conn, "sub_typed", event_types=["type_a"], start="beginning")

        _begin(conn)
        publish(conn, event_type="type_a", source="s", payload={"task_id": "t_1"})
        publish(conn, event_type="type_b", source="s", payload={"task_id": "t_2"})
        _commit(conn)

        events = poll(conn, "sub_typed")
        assert len(events) == 1
        assert events[0].event_type == "type_a"
        conn.close()


# ---------------------------------------------------------------------------
# ACK tests
# ---------------------------------------------------------------------------

class TestAck:
    """ack() contract."""

    def test_basic_ack_advances_cursor(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        evt = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        ack(conn, "sub", evt.id)
        _commit(conn)

        # Cursor advanced.
        row = conn.execute(
            "SELECT last_event_id FROM hermes_event_cursors "
            "WHERE subscriber_id = 'sub'"
        ).fetchone()
        assert row[0] == evt.id
        conn.close()

    def test_ack_requires_transaction(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")
        _begin(conn)
        evt = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        _commit(conn)

        with pytest.raises(RuntimeError, match="write transaction"):
            ack(conn, "sub", evt.id)
        conn.close()

    def test_ack_monotonic_reject_backward(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        evt1 = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        evt2 = publish(conn, event_type="test", source="s", payload={"task_id": "t_2"})
        ack(conn, "sub", evt1.id)
        ack(conn, "sub", evt2.id)
        _commit(conn)

        # Try to ack an older event — must fail.
        _begin(conn)
        with pytest.raises(ValueError, match="monotonically"):
            ack(conn, "sub", evt1.id)
        _rollback(conn)
        conn.close()

    def test_ack_duplicate_current_cursor_harmless(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        evt = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        ack(conn, "sub", evt.id)
        # Duplicate ACK of current cursor — no-op, no error.
        ack(conn, "sub", evt.id)
        _commit(conn)
        conn.close()

    def test_ack_nonexistent_event_rejected(self):
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        with pytest.raises(ValueError, match="does not exist"):
            ack(conn, "sub", 99999)
        _rollback(conn)
        conn.close()

    def test_ack_gap_skip_rejected(self):
        """Cannot skip the first unacked matching event."""
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        evt1 = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        evt2 = publish(conn, event_type="test", source="s", payload={"task_id": "t_2"})
        # Try to ack evt2 without acking evt1 — must fail.
        with pytest.raises(ValueError, match="first unacked"):
            ack(conn, "sub", evt2.id)
        _rollback(conn)
        conn.close()

    def test_ack_gap_skip_with_type_filter(self):
        """With typed interests, ack must still respect matching-event order."""
        conn = _fresh_db()
        subscribe(conn, "sub_a", event_types=["type_a"], start="beginning")

        _begin(conn)
        evt_a1 = publish(conn, event_type="type_a", source="s", payload={"task_id": "t_1"})
        _evt_b = publish(conn, event_type="type_b", source="s", payload={"task_id": "t_2"})
        evt_a2 = publish(conn, event_type="type_a", source="s", payload={"task_id": "t_3"})

        # Ack first matching event (type_a) — OK.
        ack(conn, "sub_a", evt_a1.id)

        # Try to skip to third matching event without acking second — WAIT,
        # evt_a2 IS the next matching one after acking evt_a1.
        # So acking evt_a2 should succeed.
        ack(conn, "sub_a", evt_a2.id)
        _commit(conn)
        conn.close()

    def test_ack_unregistered_subscriber_raises(self):
        conn = _fresh_db()
        _begin(conn)
        with pytest.raises(ValueError, match="not registered"):
            ack(conn, "nonexistent", 1)
        _rollback(conn)
        conn.close()


# ---------------------------------------------------------------------------
# Independent subscriber tests
# ---------------------------------------------------------------------------

class TestIndependentSubscribers:
    """Two subscribers receive same event and ACK independently."""

    def test_independent_cursors(self):
        conn = _fresh_db()
        subscribe(conn, "sub_a", start="beginning")
        subscribe(conn, "sub_b", start="beginning")

        _begin(conn)
        evt = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        _commit(conn)

        # Both see the event.
        events_a = poll(conn, "sub_a")
        events_b = poll(conn, "sub_b")
        assert len(events_a) == 1
        assert len(events_b) == 1
        assert events_a[0].id == events_b[0].id

        # A acks, B doesn't.
        _begin(conn)
        ack(conn, "sub_a", evt.id)
        _commit(conn)

        # A no longer sees it; B still does.
        events_a = poll(conn, "sub_a")
        events_b = poll(conn, "sub_b")
        assert len(events_a) == 0
        assert len(events_b) == 1
        conn.close()

    def test_subscribers_never_steal_cursors(self):
        """Two distinct subscribers never interfere with each other's cursor."""
        conn = _fresh_db()
        subscribe(conn, "sub_a", start="beginning")
        subscribe(conn, "sub_b", start="beginning")

        _begin(conn)
        evt1 = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        evt2 = publish(conn, event_type="test", source="s", payload={"task_id": "t_2"})
        _commit(conn)

        # A acks both.
        _begin(conn)
        ack(conn, "sub_a", evt1.id)
        ack(conn, "sub_a", evt2.id)
        _commit(conn)

        # B's cursor unchanged.
        row = conn.execute(
            "SELECT last_event_id FROM hermes_event_cursors "
            "WHERE subscriber_id = 'sub_b'"
        ).fetchone()
        assert row[0] == 0
        conn.close()


# ---------------------------------------------------------------------------
# Crash / rollback semantics
# ---------------------------------------------------------------------------

class TestCrashRollback:
    """Crash-before-ACK redelivers; rollback leaves no traces."""

    def test_crash_before_ack_redelivers(self):
        """If the transaction rolls back before ack commits, the event re-appears."""
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        evt = publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        _commit(conn)

        # Start a transaction, ack, then rollback (simulating crash).
        _begin(conn)
        ack(conn, "sub", evt.id)
        _rollback(conn)

        # Event should re-appear on poll.
        events = poll(conn, "sub")
        assert len(events) == 1
        assert events[0].id == evt.id
        conn.close()

    def test_outer_transaction_rollback_no_event_no_cursor(self):
        """Rollback of outer transaction leaves neither event nor cursor ACK."""
        conn = _fresh_db()
        subscribe(conn, "sub", start="beginning")

        _begin(conn)
        publish(conn, event_type="test", source="s", payload={"task_id": "t_1"})
        _rollback(conn)

        # No events in DB.
        count = conn.execute("SELECT COUNT(*) FROM hermes_events").fetchone()[0]
        assert count == 0

        # Cursor unchanged.
        events = poll(conn, "sub")
        assert len(events) == 0
        conn.close()

    def test_persisted_cursor_after_close_reconnect(self):
        """Cursor persisted across close/reconnect."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()

        # Session 1: publish and ack.
        conn1 = sqlite3.connect(tmp.name, autocommit=True)
        conn1.row_factory = sqlite3.Row
        migrate_event_bus(conn1)
        subscribe(conn1, "sub", start="beginning")

        conn1.execute("BEGIN IMMEDIATE")
        evt = publish(conn1, event_type="test", source="s", payload={"task_id": "t_1"})
        ack(conn1, "sub", evt.id)
        conn1.execute("COMMIT")
        conn1.close()

        # Session 2: reconnect, verify cursor.
        conn2 = sqlite3.connect(tmp.name, autocommit=True)
        conn2.row_factory = sqlite3.Row
        migrate_event_bus(conn2)

        events = poll(conn2, "sub")
        assert len(events) == 0  # Already acked.

        # Publish new event — should be visible.
        conn2.execute("BEGIN IMMEDIATE")
        evt2 = publish(conn2, event_type="test", source="s", payload={"task_id": "t_2"})
        conn2.execute("COMMIT")

        events = poll(conn2, "sub")
        assert len(events) == 1
        assert events[0].id == evt2.id
        conn2.close()
        os.unlink(tmp.name)

    def test_replay_in_bounded_batches(self):
        """Events replayed in bounded batches after reconnect."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()

        conn = sqlite3.connect(tmp.name, autocommit=True)
        conn.row_factory = sqlite3.Row
        migrate_event_bus(conn)
        subscribe(conn, "sub", start="beginning")

        # Publish 10 events.
        conn.execute("BEGIN IMMEDIATE")
        for i in range(10):
            publish(conn, event_type="test", source="s", payload={"task_id": f"t_{i}"})
        conn.execute("COMMIT")
        conn.close()

        # Reconnect and poll in batches of 3.
        conn2 = sqlite3.connect(tmp.name, autocommit=True)
        conn2.row_factory = sqlite3.Row
        migrate_event_bus(conn2)

        all_events: list[Event] = []
        while True:
            batch = poll(conn2, "sub", limit=3)
            if not batch:
                break
            all_events.extend(batch)
            conn2.execute("BEGIN IMMEDIATE")
            for evt in batch:
                ack(conn2, "sub", evt.id)
            conn2.execute("COMMIT")

        assert len(all_events) == 10
        conn2.close()
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Concurrent WAL tests
# ---------------------------------------------------------------------------

class TestConcurrentWAL:
    """True concurrent writer and reader processes on same WAL DB."""

    def test_concurrent_writer_reader(self):
        """Separate processes can write and read concurrently via WAL."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

        # Set up DB in parent.
        conn = sqlite3.connect(db_path, autocommit=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        migrate_event_bus(conn)
        subscribe(conn, "reader_sub", start="beginning")

        # Write events in parent.
        conn.execute("BEGIN IMMEDIATE")
        for i in range(5):
            publish(conn, event_type="test", source="parent", payload={"task_id": f"t_{i}"})
        conn.execute("COMMIT")
        conn.close()

        # Spawn a child process that writes more events.
        child_script = f"""
import sqlite3, sys, json, time
sys.path.insert(0, {str(_REPO)!r})
from hermes_cli.event_bus import migrate_event_bus, publish

conn = sqlite3.connect({db_path!r}, autocommit=True)
conn.row_factory = sqlite3.Row
conn.execute("PRAGMA journal_mode=WAL")
migrate_event_bus(conn)

conn.execute("BEGIN IMMEDIATE")
for i in range(5):
    publish(conn, event_type="test", source="child", payload={{"task_id": f"t_child_{{i}}"}})
conn.execute("COMMIT")
conn.close()
print("child_done")
"""
        result = subprocess.run(
            [sys.executable, "-c", child_script],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Child failed: {result.stderr}"
        assert "child_done" in result.stdout

        # Read back in parent — should see all 10 events.
        conn2 = sqlite3.connect(db_path, autocommit=True)
        conn2.row_factory = sqlite3.Row
        migrate_event_bus(conn2)

        events = poll(conn2, "reader_sub")
        assert len(events) == 10
        # Verify sources.
        sources = {e.source for e in events}
        assert sources == {"parent", "child"}
        conn2.close()
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Integration with kanban_db_connect
# ---------------------------------------------------------------------------

class TestKanbanDbConnectIntegration:
    """Verify migration runs via kanban_db_connect.connect."""

    def test_connect_creates_event_bus_tables(self):
        """kanban_db_connect.connect auto-creates event bus tables."""
        tmp = tempfile.mkdtemp()
        db_path = Path(tmp) / "test_board" / "kanban.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Force fresh init by setting env.
        old_env = os.environ.get("HERMES_KANBAN_DB")
        os.environ["HERMES_KANBAN_DB"] = str(db_path)
        try:
            from hermes_cli.kanban_db_connect import connect, _INITIALIZED_PATHS
            # Clear cache so it runs init.
            _INITIALIZED_PATHS.discard(str(db_path.resolve()))
            conn = connect(db_path)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "hermes_events" in tables
            assert "hermes_event_cursors" in tables
            assert "hermes_event_schema_version" in tables
            # Existing kanban tables also present.
            assert "tasks" in tables
            assert "task_events" in tables
            conn.close()
        finally:
            if old_env is not None:
                os.environ["HERMES_KANBAN_DB"] = old_env
            else:
                os.environ.pop("HERMES_KANBAN_DB", None)
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def test_existing_kanban_init_still_works(self):
        """init_db still works and event bus tables are created alongside."""
        tmp = tempfile.mkdtemp()
        db_path = Path(tmp) / "board2" / "kanban.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        old_env = os.environ.get("HERMES_KANBAN_DB")
        os.environ["HERMES_KANBAN_DB"] = str(db_path)
        try:
            from hermes_cli.kanban_db_connect import init_db, _INITIALIZED_PATHS
            _INITIALIZED_PATHS.discard(str(db_path.resolve()))
            path = init_db(db_path)
            assert path == db_path

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "hermes_events" in tables
            assert "tasks" in tables
            conn.close()
        finally:
            if old_env is not None:
                os.environ["HERMES_KANBAN_DB"] = old_env
            else:
                os.environ.pop("HERMES_KANBAN_DB", None)
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

# P0a Event Bus — SQLite Substrate with Durable Per-Subscriber ACK

Dark, standalone event substrate for the Hermes kanban system.
No runtime activation, no customer action, no router or publisher wiring.

## Module

`hermes_cli/event_bus.py` — sole owner, importable by later P0a/P0b modules.

## Public API

```python
from hermes_cli.event_bus import publish, subscribe, poll, ack, Event

# Publish (requires active write transaction)
with write_txn(conn):
    evt = publish(conn, event_type="task.status_change", source="kanban",
                  payload={"task_id": "t_1", "from": "todo", "to": "running"})

# Subscribe (idempotent; interests immutable after first registration)
cursor = subscribe(conn, "my_subscriber",
                   event_types=["task.status_change"],
                   start="latest")  # or "beginning"

# Poll (bounded, does NOT advance cursor)
events = poll(conn, "my_subscriber", limit=100)

# Ack (requires active write transaction; monotonic, first-matching-only)
with write_txn(conn):
    ack(conn, "my_subscriber", events[0].id)
```

## Schema

Independent `hermes_event_schema_version` table (singleton integer).
Version 1 creates:

- `hermes_events` — immutable event log (id, event_type, source, payload, created_at, idempotency_key)
- `hermes_event_cursors` — per-subscriber durable cursor with immutable interests
- Partial unique index on `(source, idempotency_key)` for idempotent publish

Migration runs idempotently via `kanban_db_connect.connect` after the existing
kanban schema and column migrations. Independent of `task_events` and existing
notifier — both remain untouched.

## Payload Allowlist (Fail-Closed)

Only keys in `_ALLOWED_PAYLOAD_KEYS` are accepted at any nesting depth.
Keys in `_FORBIDDEN_PAYLOAD_KEYS` (PII, credentials, verbose) are hard-blocked.
Unknown keys are rejected. No free-text, no raw webhook body, no patient data.

## Semantics

- **At-least-once delivery**: two processes on the same subscriber may both
  observe one event; downstream must use event-id idempotency.
- **Monotonic ACK**: cursor advances forward only; gap-skipping rejected.
- **Crash safety**: rollback before ACK means redelivery; rollback of the
  outer publish transaction means no event was stored.
- **Immutable interests**: subscribe-time event_types cannot be changed;
  create a new subscriber for different interests.
- **Ordering by id, NOT created_at**.

## Tests

```bash
scripts/run_tests.sh tests/hermes_cli/test_event_bus.py
```

77 tests covering: migration (fresh, legacy, idempotent, future-version refusal),
publish (basic, idempotent, canonical serialization, ordering), payload validation
(all forbidden keys, nested, allowlist, oversized), subscribe (latest, beginning,
re-registration), poll (type filter, bounded, ordering), ack (monotonic, gap-skip,
crash-before-ack, rollback), independent subscribers, concurrent WAL processes,
and kanban_db_connect integration.

## Benchmark

```bash
python3 scripts/benchmark_event_bus.py [--iterations N] [--poll-interval SECS]
```

Reports p50/p95/p99 of publish-commit to poll+ack-commit latency.

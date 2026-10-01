# Goal Metrics (P0b): Provenance-Backed Board Goals, Measurements and Gap Events

> **Status:** Dark (no live publishers, no router activation, no live targets)
> **Owner:** `hermes_cli/goal_metrics.py`
> **Phase:** P0b — builds on the P0a event bus substrate

## Overview

The goal-metrics system adds structured, provenance-backed goal tracking to the
Hermes Kanban board. Goals define measurable targets with explicit approval
evidence. Measurements are append-only, deduplicated, and validated. Gap
evaluation uses signed-gap arithmetic with hysteresis edge detection to emit
exactly-once crossing and recovery events through the P0a event bus.

## Architecture

```
                     ┌──────────────────┐
                     │  Board SQLite DB │
                     │  (same file)     │
                     ├──────────────────┤
                     │ goals            │  Goal definitions (draft/active/paused)
                     │ goal_measurements│  Append-only provenance records
                     │ goal_schema_ver  │  Independent version tracking
                     │ hermes_events    │  P0a event bus (shared)
                     │ hermes_event_*   │  Subscriber cursors (shared)
                     └──────────────────┘
```

### Schema Migration

The goal-metrics schema uses its own `goal_schema_version` table (singleton
integer), independent of `PRAGMA user_version` (core kanban) and
`hermes_event_schema_version` (P0a event bus). Goal schema migration is
**not** run by `kanban_db_connect.connect` or `init_db`. It requires an
explicit `ensure_goal_schema(conn)` call on the intended board connection,
only after that board's migration and activation have been authorized. The
canonical server board remains unmigrated during this dark-code release;
private disposable TEST boards opt in explicitly for the witness. Goal CRUD
fails closed with `GoalSchemaNotInitializedError` before that opt-in.

- **Future schema**: If a newer schema version is detected, `RuntimeError` is
  raised (fail-closed; the binary is too old).
- **Idempotent**: Re-running migration on an already-current DB is a no-op.
- **Fresh + legacy**: Works on both new DBs and existing boards.
- **Version 2**: Adds nullable `goal_measurements.metric_version`; historical
  readings stay NULL (unknown), never backfilled from the current goal.
- **Does not alter**: `task_events`, `tasks`, or the notifier.

## Goals

### Schema

| Column | Type | Constraints |
|--------|------|-------------|
| id | TEXT PK | Opaque UUID |
| parent_id | TEXT FK | Optional hierarchy |
| metric_id | TEXT | Machine-readable metric key |
| source_id | TEXT | Where measurements come from |
| direction | TEXT | `higher_is_better` or `lower_is_better` |
| target_value | REAL | Nullable (no gap eval without target) |
| tolerance | REAL | Non-negative finite |
| max_age_seconds | INTEGER | Positive; staleness cutoff |
| state | TEXT | `draft`, `active`, `paused` |
| title | TEXT | Bounded internal text (≤512 chars) |
| approval_evidence_ref | TEXT | Required for activation |
| approval_timestamp | REAL | Required for activation |
| created_at | REAL | Unix timestamp |
| metric_version | INTEGER | Guards against metric changes |

### Lifecycle

1. **Draft**: Created via `create_goal()`. No measurements accepted.
2. **Active**: Requires explicit `activate_goal()` with Keion-approved evidence
   reference and timestamp. Only active goals accept measurements.
3. **Paused**: Via `pause_goal()`. Measurements rejected until re-activated.

**No seeded examples or automatic activation.** Architecture percentages (85%,
4h, etc.) are examples only and are never silently promoted to live targets.

## Measurements

### Schema

| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER PK | Autoincrement |
| goal_id | TEXT FK | References goals(id) |
| metric_id | TEXT | Must match goal's metric_id |
| value | REAL | Finite numeric |
| source_id | TEXT | Must match goal's source_id |
| source_timestamp | REAL | When evidence was valid |
| observed_at | REAL | When fetched |
| source_ref | TEXT | Opaque provenance reference |
| payload_hash | TEXT | Content hash (not raw data) |
| idempotency_key | TEXT | Unique per goal+source |
| edge_state | TEXT | `unknown`, `healthy`, `breached` |
| gap_value | REAL | Signed gap at time of measurement |
| created_at | REAL | Unix timestamp |
| metric_version | INTEGER NULL | Definition at observation; NULL is unproven legacy |

### Validation (Fail-Closed)

All invalid inputs are rejected — no measurements on snapshot failure:

- **Mismatched metric/source**: metric_id and source_id must match the goal's
- **Invalid timestamps**: Future timestamps (>5 min clock skew) rejected
- **Stale data**: age > goal.max_age_seconds rejected (explicit, not treated as
  zero)
- **Missing provenance**: At least one of source_ref or payload_hash required
- **Non-finite values**: NaN, ±Inf rejected
- **Non-monotonic**: source_timestamp must be ≥ previous valid measurement

### Idempotency

- **Same key replay**: Returns the prior immutable measurement and event IDs
  without re-evaluation
- **Conflicting key**: Same idempotency_key with different payload_hash fails
  closed (no silent overwrite)

### Dark P2 integration

On explicitly migrated disposable boards, each eligible reading (even without
a gap-edge transition) re-evaluates up to 50 awaiting outcomes in the caller's
write transaction. Verdict, safe `outcome.evaluated` event and comparable
proposal confidence adjustment commit together or all roll back. Missing or
unversioned baselines, stale/foreign/unversioned after readings and definition
drift cannot confirm. `evaluate_due_batch` is an explicit bounded safety-net
scan, not the primary driver. `propose_gap_once` consumes opted-in gap events
with caller-supplied inert candidates; neither activates the canonical board.

## Gap Evaluation

### Signed Gap Arithmetic

```
higher_is_better: gap = target_value - value
lower_is_better:  gap = value - target_value
```

- **Within tolerance**: gap ≤ tolerance → healthy (exact boundary is healthy)
- **Gap magnitude**: max(0, gap - tolerance)

### Edge-State Transitions

| Previous | Current | Events Emitted |
|----------|---------|----------------|
| unknown | healthy | None (no fabricated recovery) |
| unknown | breached | `goal.gap_detected` + `metric.threshold_crossed` |
| healthy | healthy | None |
| healthy | breached | `goal.gap_detected` + `metric.threshold_crossed` |
| breached | healthy | `goal.achieved` (recovery, re-arms) |
| breached | breached | None (no repeated crossing) |

After a recovery (`breached→healthy`), a subsequent breach emits events again
(re-armed).

### Event Payloads (Metadata Only)

Events are published through the P0a `event_bus.publish()` under the same
write transaction. Payload contains only:

```json
{
  "goal_id": "...",
  "metric_id": "...",
  "value": 80.0,
  "threshold": 95.0,
  "direction": "higher_is_better",
  "source_timestamp": 1727000000.0,
  "status": "breached"
}
```

No title, reason, message, or raw fields. Deterministic per-edge idempotency
keys prevent duplicate events on retry.

### Threshold Computation

- `higher_is_better`: threshold = target_value - tolerance
- `lower_is_better`: threshold = target_value + tolerance

## Read-Only Kanban Adapter

### `snapshot_board_counts(conn)`

Takes a read-only snapshot of board counts from the SAME board:

- `blocked_count`: `COUNT(*) FROM tasks WHERE status = 'blocked'`
- `non_archived_count`: `COUNT(*) FROM tasks WHERE status != 'archived'`

Matches `board_stats(conn)` counting logic. One consistent SQLite read
snapshot with captured `source_timestamp` and snapshot UUID evidence ID.
No writes to the source board during read.

### `record_board_snapshot(target_conn, snapshot, goal_id)`

Records snapshot counts as measurements against an active goal. Provenance
hash computed from snapshot evidence. If source and target board are the same
DB, reads + writes are coordinated under the caller's existing write
transaction.

## Testing

Run tests via:

```bash
scripts/run_tests.sh tests/hermes_cli/test_goal_metrics.py
```

Test coverage includes:

- Migration: fresh, idempotent, future-version, legacy board
- Goal CRUD: create, activate, pause, list, parent FK, validation
- Gap computation: both directions, tolerance boundaries, exact boundary
- Edge transitions: first healthy, first breach, recovery, re-arm, repeated
- Idempotency: same-key replay, conflicting key, event replay
- Event bus integration: payload validation, forbidden fields, subscribers
- Board snapshot: count reconciliation, empty board, same-DB recording
- Rollback atomicity: measurements + events rolled back together
- Legacy board coexistence: task_events not modified
- Replay after reopen: events survive close/reopen

## File Ownership

| File | Owner |
|------|-------|
| `hermes_cli/goal_metrics.py` | This module (P0b builder) |
| `hermes_cli/kanban_db_connect.py` | Migration hook only |
| `tests/hermes_cli/test_goal_metrics.py` | This module |
| `docs/goal_metrics.md` | This module |

No other builder edits these files.

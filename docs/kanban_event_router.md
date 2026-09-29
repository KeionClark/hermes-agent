# Dark opt-in Kanban event router

The router is not imported by gateway startup, daemon, or dispatcher. It never
spawns workers or sends messages. Use only after publisher/router revision and
privacy review. The built-in inbox stores subscriber ID, event ID, event type,
and enqueue timestamp; it stores no event payload. Future OODA consumers must
have their own gate and must not infer authorization from inbox presence.

For a disposable or specifically approved board, explicitly initialize a
subscriber before publishing an event (default `latest` intentionally ignores
all existing bus history and preexisting blocked tasks):

    python -m hermes_cli.kanban_event_router --board-path /path/to/test.db \
      --subscriber-id ooda-inbox --event-type task.blocked --iterations 1

Then publish events on that same board and invoke it again. Use `--iterations N
--interval SECONDS` for a finite poll window; `--max-events N` caps effects
per subscriber per tick at 1000. Startup without an explicit existing board,
subscriber ID and at least one event type fails. Re-registration never rewinds
the cursor or changes its interests; a mismatch fails closed. Programmatic
`run_once(conn, {id: Subscriber(frozenset(types), callback)})` supports multiple
streams and round-robin bounded processing. All callbacks are short,
synchronous, same-connection, same-DB operations executed in `write_txn`.
Callbacks must not commit, spawn a worker, call external APIs, or block. A
blocking callback holds the SQLite writer and therefore cannot be preempted by
this synchronous router: use strictly bounded callbacks only.

Failure effects roll back; only an enumerated error code persists in
`hermes_router_failures` (no exception message). Backoff is exponential and
capped; a retry holds its own stream's oldest matching event, not other
streams. After max attempts an atomic deadletter row and ACK advances that
stream. `hermes_router_inbox`'s `(subscriber_id,event_id)` key deduplicates
replays. Callback effect and ACK share one transaction; a crash before commit
leaves both absent, a crash after commit leaves both present. No guarantee is
possible for external callback effects, so they are prohibited.

`migrate_router(conn)` owns schema version `hermes_router_schema_version`,
independent of `hermes_event_schema_version`; it is idempotent and additive.
The router tables are created only by explicit router invocation, not on
ordinary board connections. Do not downgrade/drop tables on a live board.

Benchmark: `python -m scripts.benchmark_kanban_event_router --iterations 100
--interval 0.01`; disposable board, reports publish COMMIT to inbox readback
with a configured worst-phase poll delay, p50/p95/p99. This is a local latency
measurement, not a production throughput or gateway-activation claim.

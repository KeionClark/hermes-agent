"""Disposable-board publish-commit -> router inbox latency, explicit poll interval.

Usage: python scripts/benchmark_kanban_event_router.py --iterations 100 --interval 0.01
No live board is opened. The inbox row is read after every router tick.
"""
import argparse
import statistics
import tempfile
import time
from pathlib import Path

from hermes_cli import event_bus
from hermes_cli.kanban_db_connect import connect, write_txn
from hermes_cli.kanban_event_router import Subscriber, enqueue_inbox, run_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.01)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10000 or not 0 <= args.interval <= 10:
        parser.error("invalid bounded iterations/interval")
    with tempfile.TemporaryDirectory(prefix="router-bench-") as directory:
        conn = connect(Path(directory) / "board.db")
        registrations = {"bench": Subscriber(frozenset({"benchmark.tick"}), enqueue_inbox("bench"))}
        run_once(conn, registrations)
        samples = []
        for i in range(args.iterations):
            with write_txn(conn):
                event = event_bus.publish(conn, event_type="benchmark.tick", source="benchmark",
                                          payload={"count": i})
            start = time.perf_counter()
            # The event commits just after a tick; wait one poll period to
            # model worst-phase scheduling rather than reporting zero delay.
            time.sleep(args.interval)
            while True:
                run_once(conn, registrations, max_events=1)
                if conn.execute("SELECT 1 FROM hermes_router_inbox WHERE subscriber_id='bench' AND event_id=?",
                                (event.id,)).fetchone():
                    samples.append((time.perf_counter() - start) * 1000)
                    break
                time.sleep(args.interval)
        conn.close()
    samples.sort()
    def percentile(p):
        return samples[min(len(samples) - 1, max(0, int((len(samples) - 1) * p + 0.5)))]
    print(f"N={len(samples)} interval={args.interval}s p50={percentile(.50):.3f}ms "
          f"p95={percentile(.95):.3f}ms p99={percentile(.99):.3f}ms "
          f"mean={statistics.mean(samples):.3f}ms")


if __name__ == "__main__":
    main()

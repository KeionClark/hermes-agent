#!/usr/bin/env python3
"""
Benchmark: P0a event bus publish-commit-poll-ack latency.

Measures arrival-to-noop-routing latency on a temp board.
Reports p50, p95, p99 over N iterations with polling interval.

Usage:
    python3 scripts/benchmark_event_bus.py [--iterations N] [--poll-interval SECS]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure repo root importable.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from hermes_cli.event_bus import ack, migrate_event_bus, poll, publish, subscribe


def _percentile(data: list[float], p: float) -> float:
    """Simple percentile calculation."""
    data_sorted = sorted(data)
    idx = (p / 100) * (len(data_sorted) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(data_sorted):
        return data_sorted[lo]
    frac = idx - lo
    return data_sorted[lo] * (1 - frac) + data_sorted[hi] * frac


def run_benchmark(iterations: int = 1000, poll_interval: float = 0.0) -> None:
    """Run the publish-poll-ack benchmark."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name

    try:
        conn = sqlite3.connect(db_path, autocommit=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        migrate_event_bus(conn)

        subscribe(conn, "bench_sub", start="beginning")

        latencies_us: list[float] = []

        for i in range(iterations):
            # Publish
            t0 = time.perf_counter()
            conn.execute("BEGIN IMMEDIATE")
            evt = publish(
                conn,
                event_type="bench.event",
                source="benchmark",
                payload={"task_id": f"t_{i}", "status": "running"},
            )
            conn.execute("COMMIT")

            # Optional poll interval (simulates real polling loop).
            if poll_interval > 0:
                time.sleep(poll_interval)

            # Poll
            events = poll(conn, "bench_sub", limit=1)

            # Ack (noop routing callback is instant).
            if events:
                conn.execute("BEGIN IMMEDIATE")
                ack(conn, "bench_sub", events[0].id)
                conn.execute("COMMIT")

            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1_000_000)

        conn.close()

        # Get git SHA.
        sha = "unknown"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=str(_REPO),
            )
            if result.returncode == 0:
                sha = result.stdout.strip()
        except Exception:
            pass

        # Report.
        p50 = _percentile(latencies_us, 50)
        p95 = _percentile(latencies_us, 95)
        p99 = _percentile(latencies_us, 99)
        mean = statistics.mean(latencies_us)
        stdev = statistics.stdev(latencies_us) if len(latencies_us) > 1 else 0

        print("=" * 60)
        print("P0a Event Bus Benchmark")
        print("=" * 60)
        print(f"  Revision SHA: {sha}")
        print(f"  Iterations:   {iterations}")
        print(f"  Poll interval: {poll_interval}s")
        print(f"  Python:       {sys.version.split()[0]}")
        print(f"  Platform:     {sys.platform}")
        print()
        print("  publish-commit -> poll -> ack-commit latency (microseconds):")
        print(f"    p50:   {p50:>10.1f} us")
        print(f"    p95:   {p95:>10.1f} us")
        print(f"    p99:   {p99:>10.1f} us")
        print(f"    mean:  {mean:>10.1f} us")
        print(f"    stdev: {stdev:>10.1f} us")
        print(f"    min:   {min(latencies_us):>10.1f} us")
        print(f"    max:   {max(latencies_us):>10.1f} us")
        print("=" * 60)

    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass
        # Clean up WAL/SHM files.
        for suffix in ("-wal", "-shm"):
            try:
                os.unlink(db_path + suffix)
            except OSError:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0a event bus benchmark")
    parser.add_argument(
        "--iterations", "-n", type=int, default=1000,
        help="Number of publish-poll-ack cycles (default: 1000)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.0,
        help="Seconds to sleep between publish and poll (default: 0)",
    )
    args = parser.parse_args()
    run_benchmark(iterations=args.iterations, poll_interval=args.poll_interval)

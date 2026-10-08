"""Host-wide admission primitives for the opt-in Kanban provider dispatcher.

This module does not read credentials, switch accounts, or start workers. The
caller supplies a shared ledger path and verified runtime/auth observations.
"""
from __future__ import annotations

import math
import sqlite3
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

LANE_CAP = 2
TOTAL_CAP = 4
ORDINARY_STAGES = frozenset({"research", "build", "audit"})
STAGES = {
    "researcher": "research", "builder": "build", "quality-auditor": "audit",
    "qa2": "audit", "qa3": "audit", "qa4": "audit", "head-dev": "plan",
    "release-engineer": "release", "release2": "release",
    "customer-success": "protected", "patient-facing": "protected",
}


def provider_lane(provider: str) -> str:
    """Router's 'openai' denotes Codex subscription, not API billing."""
    return {"anthropic": "anthropic", "openai": "openai-codex",
            "openai-codex": "openai-codex"}.get(provider, "")


@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str
    effort: str | None = None


@dataclass(frozen=True)
class Allowance:
    remaining: float
    observed_at: float
    source: str
    unit: str


def load_router(path: Path) -> dict[str, tuple[Candidate, ...]]:
    """Read only routing data; never copy provider credentials into a ledger."""
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    routes = {}
    for stage, section in data.get("stages", {}).items():
        candidates = []
        for row in section.get("candidates", []):
            provider = provider_lane(row.get("provider", ""))
            model = row.get("model")
            if not provider or not isinstance(model, str) or not model.strip():
                raise ValueError(f"unsupported route in stage {stage}")
            candidates.append(Candidate(provider, model, row.get("effort")))
        routes[stage] = tuple(candidates)
    return routes


def order_candidates(
    candidates: Sequence[Candidate], *, stage: str, pinned_provider: str,
    built_by: str | None = None, allowances: Mapping[str, Allowance] | None = None,
    trusted_sources: frozenset[str] = frozenset(), now: float | None = None,
    max_allowance_age: float = 300,
) -> tuple[Candidate, ...]:
    """Only ordinary stages may cross provider; unknown stages are protected.

    Audit independence is fail-closed when known: no same-provider audit.
    Incomparable, stale, partial or untrusted allowances retain router order.
    """
    pin = provider_lane(pinned_provider)
    if not pin or (stage == "audit" and built_by and not provider_lane(built_by)):
        return ()
    allowed = tuple(c for c in candidates if (
        stage in ORDINARY_STAGES or c.provider == pin
    ) and not (stage == "audit" and built_by and c.provider == provider_lane(built_by)))
    readings = allowances or {}
    clock = time.time() if now is None else now
    values = [readings.get(c.provider) for c in allowed]
    valid = bool(values) and all(v is not None and v.source in trusted_sources
        and math.isfinite(v.remaining) and v.remaining >= 0
        and math.isfinite(v.observed_at) and 0 <= clock - v.observed_at <= max_allowance_age
        and bool(v.unit) for v in values)
    if valid and len({v.unit for v in values if v is not None}) == 1:
        return tuple(sorted(allowed, key=lambda c: readings[c.provider].remaining, reverse=True))
    return allowed


def memory_admits(available_bytes: int | None, minimum_bytes: int) -> bool:
    """Unknown/invalid free-memory measurements never admit a new worker."""
    return (type(available_bytes) is int and type(minimum_bytes) is int
            and minimum_bytes > 0 and available_bytes >= minimum_bytes)


def aged_priority(priority: int, created_at: float, now: float, *,
                  aging_seconds: int = 900, maximum_bonus: int = 20) -> int:
    """Bounded priority aging; it never changes or preempts running workers."""
    if aging_seconds <= 0 or maximum_bonus < 0:
        raise ValueError("invalid aging bounds")
    return priority + min(maximum_bonus, int(max(0, now - created_at) // aging_seconds))


class LaneLedger:
    """SQLite reservations shared by ALL dispatchers/boards on one host.

    Reservation and capacity checks run in one BEGIN IMMEDIATE transaction.
    No TTL evicts a live worker. The liveness resolver must identify a worker
    by its persisted process fingerprint/run identity, not by PID alone.
    Unknown liveness retains capacity (fail closed). Pending reservations use
    the same resolver so callers can recover the spawn-before-bind crash gap.
    """
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 30000")
        self.conn.execute("""CREATE TABLE IF NOT EXISTS reservations (
            token TEXT PRIMARY KEY, board TEXT NOT NULL, task TEXT NOT NULL,
            run INTEGER NOT NULL, requested_provider TEXT NOT NULL, requested_model TEXT NOT NULL,
            owner TEXT NOT NULL, worker TEXT, created_at REAL NOT NULL,
            observed_provider TEXT, observed_model TEXT,
            UNIQUE(board, task, run))""")

    def close(self) -> None:
        self.conn.close()

    def reserve(self, *, board: str, task: str, run: int,
                candidates: Sequence[Candidate], owner: str,
                alive: Callable[[Mapping], bool | None]) -> tuple[str, Candidate] | None:
        """Reserve the first available ordered lane; never overwrite an owner."""
        if not owner or not board or not task or type(run) is not int:
            raise ValueError("reservation requires board, task, run and owner")
        if any(c.provider not in {"anthropic", "openai-codex"} for c in candidates):
            raise ValueError("unsupported provider lane")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for row in self.conn.execute("SELECT * FROM reservations").fetchall():
                if alive(dict(row)) is False:
                    self.conn.execute("DELETE FROM reservations WHERE token = ?", (row["token"],))
            rows = self.conn.execute("SELECT * FROM reservations").fetchall()
            if any((r["board"], r["task"], r["run"]) == (board, task, run) for r in rows):
                return None
            if len(rows) >= TOTAL_CAP:
                return None
            for candidate in candidates:
                if sum(r["requested_provider"] == candidate.provider for r in rows) >= LANE_CAP:
                    continue
                token = uuid.uuid4().hex
                self.conn.execute("""INSERT INTO reservations
                    (token, board, task, run, requested_provider, requested_model, owner, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (token, board, task, run, candidate.provider, candidate.model, owner, time.time()))
                return token, candidate
            return None
        except BaseException:
            self.conn.rollback()
            raise
        finally:
            if self.conn.in_transaction:
                self.conn.commit()

    def bind_worker(self, token: str, owner: str, worker: str) -> bool:
        if not worker:
            raise ValueError("worker fingerprint required")
        return self.conn.execute("UPDATE reservations SET worker = ? WHERE token = ? AND owner = ?",
                                 (worker, token, owner)).rowcount == 1

    def observe(self, token: str, worker: str, *, provider: str, model: str) -> bool:
        """Store real runtime evidence separately, including a route mismatch.

        The caller must pass observations from the executing runtime, never
        config/argv. Observation cannot change the capacity-charged lane.
        """
        if not provider or not model or not worker:
            raise ValueError("runtime provider/model and worker identity required")
        return self.conn.execute("""UPDATE reservations SET observed_provider = ?, observed_model = ?
            WHERE token = ? AND worker = ?""", (provider, model, token, worker)).rowcount == 1

    def release(self, token: str, owner: str) -> bool:
        return self.conn.execute("DELETE FROM reservations WHERE token = ? AND owner = ?",
                                 (token, owner)).rowcount == 1

    def snapshot(self) -> list[dict]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM reservations")]

"""Dispatch spawn callable adaptation (legacy two-argument consumers)."""
from typing import Optional


def call_spawn_fn(spawn_fn, task, workspace: str, board: Optional[str]) -> Optional[int]:
    """Pass board only when supported; retain legacy spawn callable signatures."""
    import inspect
    try:
        sig = inspect.signature(spawn_fn)
        if "board" in sig.parameters:
            return spawn_fn(task, workspace, board=board)
        return spawn_fn(task, workspace)
    except (TypeError, ValueError):
        return spawn_fn(task, workspace)

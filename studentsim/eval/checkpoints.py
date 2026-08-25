"""Finding the adapter a per-student run produced.

ms-swift writes checkpoints as ``<output_dir>/v<N>-<timestamp>/checkpoint-<step>``,
and evaluation always reads the highest-step one. The search runs at any depth
because a checkpoint copied from another machine arrives with an extra
directory level.
"""

from __future__ import annotations

from pathlib import Path


def _step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def find_last_checkpoint(root: str | Path) -> Path | None:
    """Return the highest-step ``checkpoint-*`` directory under ``root``."""
    candidates = [p for p in Path(root).glob("**/checkpoint-*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=_step)[-1]

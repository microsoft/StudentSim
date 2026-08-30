"""Checking that the run will use the batch size the config declares.

``world_size`` is a declaration, not an instruction: ms-swift sizes the run
from the GPUs it can see. Declaring eight and launching on two silently
quarters the effective batch, and the run then trains to completion with
normal-looking logs on a different optimization trajectory than the config
describes. This module makes that mismatch stop the run instead.
"""

from __future__ import annotations

import os


def visible_gpu_count() -> int | None:
    """How many GPUs this process will train on, or ``None`` if unknown."""
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is not None:
        listed = [d for d in devices.split(",") if d.strip()]
        return len(listed)
    nproc = os.environ.get("NPROC_PER_NODE")
    if nproc:
        try:
            return int(nproc)
        except ValueError:
            return None
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.device_count() if torch.cuda.is_available() else None


def check_world_size(declared: int, *, effective_batch: int, per_device: int) -> None:
    """Raise when the run would train at a different effective batch.

    Passing a ``declared`` world size that does not match the visible GPUs
    means the accumulation steps in the config no longer multiply out to
    ``effective_batch``.
    """
    actual = visible_gpu_count()
    if actual is None or actual == declared:
        return
    accumulation = effective_batch // (per_device * declared)
    would_be = per_device * accumulation * actual
    raise RuntimeError(
        f"config declares world_size={declared} but {actual} GPU(s) are visible, "
        f"so this run would train at an effective batch of {would_be} instead of "
        f"{effective_batch}. Set world_size={actual} and "
        f"gradient_accumulation={effective_batch // (per_device * actual)} to keep "
        f"the effective batch, or make {declared} GPUs visible."
    )

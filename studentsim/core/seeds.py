"""The seeds every stage draws from, so two builds of the same thing agree.

Two of them, because shuffling a batch and drawing the instance pool are
separately reproducible: the trainer seed drives optimizer state and dataloader
order, and the data-sampler seed decides which records a domain's pool holds.
Both are imported by name wherever a seed is needed; nothing here passes a
numeric literal, so changing one changes it everywhere at once.

:data:`CROSS_SEED_RUNS` holds three pairs. Training all three and taking the
spread measures variation over the whole pipeline rather than over shuffling
alone, because both seeds move together.
"""

from __future__ import annotations

import os
import random
from typing import Final

TRAINER_SEED: Final[int] = 42
"""ms-swift trainer default; controls optimizer state and dataloader shuffling."""

DATA_SAMPLER_SEED: Final[int] = 65
"""Data sampler that constructs the per-domain training-instance pool."""


CROSS_SEED_RUNS: Final[tuple[tuple[int, int], ...]] = (
    (65, 65),
    (7, 7),
    (42, 42),
)
"""Three ``(data_sampler_seed, trainer_seed)`` pairs, for measuring spread.

No code path reads this. It is the set of pairs to pass to the training config,
one per run, when the question is how much a result moves between runs.

Averaging three runs means no single run's output equals the average. None of
the three is the default pair above, which leaves that pair available as an
independent check.
"""


def seed_everything(seed: int = TRAINER_SEED) -> None:
    """Seed Python ``random``, NumPy, and PyTorch (if installed) with one value.

    The function is intentionally tolerant of torch/numpy not being installed so it
    can be called from contexts that only depend on ``studentsim.core``.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

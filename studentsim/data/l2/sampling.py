"""How guidance records are chosen for training.

Every draw is seeded from the learner it belongs to, so a rebuild on the same
corpus lands on the same records. The learner identifier never leaves the
build: it seeds the draw and is then replaced by a positional name.
"""

from __future__ import annotations

import random
import zlib
from collections import defaultdict


def learner_rng(learner_id: str, seed: int) -> random.Random:
    """The draw for one learner, reproducible from their identifier."""
    return random.Random(seed ^ (zlib.crc32(learner_id.encode()) & 0xFFFFFFFF))


def guidance_budget(n_single: int, n_available: int, total: int, ratio: float) -> tuple[int, int]:
    """How many guided and unguided records one learner contributes.

    A learner with few essays can still have thousands of corrections, so the
    guided share is held to ``ratio`` of what the essays support instead of
    being allowed to fill the whole budget.
    """
    if ratio >= 1.0:
        return min(total, n_available), 0
    if ratio <= 0.0:
        return 0, min(total, n_single)
    ceiling = round(n_single * ratio / (1.0 - ratio))
    n_guided = min(round(total * ratio), n_available, ceiling)
    return n_guided, min(total - n_guided, n_single)


def select_by_style(
    candidates: list[dict],
    styles: dict[str, float],
    target: int,
    rng: random.Random,
) -> list[dict]:
    """Take ``target`` guidance records, spread across the tutor styles."""
    if not candidates or target <= 0:
        return []
    by_style: dict[str, list[dict]] = defaultdict(list)
    for record in candidates:
        by_style[record["style"]].append(record)

    present = list(by_style)
    weights = [styles.get(style, 1.0) for style in present]
    total_weight = sum(weights) or 1.0

    selected: list[dict] = []
    for style, weight in zip(present, weights):
        pool = list(by_style[style])
        rng.shuffle(pool)
        selected.extend(pool[: round(target * weight / total_weight)])
    rng.shuffle(selected)
    return selected[:target]


def select_balanced(
    pools: dict[str, list[dict]],
    styles: dict[str, float],
    target: int,
    seed: int,
) -> list[dict]:
    """Spread ``target`` guidance records evenly over learners and styles.

    Each correction is offered in every style, so first one style is drawn per
    correction, then each learner contributes the same number per style. That
    keeps a prolific learner from filling the pooled stage on their own.
    """
    n_learners = len(pools) or 1
    present = sorted({record["style"] for pool in pools.values() for record in pool})
    active = [style for style in present if styles.get(style, 1.0) > 0]
    total_weight = sum(styles.get(style, 1.0) for style in active) or 1.0
    per_cell = {
        style: max(1, round(target * styles.get(style, 1.0) / total_weight / n_learners))
        for style in active
    }

    selected: list[dict] = []
    for learner in sorted(pools):
        at_position: dict[str, list[dict]] = defaultdict(list)
        for record in pools[learner]:
            at_position[record["position"]].append(record)

        assign = random.Random(
            seed ^ (zlib.crc32(f"assign:{learner}".encode()) & 0xFFFFFFFF)
        )
        chosen_by_style: dict[str, list[dict]] = defaultdict(list)
        for position in sorted(at_position):
            offered = sorted({r["style"] for r in at_position[position] if r["style"] in active})
            if not offered:
                continue
            weights = [styles.get(style, 1.0) for style in offered]
            style = assign.choices(offered, weights=weights, k=1)[0]
            chosen_by_style[style].append(
                next(r for r in at_position[position] if r["style"] == style)
            )

        for style in active:
            pool = list(chosen_by_style.get(style, []))
            draw = random.Random(
                seed ^ (zlib.crc32(f"sample:{learner}:{style}".encode()) & 0xFFFFFFFF)
            )
            draw.shuffle(pool)
            selected.extend(pool[: per_cell[style]])

    random.Random(seed).shuffle(selected)
    return selected[:target]

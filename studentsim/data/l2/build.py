"""Build the L2 training and evaluation sets from an EFCAMDAT extract.

The learners, their split, and every record are derived from the corpus itself,
so two builds of the same extract give identical
record sets. A learner's identifier seeds their draws and is then replaced by a positional
name, so nothing in the output identifies anyone.

    python -m studentsim.data.build_l2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from studentsim.core.seeds import DATA_SAMPLER_SEED
from studentsim.data.l2.guidance import (
    build_point_tutor_turn,
    build_rule_tutor_turn,
    target_correction,
)
from studentsim.data.l2.profile import compute_profile
from studentsim.data.l2.prompts import build_user_message
from studentsim.data.l2.sampling import (
    guidance_budget,
    learner_rng,
    select_balanced,
    select_by_style,
)
from studentsim.data.l2.spans import extract_spans

#: Learners in each pool, the larger one a superset of the smaller.
STAGE1_LEARNERS = 200
STAGE2_LEARNERS = 15

#: The share of a learner's essays, oldest first, that may be trained on. The
#: rest is held out.
TRAIN_RATIO = 0.7

#: Share of training records that carry a tutor turn.
MULTITURN_RATIO = 0.2

#: Held-out guidance records per tutor style, per learner.
TEST_PER_STYLE = 20

#: The budget the specialization stage draws from, across all its learners.
SPECIALIZE_BUDGET = 870

#: The two tutor styles, how each renders its turn, and their equal weight.
STYLES = {"point": build_point_tutor_turn, "rule": build_rule_tutor_turn}
STYLE_WEIGHTS = {"point": 1.0, "rule": 1.0}

#: Appended to the tutor turn in the pooled stage only.
POOLED_TUTOR_SUFFIX = (
    "Reply with only the corrected text — no explanation, no rewriting the rest."
)

SEED = DATA_SAMPLER_SEED


@dataclass(slots=True)
class Essay:
    writing_id: int
    level: int
    unit: int
    topic_id: int
    topic: str
    grade: int
    nationality: str
    original: str
    spans: list


def _as_int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def rank_learners(csv_path: Path) -> list[str]:
    """Most prolific learners first, breaking ties by the level range covered.

    Selecting the pools this way keeps them reproducible from the corpus while
    giving every selected learner enough essays to both train on and hold out.
    Only the counts are held in memory, so the whole corpus never is.
    """
    counts: dict[str, int] = defaultdict(int)
    lowest: dict[str, int] = {}
    highest: dict[str, int] = {}
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            learner = row["learnerID"]
            counts[learner] += 1
            level = _as_int(row.get("level"))
            if level > 0:
                lowest[learner] = min(lowest.get(learner, level), level)
                highest[learner] = max(highest.get(learner, level), level)

    def key(learner: str) -> tuple[int, int]:
        return (-counts[learner], -(highest.get(learner, 0) - lowest.get(learner, 0)))

    return sorted(counts, key=key)


def read_corpus(csv_path: Path, keep: set[str]) -> dict[str, list[Essay]]:
    """Group the selected learners' essays, each learner's in time order."""
    by_learner: dict[str, list[Essay]] = defaultdict(list)
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            learner = row["learnerID"]
            if learner not in keep:
                continue
            by_learner[learner].append(
                Essay(
                    writing_id=_as_int(row.get("writingID")),
                    level=_as_int(row.get("level")),
                    unit=_as_int(row.get("unit")),
                    topic_id=_as_int(row.get("topicID")),
                    topic=(row.get("topic") or "").strip(),
                    grade=_as_int(row.get("grade")),
                    nationality=row.get("nationality", ""),
                    original=row.get("original") or "",
                    spans=extract_spans(row.get("text") or ""),
                )
            )
    for essays in by_learner.values():
        essays.sort(key=lambda e: e.writing_id)
    return dict(by_learner)


def split_chronologically(essays: list[Essay]) -> dict[str, list[Essay]]:
    """Oldest essays train, newest are held out."""
    n_train = int(len(essays) * TRAIN_RATIO)
    return {"train": essays[:n_train], "test": essays[n_train:]}


def _as_dict(essay: Essay) -> dict:
    return {
        "level": essay.level,
        "unit": essay.unit,
        "topic_id": essay.topic_id,
        "topic": essay.topic,
        "grade": essay.grade,
        "original": essay.original,
        "nationality": essay.nationality,
        "spans": essay.spans,
    }


def single_turn_record(name: str, history: list[Essay], index: int) -> dict:
    """One essay the learner wrote, with the prompt that asks for it."""
    earlier = [_as_dict(e) for e in history[:index]]
    prompt = build_user_message(
        learner_name=name,
        profile=compute_profile(earlier),
        current=_as_dict(history[index]),
        earlier=earlier,
        detailed_errors=False,
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": history[index].original},
        ]
    }


def multi_turn_records(name: str, history: list[Essay], index: int, style: str) -> list[dict]:
    """One record per correction in this essay, in the given tutor style."""
    earlier = [_as_dict(e) for e in history[:index]]
    prompt = build_user_message(
        learner_name=name,
        profile=compute_profile(earlier),
        current=_as_dict(history[index]),
        earlier=earlier,
        detailed_errors=True,
    )
    render = STYLES[style]
    return [
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": history[index].original},
                {"role": "user", "content": render(span)},
                {"role": "assistant", "content": target_correction(span)},
            ],
            "style": style,
            "position": f"{history[index].writing_id}_{span.symbol}",
        }
        for span in history[index].spans
    ]


def _write(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(records):,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m studentsim.data.l2.build")
    parser.add_argument(
        "--raw", type=Path, default=Path("data/l2/raw/ef_POStagged_original_corrected.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("data/l2"))
    args = parser.parse_args(argv)

    print(f"Ranking learners in {args.raw} ...")
    ranked = rank_learners(args.raw)
    stage1 = ranked[:STAGE1_LEARNERS]
    stage2 = ranked[:STAGE2_LEARNERS]
    print(
        f"  {len(ranked):,} learners, taking {len(stage1)} for pooled training "
        f"and the top {len(stage2)} for specialization"
    )

    everyone = read_corpus(args.raw, keep=set(stage1))
    splits = {learner: split_chronologically(everyone[learner]) for learner in stage1}

    # Every learner contributes the same amount, so none of them dominates a
    # pooled mean. Each cap is whatever the thinnest learner allows.
    cap_pooled = min(len(splits[l]["train"]) for l in stage1)
    cap_specialize = min(len(splits[l]["train"]) for l in stage2)
    cap_held_out = min(len(splits[l]["test"]) for l in stage2)
    print(
        f"  per learner: {cap_pooled} pooled, {cap_specialize} specialization, "
        f"{cap_held_out} held out"
    )

    names = {learner: f"learner_{i:02d}" for i, learner in enumerate(ranked)}

    def essays_of(learner: str, which: str, cap: int) -> list[dict]:
        history = everyone[learner]
        offsets = {e.writing_id: i for i, e in enumerate(history)}
        return [
            single_turn_record(names[learner], history, offsets[essay.writing_id])
            for essay in splits[learner][which][-cap:]
        ]

    def guidance_of(learner: str, which: str, cap: int) -> list[dict]:
        """This learner's guidance records, all of one style before the next."""
        history = everyone[learner]
        offsets = {e.writing_id: i for i, e in enumerate(history)}
        pool: list[dict] = []
        for style in STYLES:
            for essay in splits[learner][which][-cap:]:
                pool.extend(
                    multi_turn_records(names[learner], history, offsets[essay.writing_id], style)
                )
        return pool

    print("Writing:")

    # The pooled stage spends a fixed budget, a fifth of it under guidance, and
    # spreads that fifth evenly over its learners.
    budget = cap_pooled * len(stage1)
    target_guided = round(budget * MULTITURN_RATIO)
    pooled_singles: list[dict] = []
    pools: dict[str, list[dict]] = {}
    for learner in stage1:
        pooled_singles.extend(essays_of(learner, "train", cap_pooled))
        pools[learner] = guidance_of(learner, "train", cap_pooled)
    pooled_guided = select_balanced(pools, STYLE_WEIGHTS, target_guided, SEED)
    for record in pooled_guided:
        record["messages"][2]["content"] += "\n\n" + POOLED_TUTOR_SUFFIX
    keep_singles = budget - len(pooled_guided)
    if keep_singles < len(pooled_singles):
        random.Random(SEED).shuffle(pooled_singles)
        pooled_singles = pooled_singles[:keep_singles]
    _write(pooled_singles + pooled_guided, args.out / "stage1_pooled.jsonl")

    # Specialization keeps the learner's own essays and adds guidance on top.
    for learner in stage2:
        rng = learner_rng(learner, SEED)
        singles = essays_of(learner, "train", cap_specialize)
        pool = guidance_of(learner, "train", cap_specialize)
        n_guided, n_single = guidance_budget(
            len(singles), len(pool), SPECIALIZE_BUDGET, MULTITURN_RATIO
        )
        guided = select_by_style(pool, STYLE_WEIGHTS, n_guided, rng)
        rng.shuffle(singles)
        _write(singles[:n_single] + guided, args.out / "stage2" / f"{names[learner]}.jsonl")

        rng = learner_rng(learner, SEED)
        _write(
            essays_of(learner, "test", cap_held_out),
            args.out / "test_st" / f"{names[learner]}.jsonl",
        )
        by_style: dict[str, list[dict]] = defaultdict(list)
        for record in guidance_of(learner, "test", cap_held_out):
            by_style[record["style"]].append(record)
        chosen: list[dict] = []
        for style in sorted(by_style):
            candidates = list(by_style[style])
            rng.shuffle(candidates)
            chosen.extend(candidates[:TEST_PER_STYLE])
        _write(chosen, args.out / "test_mt" / f"{names[learner]}.jsonl")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

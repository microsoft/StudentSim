"""``studentsim-baselines``: score a prompted or state-tracking baseline.

A baseline does not have a per-student adapter, so it cannot go through the
evaluation path the trained simulators use, which decodes from a checkpoint.
It generates from a model as it is, and this entry point runs that generation
over the same held-out records and scores what comes back.

The scoring is deliberately more forgiving than the trained simulators get.
See :mod:`studentsim.baselines.scoring` for what that means and why: a model
that was never trained on the answer format would otherwise be measured on
its formatting.

Fidelity in chess and L2 reads the first turn's answer; in math it reads the
model's ranking over the four letters, since a closed model may decline to
emit a bare letter but will still rank one. Responsiveness reads the second
turn, after the tutor has spoken.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from studentsim.baselines.scoring import guidance_accuracy, matches, math_fidelity_accuracy
from studentsim.core.decoding import DecodingConfig
from studentsim.eval.protocol import protocol_for

BASELINES = ("maia2", "azure", "qwen3")
MATH_LETTERS = ("A", "B", "C", "D")
CHESS_MAIA_CANDIDATES_RE = re.compile(r"(?m)^[ \t]*Likely next moves: .*(?:\n|$)")
CHESS_MOVE_REQUEST = "What is your next move? Respond in UCI format."
CHESS_BARE_MOVE_REQUEST = (
    "Respond with ONLY the UCI move (4-5 lowercase characters like 'e2e4', no explanation)."
)
CHESS_SYSTEM_PROMPT = (
    "You are a chess move generator. Output ONLY the UCI move as 4-5 lowercase "
    "characters (e.g., 'e2e4', 'g1f3', 'a7a8q'). No reasoning, no explanation, "
    "no punctuation, no prefix or suffix \u2014 just the move."
)


def read_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chess_baseline_prompt(prompt: str) -> str:
    """The chess prompt as a model without a per-student adapter receives it.

    Two things separate it from what the trained simulator reads. The
    ``Likely next moves`` line is a Maia2-derived probability list, and Maia2
    stands as its own baseline, so a prompted model is scored on the profile
    the two share rather than on Maia2's move distribution. The move request
    then asks for a bare move, because the reply is compared against a
    greedy-decoded UCI move and prose around it would not parse.
    """
    prompt = CHESS_MAIA_CANDIDATES_RE.sub("", prompt)
    return prompt.replace(CHESS_MOVE_REQUEST, CHESS_BARE_MOVE_REQUEST)


def prompt_and_answer(
    record: dict,
    *,
    turn: int,
    render=None,
) -> tuple[str, str]:
    """The prompt up to ``turn`` and the answer recorded for it.

    ``turn`` 0 is the student's first response; turn 1 is the response after
    the tutor's message, which is where responsiveness is measured.
    """
    messages = record["messages"]
    index = 1 if turn == 0 else 3
    if len(messages) <= index:
        raise ValueError(f"record has {len(messages)} messages, needs more than {index}")
    prompt = "\n\n".join(m["content"] for m in messages[:index])
    if render is not None:
        prompt = render(prompt)
    return prompt, messages[index]["content"]


def build_simulator(name: str, *, domain: str, model: str | None, elo: int, device: str):
    """Construct the baseline named on the command line."""
    if name == "maia2":
        from studentsim.baselines.maia2 import Maia2Simulator

        return Maia2Simulator(elo=elo, device=device)
    if name == "azure":
        from studentsim.baselines.azure_openai import AzureOpenAIClient
        from studentsim.baselines.llm_simulator import LLMClientSimulator

        if not model:
            raise ValueError("--model names the deployment to call")
        # The GPT-5 family caps top_logprobs at 5; anything else takes the
        # default. A letter outside the returned set falls to the floor.
        top_k = 5 if "gpt-5" in model.lower() else 20
        # Chess answers are compared as bare UCI moves, so the model is told to
        # emit one and nothing else. The other domains score free text and take
        # the reply as it comes.
        system = CHESS_SYSTEM_PROMPT if domain == "chess" else None
        return LLMClientSimulator(
            client=AzureOpenAIClient(deployment=model),
            domain=domain,
            top_logprobs_k=top_k,
            system_prompt=system,
        )
    if name == "qwen3":
        from studentsim.baselines.base_qwen3 import build_base_qwen3_simulator

        if not model:
            raise ValueError("--model names the base checkpoint")
        return build_base_qwen3_simulator(base_model=model, domain=domain, device=device)
    raise ValueError(f"unknown baseline {name!r}; expected one of {list(BASELINES)}")


def score_fidelity(
    simulator,
    records: Sequence[dict],
    *,
    domain: str,
    decoding,
    render=None,
) -> dict:
    """The share of first turns the baseline reproduces."""
    if domain == "math":
        rows, answers = [], []
        for record in records:
            prompt, answer = prompt_and_answer(
                record, turn=0, render=render
            )
            rows.append(dict(simulator.logprobs(prompt, candidates=MATH_LETTERS)))
            answers.append(answer.strip())
        return {
            "metric": "fidelity",
            "domain": domain,
            "accuracy": math_fidelity_accuracy(rows, answers, MATH_LETTERS),
            "n_samples": len(rows),
        }

    pairs = [
        prompt_and_answer(record, turn=0, render=render)
        for record in records
    ]
    predictions = simulator.generate_batch([p for p, _ in pairs], decoding=decoding)
    hits = sum(matches(p, a) for p, (_, a) in zip(predictions, pairs))
    return {
        "metric": "fidelity",
        "domain": domain,
        "accuracy": hits / len(pairs) if pairs else 0.0,
        "n_samples": len(pairs),
    }


def score_responsiveness(
    simulator,
    records: Sequence[dict],
    *,
    domain: str,
    decoding,
    render=None,
) -> dict:
    """The share of second turns the baseline gets right after guidance.

    Per-mode figures come out alongside the total, because the held-out set
    holds an equal number of records per guidance mode and the breakdown is
    what makes modes comparable.
    """
    protocol = protocol_for(domain)
    pairs, modes = [], []
    for record in records:
        prompt, answer = prompt_and_answer(
            record, turn=1, render=render
        )
        pairs.append((f"{prompt}\n\n{protocol.turn2_suffix}", answer))
        modes.append(record.get(protocol.mode_field))
    predictions = simulator.generate_batch([p for p, _ in pairs], decoding=decoding)
    references = [a for _, a in pairs]

    per_mode: dict[str, float] = {}
    for mode in sorted({m for m in modes if m}):
        chosen = [i for i, m in enumerate(modes) if m == mode]
        per_mode[mode] = guidance_accuracy(
            [predictions[i] for i in chosen], [references[i] for i in chosen]
        )
    return {
        "metric": "responsiveness",
        "domain": domain,
        "accuracy": guidance_accuracy(list(predictions), references),
        "n_samples": len(pairs),
        "per_mode": per_mode,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="studentsim-baselines",
        description="Score a baseline on the held-out sets the simulators are scored on.",
    )
    parser.add_argument("--domain", required=True, choices=("chess", "l2", "math"))
    parser.add_argument("--baseline", required=True, choices=BASELINES)
    parser.add_argument("--model", default=None, help="Deployment name or checkpoint path.")
    parser.add_argument("--single-turn", type=Path, default=None, help="Fidelity records.")
    parser.add_argument("--multi-turn", type=Path, default=None, help="Guidance records.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--elo", type=int, default=1500, help="Maia2 only.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if args.single_turn is None and args.multi_turn is None:
        parser.error("nothing to score; pass --single-turn, --multi-turn, or both")
    if args.baseline == "maia2" and args.domain != "chess":
        parser.error("maia2 plays chess and nothing else")

    protocol = protocol_for(args.domain)
    simulator = build_simulator(
        args.baseline, domain=args.domain, model=args.model, elo=args.elo, device=args.device
    )
    render = chess_baseline_prompt if args.domain == "chess" else None

    report: dict = {"domain": args.domain, "baseline": args.baseline, "model": args.model}
    if args.single_turn is not None:
        decoding = DecodingConfig(
            max_new_tokens=protocol.max_new_tokens,
            temperature=0.0,
            repetition_penalty=protocol.fidelity_repetition_penalty,
        )
        report["fidelity"] = score_fidelity(
            simulator,
            read_records(args.single_turn),
            domain=args.domain,
            decoding=decoding,
            render=render,
        )
    if args.multi_turn is not None:
        decoding = DecodingConfig(
            max_new_tokens=protocol.max_new_tokens,
            temperature=0.0,
            repetition_penalty=protocol.guidance_repetition_penalty,
        )
        report["responsiveness"] = score_responsiveness(
            simulator,
            read_records(args.multi_turn),
            domain=args.domain,
            decoding=decoding,
            render=render,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

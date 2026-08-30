"""Checking the platform's answer keys before anything trains on them.

Some keys in the source data are wrong: arithmetic slips, and problems whose
figure was lost when the text was extracted, leaving a question that cannot be
answered from what remains. A model solves each problem on its own and the
records whose key it disputes are dropped, so that no student is scored against
an answer that is not the right one.

This step calls a language model, so it needs credentials and it costs money.
Run it once and keep the result.

    python -m studentsim.data.math.audit --raw data/math/raw --out data/math/audit_excluded.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from studentsim.core.llm import LLMClient, Message, open_client
from studentsim.data.math.problems import clean_text
from studentsim.data.math.rosters import choose_students, count_free_text
from studentsim.data.math.splits import answer_keys

csv.field_size_limit(sys.maxsize)

SYSTEM_PROMPT = """You are a careful mathematics expert verifying answer keys for middle-school math problems.

You will be shown a problem and a claimed correct answer. Your task:
1. Solve the problem independently.
2. Compare your answer to the claimed correct answer.
3. Output a verdict: AGREE or DISAGREE.

If you DISAGREE, you must briefly explain why and state YOUR answer.

Notes:
- The "claimed answer" comes from a real platform's answer key, but it may contain errors.
- Equivalent forms are AGREE (e.g., "0.5" vs "1/2", "x = -4" vs "-4").
- Pure formatting differences (extra spaces, decimal point styles) are AGREE.
- If the problem is ambiguous or asks something subtle that could legitimately have a different answer, prefer DISAGREE and explain.

Output format (plain text, no JSON):
VERDICT: AGREE
or
VERDICT: DISAGREE
MY_ANSWER: <your answer>
REASON: <one sentence>"""

USER_PROMPT = """Problem:
{problem}

Claimed correct answer: {answer}

Verify."""


def parse_verdict(reply: str) -> str:
    """``AGREE`` when the model accepts the key, anything else rejects it."""
    for line in (reply or "").splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            return line.split(":", 1)[1].strip().upper()
    return "UNPARSED"


def audit(
    client: LLMClient,
    bodies: dict[str, str],
    keys: dict[str, str],
) -> dict[str, str]:
    """One verdict per problem, in the order the problems are given."""
    verdicts: dict[str, str] = {}
    for i, (problem_id, body) in enumerate(bodies.items(), 1):
        reply = client.complete(
            [
                Message("system", SYSTEM_PROMPT),
                Message("user", USER_PROMPT.format(problem=body, answer=keys[problem_id])),
            ],
            max_tokens=400,
            temperature=0.0,
        )
        verdicts[problem_id] = parse_verdict(reply.text)
        if i % 100 == 0:
            print(f"  {i:,}/{len(bodies):,}")
    return verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m studentsim.data.math.audit")
    parser.add_argument("--raw", type=Path, default=Path("data/math/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/math/audit_excluded.json"))
    parser.add_argument("--model", default="gpt-4o", help="the model that checks the keys")
    args = parser.parse_args(argv)

    interactions = args.raw / "Interactions.csv"
    keys = answer_keys(args.raw / "Problems.csv")

    # Only the problems the selected students actually worked on can reach a
    # dataset, so only those are worth paying to check.
    _, pooled = choose_students(count_free_text(interactions, set(keys)))
    attempted = set()
    with open(interactions, encoding="utf-8", newline="") as handle:
        students = set(pooled)
        for row in csv.DictReader(handle):
            if row.get("user_id") in students:
                attempted.add(str(row.get("problem_id")))

    bodies = {}
    with open(args.raw / "Problems.csv", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            problem = str(row["problem_id"])
            if problem in keys and problem in attempted:
                bodies[problem] = clean_text(row.get("Problem Body") or "")

    print(f"Auditing {len(bodies):,} answer keys with {args.model} ...")
    verdicts = audit(open_client(args.model), bodies, keys)

    rejected = sorted(int(p) for p, verdict in verdicts.items() if verdict != "AGREE")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rejected))
    print(f"  {len(rejected):,} of {len(bodies):,} keys rejected -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

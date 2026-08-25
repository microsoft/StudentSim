"""Writing the reference tutor guidance the SFT corpus is built from.

The tutor is trained on examples of a teacher explaining a student's mistake,
and those examples are written by a model. Each position is put to the model
once per guidance mode, so the same mistake is explained four ways: naming the
error, asking about it, framing it as a plan, or setting it against the better
move.

The model sees the board as an image and the same position again as text, and
answers with three fields: its private reasoning, the explanation in algebraic
notation, and the same explanation in the coordinate notation the reward reads.
Two notations are asked for because the simulator is scored on coordinate
moves while a human reader wants algebraic ones.

The corpus this produces cannot be redistributed, so the release ships this
step rather than its output, and the wording varies between runs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MODES: Final = ("error_remediation", "socratic", "strategic", "comparative")

REQUIRED_FIELDS: Final = ("thinking", "instruction_san", "instruction_uci")

_PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")


@dataclass(frozen=True)
class PromptTemplate:
    """The system and user text for one guidance mode."""

    mode: str
    system: str
    user: str

    def placeholders(self) -> set[str]:
        return set(_PLACEHOLDER.findall(self.user))

    def render(self, fields: Mapping[str, str]) -> str:
        """Fill the user template.

        A placeholder with nothing to fill it is an error rather than an empty
        string: a prompt missing the student's move or the engine's evaluation
        would still look well-formed to the model and produce guidance about
        nothing.
        """
        missing = self.placeholders() - set(fields)
        if missing:
            raise ValueError(f"{self.mode}: no value for {sorted(missing)}")
        return _PLACEHOLDER.sub(lambda m: str(fields[m.group(1)]), self.user)


def load_templates(directory: str | Path, modes: Sequence[str] = MODES) -> dict[str, PromptTemplate]:
    """Read one system and user file per mode from a template directory."""
    root = Path(directory)
    templates = {}
    for mode in modes:
        system = root / f"{mode}_system.txt"
        user = root / f"{mode}_user.txt"
        for path in (system, user):
            if not path.is_file():
                raise FileNotFoundError(f"missing template {path}")
        templates[mode] = PromptTemplate(
            mode=mode,
            system=system.read_text(encoding="utf-8"),
            user=user.read_text(encoding="utf-8"),
        )
    return templates


@dataclass(frozen=True)
class GuidanceRequest:
    """One position, ready to be explained.

    ``wrong_move`` is the move being explained, and it is a field of its own
    rather than only a template placeholder: every stage downstream needs it,
    to draw the board, to check the guidance against the position, and to score
    what the student did next.
    """

    position_id: str
    fen: str
    fields: Mapping[str, str]
    wrong_move: str = ""
    board_image_path: str | None = None


def parse_guidance(text: str) -> dict[str, str]:
    """Read the model's answer.

    The answer is a JSON object, sometimes wrapped in a code fence despite the
    instruction not to. Anything missing a field is rejected rather than
    patched, because a record without its coordinate-notation explanation
    cannot be scored later.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"guidance is not JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"guidance is not an object, got {type(parsed).__name__}")
    missing = [field for field in REQUIRED_FIELDS if not parsed.get(field)]
    if missing:
        raise ValueError(f"guidance is missing {missing}")
    return {field: str(parsed[field]) for field in REQUIRED_FIELDS}


def build_messages(
    template: PromptTemplate, request: GuidanceRequest, *, image_url: str | None = None
) -> list[dict]:
    """Assemble the chat for one request.

    The board image goes alongside the text rather than replacing it: the model
    is asked to read the position both ways so that its explanation can be
    checked against the position afterwards.
    """
    user_text = template.render(request.fields)
    if image_url is None:
        content: object = user_text
    else:
        content = [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": user_text},
        ]
    return [
        {"role": "system", "content": template.system},
        {"role": "user", "content": content},
    ]


def generate(
    requests: Sequence[GuidanceRequest],
    templates: Mapping[str, PromptTemplate],
    *,
    complete,
    modes: Sequence[str] = MODES,
    image_url_for=None,
) -> Iterator[dict]:
    """Explain every request in every mode.

    ``complete`` takes the assembled messages and returns the model's text. A
    request whose answer cannot be read is reported and skipped, so one bad
    generation does not cost the rest of the run.
    """
    for request in requests:
        for mode in modes:
            template = templates[mode]
            image_url = image_url_for(request) if image_url_for else None
            messages = build_messages(template, request, image_url=image_url)
            try:
                guidance = parse_guidance(complete(messages))
            except ValueError as exc:
                yield {
                    "position_id": request.position_id,
                    "mode": mode,
                    "fen": request.fen,
                    "wrong_move": request.wrong_move,
                    "error": str(exc),
                }
                continue
            yield {
                "position_id": request.position_id,
                "mode": mode,
                "fen": request.fen,
                "wrong_move": request.wrong_move,
                **guidance,
            }


def write_guidance(rows: Sequence[dict], directory: str | Path) -> dict[str, int]:
    """Write one file per mode, and report how many each holds."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for mode in MODES:
        kept = [row for row in rows if row.get("mode") == mode and "error" not in row]
        with (root / f"{mode}.jsonl").open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[mode] = len(kept)
    return counts

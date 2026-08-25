"""Building the positions the tutor practises on, and the table that scores them.

An episode needs a position, the move the student actually played there, and a
way to score whatever move the student plays after hearing the tutor. All three
come from one pass so that the prompt and the reward cannot disagree.

That agreement is the reason this is a single step rather than two. Scoring the
candidates separately from the values the reward reads lets the two drift, and a
tutor can then be rewarded for guidance the prompt had already given away.
Here both are read from the same cache in the same pass.

Positions are kept only when they are worth teaching on: the cache has to cover
every legal move, so the reward is defined wherever the student goes, and the
student's move has to be enough worse than the best move that there is
something to fix.

An episode carries only the position, the side to move, and the move the
student played. The reward needs nothing else: it runs inside the training
worker, so it can build the student's turn from the position and the move
rather than being handed that text, and it reads centipawn values from the
table written here alongside the episodes rather than calling an engine. A
reward served as a separate process would need both of those carried in the
row, so a reward served out of process would need both carried in the row.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from studentsim.core.seeds import DATA_SAMPLER_SEED
from studentsim.tutor_rl.reward import move_quality

MODES: Final = ("error_remediation", "socratic", "strategic", "comparative")

DATA_SOURCE: Final = "chess_tutor"
"""Names the task on every row; the trainer routes rewards by it."""

DEFAULT_CP_GAP: Final = 100
DEFAULT_TOP_N: Final = 10
DEFAULT_MODE_TAG_FRACTION: Final = 0.0
"""No episode carries a mode tag by default, which is what makes the
style the tutor writes in a decision the policy makes."""
DEFAULT_VALIDATION_FRACTION: Final = 0.05
DEFAULT_SEED: Final = DATA_SAMPLER_SEED


@dataclass(frozen=True)
class Position:
    """One position a student got wrong, with what the engine thinks of it."""

    fen: str
    player_color: str
    wrong_move: str
    candidates: dict[str, int]
    """Every legal move at this position mapped to its centipawn value."""

    def best_move(self) -> str:
        return max(self.candidates, key=lambda move: self.candidates[move])

    def cp_gap(self) -> int:
        """How much the student's move gave up against the best one."""
        return self.candidates[self.best_move()] - self.candidates[self.wrong_move]


@dataclass
class PlaygroundRow:
    """One training episode."""

    fen: str
    player_color: str
    wrong_move: str
    prompt: str
    mode: str | None = None

    def to_record(self) -> dict:
        """The row as the RL trainer reads it.

        The trainer expects a chat-shaped prompt and reaches for the reward
        payload under ``reward_model``, passing that string to the reward
        function untouched. What is left describes the position and rides
        along in ``extra_info``.
        """
        return {
            "data_source": DATA_SOURCE,
            "prompt": [{"role": "user", "content": self.prompt}],
            "reward_model": {
                "style": "rule",
                "ground_truth": f"{self.fen}|{self.player_color}|{self.wrong_move}",
            },
            "extra_info": {
                "fen": self.fen,
                "player_color": self.player_color,
                "wrong_move": self.wrong_move,
                "mode": self.mode,
            },
        }


@dataclass
class PlaygroundSpec:
    """What to keep and how to present it."""

    cp_gap: int = DEFAULT_CP_GAP
    top_n_candidates: int = DEFAULT_TOP_N
    shuffle_candidates: bool = True
    drop_best_move_block: bool = True
    mode_tag_fraction: float = DEFAULT_MODE_TAG_FRACTION
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    seed: int = DEFAULT_SEED
    limit: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.mode_tag_fraction <= 1.0:
            raise ValueError(f"mode_tag_fraction must be in [0, 1], got {self.mode_tag_fraction}")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in [0, 1), got {self.validation_fraction}"
            )


def read_positions(path: str | Path) -> Iterator[Position]:
    """Read the wrong-move records the playground draws from."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield Position(
                fen=row["fen"],
                player_color=row.get("player_color", "white"),
                wrong_move=row["wrong_move"],
                candidates={move: int(cp) for move, cp in row["candidates"].items()},
            )


def covers_every_legal_move(position: Position, legal_moves: Sequence[str]) -> bool:
    """Whether the cache scores everywhere the student could go.

    Without full coverage a student who plays an uncached move would be scored
    as if the move were illegal, which is a penalty the tutor cannot influence.
    """
    return set(legal_moves) <= set(position.candidates)


def keep(position: Position, spec: PlaygroundSpec) -> bool:
    """Whether this position is worth an episode."""
    if position.wrong_move not in position.candidates:
        return False
    return position.cp_gap() >= spec.cp_gap


def build_prompt(
    position: Position, spec: PlaygroundSpec, rng: random.Random, *, mode: str | None
) -> str:
    """Write the tutor's prompt for one position.

    The candidate list is what the tutor is allowed to reason from. It is
    shuffled so the tutor cannot learn that the first entry is the answer, and
    the best move is normally withheld for the same reason.
    """
    ranked = sorted(position.candidates.items(), key=lambda item: -item[1])
    shown = list(ranked[: spec.top_n_candidates])
    # The student's move joins the list wherever it ranks. A position is kept
    # only because that move gave something up, so it often falls outside the
    # best few, and a tutor asked to explain a move it cannot see is being asked
    # the wrong question.
    if position.wrong_move not in [move for move, _ in shown]:
        played = position.candidates.get(position.wrong_move)
        if played is not None:
            shown.append((position.wrong_move, played))
            shown.sort(key=lambda item: -item[1])
    if spec.shuffle_candidates:
        rng.shuffle(shown)

    # Sectioned the way the tutor's own training data is, so that the policy
    # meets prompts it was trained on when it starts practising here.
    from studentsim.tutor_rl.sft_corpus import pieces_by_square

    sections = []
    if mode:
        sections.append(f"MODE: {mode}")
    sections.append(f"POSITION (FEN):\n{position.fen}")
    sections.append(f"PIECES BY SQUARE:\n{pieces_by_square(position.fen)}")
    sections.append(
        "STUDENT'S MOVE:\n"
        f"  Color: {position.player_color}\n"
        f"  Move: {position.wrong_move}"
    )
    if not spec.drop_best_move_block:
        sections.append(f"BEST MOVE: {position.best_move()}")
    sections.append("CANDIDATES: " + ", ".join(f"{move} ({cp})" for move, cp in shown))
    sections.append("Explain to the student how to improve on their move.")
    return "\n\n".join(sections)


def build_playground(
    positions: Sequence[Position], spec: PlaygroundSpec | None = None
) -> tuple[list[PlaygroundRow], list[PlaygroundRow]]:
    """Turn positions into training and validation episodes.

    Rows carrying a mode tag are placed first, so a run that reads its data in
    order trains on tagged episodes before untagged ones. Validation is drawn
    from the untagged tail, and never from the tagged opening.
    """
    spec = spec or PlaygroundSpec()
    rng = random.Random(spec.seed)

    # One episode per position. The same position reaching the playground twice
    # would be sampled twice during training, which weights it without saying so.
    seen: set[str] = set()
    kept = []
    for position in positions:
        if position.fen in seen or not keep(position, spec):
            continue
        seen.add(position.fen)
        kept.append(position)
    if spec.limit is not None:
        kept = kept[: spec.limit]

    tagged_count = int(len(kept) * spec.mode_tag_fraction)
    rows = []
    for index, position in enumerate(kept):
        mode = rng.choice(MODES) if index < tagged_count else None
        rows.append(
            PlaygroundRow(
                fen=position.fen,
                player_color=position.player_color,
                wrong_move=position.wrong_move,
                prompt=build_prompt(position, spec, rng, mode=mode),
                mode=mode,
            )
        )

    validation_size = int(len(rows) * spec.validation_fraction)
    if validation_size and validation_size < len(rows) - tagged_count:
        return rows[:-validation_size], rows[-validation_size:]
    return rows, []


def build_reward_table(
    positions: Sequence[Position],
    *,
    mate_clip_cp: int = 1500,
    scale_cp: int = 500,
    illegal_reward: float = -1.0,
) -> list[dict]:
    """Score every move the student could reach, once, ahead of training.

    Runtime then looks a move up instead of calling an engine, which is what
    keeps the reward identical to what the prompt was built from.
    """
    rows = []
    for position in positions:
        before = position.candidates.get(position.wrong_move)
        for move, cp in position.candidates.items():
            rows.append(
                {
                    "fen": position.fen,
                    "move": move,
                    "cp": cp,
                    "reward": move_quality(
                        cp,
                        before,
                        mate_clip_cp=mate_clip_cp,
                        scale_cp=scale_cp,
                        illegal_reward=illegal_reward,
                    ),
                }
            )
    return rows


def write_jsonl(rows: Sequence[dict], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

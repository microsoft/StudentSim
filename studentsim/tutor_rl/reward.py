"""Tutor RL reward functions.

The base reward is the centipawn improvement of the student simulator's
post-guidance move over the student's wrong move, squashed through a tanh:

    delta = clip(q_post - q_wrong, +-mate_clip_cp)
    reward = tanh(delta / scale_cp)

The illegal-move case (the model emits a non-UCI string or an illegal move on
the FEN) returns a fixed penalty (default -1.0). The composer
:class:`GatedReward` multiplies the base reward by the style and perception gates,
used to encourage a preferred instructional style.
:class:`PromptedStudentReward` keeps the base reward and replaces the trained
simulator with a model prompted to play the student, which is the comparison
the trained one is measured against.

All reward classes are stateless; configure them with constructor arguments
and call them per-rollout. The trainer wraps them in a verl-compatible
entry function (see :mod:`studentsim.tutor_rl.verl_runner`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from studentsim.core.decoding import DecodingConfig
from studentsim.core.records import StudentProfile
from studentsim.core.simulator import Simulator
from studentsim.domains.chess.modes import CHESS_DOMAIN_NAME
from studentsim.domains.chess.prompts import build_chess_multi_turn_messages
from studentsim.tutor_rl.gates import ALPHA_PERCEPTION, ALPHA_STYLE, gated_reward
from studentsim.tutor_rl.prompted_student import PromptedStudent
from studentsim.tutor_rl.stockfish_cache import StockfishLookup


@runtime_checkable
class RewardModel(Protocol):
    """Reward for one tutor rollout.

    Implementations consume the (fen, student's wrong move, tutor message)
    triple plus a domain-supplied :class:`StudentProfile` (used to materialize
    the student-simulator prompt) and produce a scalar reward in roughly
    ``[-1, 1]``. Stateless.
    """

    def __call__(
        self,
        *,
        fen: str,
        player_color: str,
        profile: StudentProfile,
        student_wrong_move: str,
        tutor_message: str,
    ) -> float: ...


@dataclass
class StudentSimReward(RewardModel):
    """Centipawn-delta reward gated by move legality.

    Parameters
    ----------
    student_sim
        Frozen Stage-1 chess simulator. The reward routes the multi-turn
        prompt through this simulator to get the post-guidance move.
    stockfish_lookup
        Centipawn lookup for both the student's wrong move and the
        simulator's post-guidance move.
    mate_clip_cp
        Clip on the centipawn delta, so a forced mate does not swamp the
        rest of the scale.
    scale_cp
        Centipawn value at which the tanh reaches its shoulder.
    illegal_reward
        Reward when the simulator's post-guidance move is illegal on the
        position. The default is the floor of the tanh range, so an illegal
        move never scores better than a legal one. Same penalty when the
        wrong-move centipawn
        is missing from the cache.
    decoding
        Decoding override for the student simulator. Defaults to chess
        single-turn decoding (max_new_tokens=32, greedy).
    """

    student_sim: Simulator
    stockfish_lookup: StockfishLookup
    mate_clip_cp: int = 1500
    scale_cp: int = 500
    illegal_reward: float = -1.0
    decoding: DecodingConfig | None = None

    def __post_init__(self) -> None:
        if self.mate_clip_cp <= 0:
            raise ValueError(f"mate_clip_cp must be positive, got {self.mate_clip_cp}")
        if self.scale_cp <= 0:
            raise ValueError(f"scale_cp must be positive, got {self.scale_cp}")

    def __call__(
        self,
        *,
        fen: str,
        player_color: str,
        profile: StudentProfile,
        student_wrong_move: str,
        tutor_message: str,
    ) -> float:
        if profile.domain != CHESS_DOMAIN_NAME:
            raise ValueError(
                f"StudentSimReward requires a chess profile; got {profile.domain!r}"
            )
        post_move = self._post_guidance_move(
            fen=fen,
            player_color=player_color,
            profile=profile,
            student_wrong_move=student_wrong_move,
            tutor_message=tutor_message,
        )
        # Import here so importing the reward module never costs a chess import.
        from studentsim.domains.chess.stockfish import is_legal_move

        if not post_move or not is_legal_move(fen=fen, move_uci=post_move):
            return self.illegal_reward
        q_wrong = self.stockfish_lookup.get(fen=fen, move_uci=student_wrong_move)
        q_post = self.stockfish_lookup.get(fen=fen, move_uci=post_move)
        if q_wrong is None or q_post is None:
            return self.illegal_reward
        delta = max(-self.mate_clip_cp, min(self.mate_clip_cp, q_post - q_wrong))
        return math.tanh(delta / self.scale_cp)

    def _post_guidance_move(
        self,
        *,
        fen: str,
        player_color: str,
        profile: StudentProfile,
        student_wrong_move: str,
        tutor_message: str,
    ) -> str:
        msgs = build_chess_multi_turn_messages(
            profile=profile,
            fen=fen,
            player_color=player_color,
            student_wrong_move=student_wrong_move,
            tutor_message=tutor_message,
        )
        # Flatten the chat to a single prompt string for Simulator.generate;
        # mirrors what render_chess_multi_turn does in the eval path.
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
        decoding = self.decoding or DecodingConfig(max_new_tokens=32, repetition_penalty=1.0)
        raw = self.student_sim.generate(prompt, decoding=decoding).strip()
        if not raw:
            return ""
        return raw.split()[0].lower()


def move_quality(
    cp_after: float | None,
    cp_before: float | None,
    *,
    mate_clip_cp: int = 1500,
    scale_cp: int = 500,
    illegal_reward: float = -1.0,
) -> float:
    """How much better the position is after the guidance than before it.

    The centipawn gain is clipped so a forced mate does not swamp the batch,
    then squashed so the reward stays bounded. A move with no entry in the
    table never happened on the board and takes the illegal-move penalty.
    """
    if cp_after is None or cp_before is None:
        return illegal_reward
    delta = max(-mate_clip_cp, min(mate_clip_cp, cp_after - cp_before))
    return math.tanh(delta / scale_cp)


@dataclass
class GatedReward(RewardModel):
    """Multiply a move-quality reward by the style and perception gates.

    The three signals come from one pass of the multi-head simulator, so this
    takes them rather than recomputing: the move the student plays after the
    guidance decides the base reward, and the two head outputs decide how much
    of it survives.
    """

    base: RewardModel
    preferred_style: str
    perception_weights: Sequence[float]
    alpha_style: float = ALPHA_STYLE
    alpha_perception: float = ALPHA_PERCEPTION

    def __call__(
        self,
        *,
        fen: str,
        player_color: str,
        profile: StudentProfile,
        student_wrong_move: str,
        tutor_message: str,
        style_probs: Sequence[float],
        perception_probs: Sequence[float],
    ) -> float:
        base_reward = self.base(
            fen=fen,
            player_color=player_color,
            profile=profile,
            student_wrong_move=student_wrong_move,
            tutor_message=tutor_message,
        )
        return gated_reward(
            base_reward,
            style_probs=style_probs,
            perception_probs=perception_probs,
            preferred_style=self.preferred_style,
            weights=self.perception_weights,
            alpha_style=self.alpha_style,
            alpha_perception=self.alpha_perception,
        )


@dataclass
class PromptedStudentReward(RewardModel):
    """Move-quality reward whose revised move comes from a prompted model.

    The comparison the trained simulator is measured against. It shares the
    move-quality term, the centipawn lookup and the illegal-move penalty with
    :class:`StudentSimReward`, and differs only in who plays the student, which
    is what the comparison is for.

    No gates. The style and perception gates read a trained simulator's own
    backbone, and a model reached through an API exposes none to read, so the
    comparison is against the move-quality term alone.

    ``profile`` is accepted to satisfy the protocol and not used: a prompted
    student has no per-student parameters, which is the property under test.
    """

    student: PromptedStudent
    stockfish_lookup: StockfishLookup
    mate_clip_cp: int = 1500
    scale_cp: int = 500
    illegal_reward: float = -1.0

    def __call__(
        self,
        *,
        fen: str,
        player_color: str,
        profile: StudentProfile,
        student_wrong_move: str,
        tutor_message: str,
    ) -> float:
        from studentsim.domains.chess.stockfish import is_legal_move

        post_move = self.student.move(
            fen=fen, wrong_move=student_wrong_move, tutor_message=tutor_message
        )
        if not post_move or not is_legal_move(fen=fen, move_uci=post_move):
            return self.illegal_reward
        return move_quality(
            self.stockfish_lookup.get(fen=fen, move_uci=post_move),
            self.stockfish_lookup.get(fen=fen, move_uci=student_wrong_move),
            mate_clip_cp=self.mate_clip_cp,
            scale_cp=self.scale_cp,
            illegal_reward=self.illegal_reward,
        )

"""Maia2 chess-specific knowledge-tracing baseline.

Maia2 is a human-style move predictor conditioned on FEN and player ELO
(McIlroy-Young 2020 / Tang 2024). Maia2 is the chess-specific
comparison point; it has no input pathway for natural-language
guidance, so under a tutor message it produces the move it would have played
anyway, and its responsiveness score is its no-guidance behaviour.

Implementation notes
--------------------
- The :class:`Maia2Simulator` parses FEN + player ELO out of the multi-turn
  record's ``meta`` (set by chess data builders) and queries the Maia2 model.
- It implements the :class:`Simulator` Protocol so the same fidelity and
  guidance runners can score it.
- ``Simulator.logprobs`` is not meaningful for Maia2 (it does not return next-
  token logprobs over UCI candidates); it raises ``NotImplementedError``.

The Maia2 package is an optional install (``pip install 'studentsim[maia2]'``);
:class:`Maia2Simulator` lazy-imports it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studentsim.core.decoding import DecodingConfig
from studentsim.core.simulator import Simulator, SimulatorSpec

# The FEN in a chess prompt is preceded by "FEN notation is:" (single-turn)
# or "FEN: " (multi-turn flattened). Match either.
_FEN_RE = re.compile(
    r"(?:FEN notation is:\s*|FEN:\s*)([rnbqkpRNBQKP1-8/]+\s+[wb]\s+\S+\s+\S+\s+\d+\s+\d+)"
)


@dataclass
class Maia2Simulator(Simulator):
    """Maia2 chess KT model adapted as a :class:`Simulator`.

    Parameters
    ----------
    elo
        Target player ELO; Maia2 is parameterized by ELO band. The chess
        baseline uses ELO matched to the held-out player's rating; set it to a
        fixed value to hold the rating constant across players instead.
    device
        ``"cuda"`` or ``"cpu"``.
    _inner
        Optional injected Maia2 model; tests pass a fake. Production callers
        leave ``None`` to construct the real model.
    """

    elo: int
    device: str = "cuda"
    _inner: Any = None

    def __post_init__(self) -> None:
        if self.elo < 100 or self.elo > 3500:
            raise ValueError(f"elo must be in [100, 3500], got {self.elo}")
        self.spec = SimulatorSpec(
            base_model="maia2",
            lora_adapter_path=None,
            domain="chess",
        )
        if self._inner is None:
            from maia2 import inference  # type: ignore[import-not-found]
            self._inner = inference.from_pretrained(self.device)

    def generate(self, prompt: str, *, decoding: DecodingConfig) -> str:
        """Parse the FEN out of ``prompt`` and return Maia2's top-1 move.

        The decoding config is ignored: Maia2 has its own non-greedy decoding
        and no concept of ``max_new_tokens``.
        """
        fen = _extract_fen(prompt)
        if fen is None:
            return ""
        return self._predict_top_move(fen)

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        decoding: DecodingConfig,
    ) -> list[str]:
        return [self.generate(p, decoding=decoding) for p in prompts]

    def logprobs(
        self,
        prompt: str,
        *,
        candidates: Sequence[str],
    ) -> Mapping[str, float]:
        raise NotImplementedError(
            "Maia2 does not expose next-token logprobs over arbitrary UCI candidates; "
            "use Maia2Simulator.generate (top-1 move) for fidelity."
        )

    def _predict_top_move(self, fen: str) -> str:
        """Run Maia2 inference and return the top-1 UCI move.

        The exact Maia2 API surface varies by package version; this wraps the
        call shape the current binding exposes. Subclass and override if yours
        exposes a different entry point.
        """
        import chess  # lazy import

        board = chess.Board(fen)
        result = self._inner.predict(board, self.elo)
        # The Maia2 inference module typically returns a list of (uci, probability)
        # tuples sorted by probability descending. The first element is the top-1.
        if not result:
            return ""
        top = result[0]
        if isinstance(top, (tuple, list)) and len(top) >= 1:
            return str(top[0])
        return str(top)


def _extract_fen(prompt: str) -> str | None:
    """Pull the FEN string out of a chess prompt. Returns ``None`` if absent."""
    match = _FEN_RE.search(prompt)
    return match.group(1).strip() if match else None

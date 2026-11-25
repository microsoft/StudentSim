"""Maia2 chess-specific knowledge-tracing baseline.

Maia2 is a human-style move predictor conditioned on FEN and the ratings of
both players (Tang et al. 2024). It is the chess-specific comparison point; it
has no input pathway for natural-language guidance, so under a tutor message
it plays the move it would have played anyway, and its responsiveness score is
its no-guidance behaviour.

Implementation notes
--------------------
- The model is the ``rapid`` checkpoint, loaded with
  ``maia2.model.from_pretrained`` and queried one position at a time through
  ``maia2.inference.inference_each``. The prediction is the move with the
  highest probability.
- :meth:`Maia2Simulator.top_move` takes the FEN and both ratings directly.
  :meth:`Maia2Simulator.generate` reads the FEN out of a chess prompt and uses
  the simulator's own rating for both players.
- ``Simulator.logprobs`` is not meaningful for Maia2 (it does not return next-
  token logprobs over UCI candidates); it raises ``NotImplementedError``.

The Maia2 package is an optional install (``pip install 'studentsim[maia2]'``
plus the package from its own repository); :class:`Maia2Simulator`
lazy-imports it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studentsim.core.decoding import DecodingConfig
from studentsim.core.simulator import Simulator, SimulatorSpec

# The FEN in a chess prompt is preceded by "FEN notation is:" (single-turn)
# or "FEN: " (multi-turn flattened). Match either.
_FEN_RE = re.compile(
    r"(?:FEN notation is:\s*|FEN:\s*)([rnbqkpRNBQKP1-8/]+\s+[wb]\s+\S+\s+\S+\s+\d+\s+\d+)"
)

MoveProbabilities = Callable[[str, int, int], Mapping[str, float]]
"""``(fen, elo_self, elo_oppo) -> {uci_move: probability}``."""


def _maia2_device(device: str) -> str:
    """Maia2 names devices ``"gpu"`` and ``"cpu"``."""
    return "cpu" if device == "cpu" else "gpu"


@dataclass
class Maia2Simulator(Simulator):
    """Maia2 chess KT model adapted as a :class:`Simulator`.

    Parameters
    ----------
    elo
        Rating used for both players when a record carries none.
    device
        ``"cuda"`` (or ``"gpu"``) or ``"cpu"``.
    model_type
        Maia2 checkpoint, ``"rapid"`` or ``"blitz"``.
    save_root
        Where Maia2 keeps (and on first use downloads) its checkpoint.
    _inner
        Optional injected :data:`MoveProbabilities`; tests pass a fake.
        Production callers leave ``None`` to load the real model.
    """

    elo: int
    device: str = "cuda"
    model_type: str = "rapid"
    save_root: str = "./maia2_models"
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
            from maia2 import inference, model  # type: ignore[import-not-found]

            loaded = model.from_pretrained(
                type=self.model_type,
                device=_maia2_device(self.device),
                save_root=self.save_root,
            )
            loaded.eval()
            prepared = inference.prepare()

            def move_probabilities(fen: str, elo_self: int, elo_oppo: int):
                move_probs, _ = inference.inference_each(
                    loaded, prepared, fen, elo_self, elo_oppo
                )
                return move_probs

            self._inner = move_probabilities

    def top_move(self, fen: str, elo_self: int, elo_oppo: int) -> str | None:
        """Maia2's most probable move, or ``None`` if it ranks none.

        An exception from Maia2 propagates, so the caller decides whether a
        position it cannot read counts as a miss or is left out.
        """
        move_probs = self._inner(fen, elo_self, elo_oppo)
        if not move_probs:
            return None
        return max(move_probs, key=move_probs.get)

    def generate(self, prompt: str, *, decoding: DecodingConfig) -> str:
        """Parse the FEN out of ``prompt`` and return Maia2's top-1 move.

        The decoding config is ignored: Maia2 has no concept of
        ``max_new_tokens``.
        """
        fen = extract_fen(prompt)
        if fen is None:
            return ""
        return self.top_move(fen, self.elo, self.elo) or ""

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
            "use Maia2Simulator.top_move (top-1 move) for fidelity."
        )


def extract_fen(prompt: str) -> str | None:
    """Pull the FEN string out of a chess prompt. Returns ``None`` if absent."""
    match = _FEN_RE.search(prompt)
    return match.group(1).strip() if match else None

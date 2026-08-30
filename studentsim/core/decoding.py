"""Decoding configuration with the ``enable_thinking=False`` invariant.

Every decode call in StudentSim sets
``enable_thinking=False`` on Qwen3's chat template. Without that, Qwen3's chat
template defaults to opening a ``<think>`` block that often does not close
within ``max_new_tokens``, so the model never emits the answer token. The
invariant lives on this dataclass so it cannot accidentally be dropped by a
caller; the only way to bypass it is to pass ``allow_thinking=True`` explicitly
(used by no production code path).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    """Per-call decoding parameters.

    The defaults decode greedily, with the thinking block turned off. Each
    domain's own values live in :class:`studentsim.eval.protocol.EvalProtocol`,
    which is where evaluation reads them from.

    Parameters
    ----------
    max_new_tokens
        Token budget. Chess and math use ``32`` (short categorical responses); L2
        uses ``256`` (free-form essay fragments).
    temperature
        Sampling temperature. ``0.0`` is greedy and is the default everywhere in
        evaluation; tutor RL rollouts override to ``1.0``.
    top_p
        Nucleus sampling threshold. Ignored when ``do_sample=False``.
    repetition_penalty
        Per-token repetition penalty. L2 uses ``1.1`` because pure greedy decoding
        on free-form essays occasionally entered repeat loops past the natural
        essay end on a non-trivial fraction of outputs; ``1.0`` everywhere else.
    do_sample
        ``False`` means greedy; ``True`` enables sampling via ``temperature`` /
        ``top_p``.
    enable_thinking
        ALWAYS ``False`` in production. See module docstring.
    """

    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    do_sample: bool = False
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if not (0.0 <= self.temperature):
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")

    def as_hf_kwargs(self) -> dict:
        """Render as ``transformers.GenerationConfig`` keyword arguments."""
        kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        return kwargs

    def as_chat_template_kwargs(self) -> dict:
        """Render as ``tokenizer.apply_chat_template`` keyword arguments."""
        return {"enable_thinking": self.enable_thinking}

    def as_vllm_kwargs(self) -> dict:
        """Render as vLLM ``SamplingParams`` keyword arguments.

        vLLM's default uses greedy when ``temperature=0`` regardless of ``do_sample``.
        """
        return {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }

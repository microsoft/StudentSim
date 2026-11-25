"""Base Qwen3-4B-Instruct (no LoRA) baseline.

Included as a baseline for L2 and math (left out of chess, where an
untrained model rarely emits a legal move in coordinate notation, so a score
would measure format compliance rather than behaviour).

The model answers whole chats: each one goes through the tokenizer's chat
template with thinking disabled and is decoded greedily, one chat at a time,
with no repetition penalty. Math fidelity reads the next-token distribution
after the prompt, taking for each letter the higher of its bare token and its
space-prefixed token.

Heavy imports (torch, transformers) are deferred to construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from studentsim.core.decoding import DecodingConfig
from studentsim.core.llm import Message
from studentsim.core.simulator import SimulatorSpec
from studentsim.domains.math.fidelity import LOGPROB_FLOOR

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _as_dicts(messages: Sequence[Message]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


class BaseQwen3:
    """The base model, unadapted, answering chats.

    Parameters
    ----------
    domain
        Domain name; recorded on :attr:`spec`.
    base_model
        Checkpoint path or hub id.
    device
        Torch device the whole model is placed on.
    """

    def __init__(
        self,
        *,
        domain: str,
        base_model: str = _DEFAULT_BASE_MODEL,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = SimulatorSpec(base_model=base_model, lora_adapter_path=None, domain=domain)
        self._torch = torch
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self._model.eval()

    def _prompt_ids(self, messages: Sequence[Message]) -> Any:
        prompt = self._tokenizer.apply_chat_template(
            _as_dicts(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self._tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self._device)

    def chat(self, messages: Sequence[Message], *, max_new_tokens: int) -> str:
        """The greedy continuation of one chat."""
        ids = self._prompt_ids(messages)
        with self._torch.no_grad():
            out = self._model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def generate_chats(
        self,
        chats: Sequence[Sequence[Message]],
        *,
        max_new_tokens: int,
        catch_errors: bool = False,
    ) -> list[str]:
        return [self.chat(chat, max_new_tokens=max_new_tokens) for chat in chats]

    def letter_logprobs(
        self, messages: Sequence[Message], letters: Sequence[str]
    ) -> dict[str, float]:
        """Next-token log-probability of each letter after the prompt."""
        ids = self._prompt_ids(messages)
        with self._torch.no_grad():
            logits = self._model(ids).logits[0, -1, :]
        log_probs = self._torch.log_softmax(logits, dim=-1)
        scores: dict[str, float] = {}
        for letter in letters:
            options = [
                float(log_probs[token_ids[0]].item())
                for token_ids in (
                    self._tokenizer.encode(letter, add_special_tokens=False),
                    self._tokenizer.encode(" " + letter, add_special_tokens=False),
                )
                if token_ids
            ]
            scores[letter] = max(options) if options else LOGPROB_FLOOR
        return scores

    def letter_logprobs_batch(
        self,
        chats: Sequence[Sequence[Message]],
        letters: Sequence[Sequence[str]],
    ) -> list[dict[str, float]]:
        return [self.letter_logprobs(chat, options) for chat, options in zip(chats, letters)]

    # --- Simulator Protocol surface -----------------------------------------

    def generate(self, prompt: str, *, decoding: DecodingConfig) -> str:
        return self.chat(
            [Message(role="user", content=prompt)], max_new_tokens=decoding.max_new_tokens
        )

    def generate_batch(self, prompts: Sequence[str], *, decoding: DecodingConfig) -> list[str]:
        return [self.generate(p, decoding=decoding) for p in prompts]

    def logprobs(self, prompt: str, *, candidates: Sequence[str]) -> Mapping[str, float]:
        return self.letter_logprobs([Message(role="user", content=prompt)], candidates)


def build_base_qwen3_simulator(
    *,
    domain: str,
    base_model: str = _DEFAULT_BASE_MODEL,
    device: str = "cuda",
) -> BaseQwen3:
    """Construct the base-model baseline for ``domain``."""
    return BaseQwen3(domain=domain, base_model=base_model, device=device)

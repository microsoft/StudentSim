"""HuggingFace + PEFT simulator backend.

The :class:`HFSimulator` loads a base model and, optionally, a LoRA adapter,
and exposes :meth:`generate`, :meth:`generate_batch` and :meth:`logprobs`,
all honouring the ``enable_thinking=False`` invariant.

An adapter arrives in more than one shape. A clean PEFT save loads directly; an
ms-swift or verl checkpoint carries merged-shape tensors under a state_dict
with no adapter config beside it, and is reconstructed into an adapter here.

Heavy imports (torch, transformers, peft) are deferred to instance construction
so that the rest of the package remains lightweight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from studentsim.core.decoding import DecodingConfig
from studentsim.core.simulator import SimulatorSpec

if TYPE_CHECKING:  # avoid heavy imports at module load
    import torch  # noqa: F401


class HFSimulator:
    """Local HuggingFace + PEFT simulator.

    Handles three checkpoint shapes:

    1. **Clean PEFT adapter** (``adapter_config.json`` + ``adapter_model.safetensors``):
       loaded via ``PeftModel.from_pretrained`` and merged with the base.
    2. **ms-swift / verl 'merged-shape state_dict'** (every key prefixed
       ``base_model.model.`` with ``.lora_A.default.weight`` / ``.lora_B.default.weight``
       tensors but no ``adapter_config.json``): reconstructed via
       ``peft.LoraConfig`` + ``get_peft_model`` + ``load_state_dict(strict=False)``
       + ``merge_and_unload``.
    3. **No adapter** (``spec.lora_adapter_path is None``): base model only.

    The chat template is invoked with ``enable_thinking=False`` for every call,
    matching the :class:`DecodingConfig` invariant.
    """

    def __init__(
        self,
        spec: SimulatorSpec,
        *,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        lora_rank: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.05,
        target_modules: Sequence[str] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self._device = device
        self._torch = torch

        dtype = _resolve_dtype(torch, torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(spec.base_model, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            spec.base_model,
            torch_dtype=dtype,
            device_map=device if device != "cuda" else "auto",
        )
        if spec.lora_adapter_path:
            self._model = _load_adapter(
                base,
                spec.lora_adapter_path,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=tuple(target_modules),
            )
        else:
            self._model = base
        self._model.eval()

    # --- Simulator Protocol surface -----------------------------------------

    def generate(self, prompt: str, *, decoding: DecodingConfig) -> str:
        return self.generate_batch([prompt], decoding=decoding)[0]

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        decoding: DecodingConfig,
    ) -> list[str]:
        if not prompts:
            return []
        templated = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                **decoding.as_chat_template_kwargs(),
            )
            for p in prompts
        ]
        inputs = self._tokenizer(
            templated,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        gen_kwargs = decoding.as_hf_kwargs() | {
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        with self._torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        # Strip the prompt portion to return only the generated continuation.
        prompt_lens = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_lens:]
        return self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    def logprobs(
        self,
        prompt: str,
        *,
        candidates: Sequence[str],
    ) -> Mapping[str, float]:
        """Natural-log probability of each candidate as the next-token continuation.

        Used for the math 4-way multiple-choice metric. Each candidate is
        tokenized; the logprob returned is the sum of per-token log-softmax
        values for the candidate's tokens, conditioned on the prompt.
        """
        templated = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self._tokenizer(
            templated, return_tensors="pt", truncation=True, max_length=2048
        ).input_ids.to(self._model.device)

        scores: dict[str, float] = {}
        with self._torch.no_grad():
            for cand in candidates:
                cand_ids = self._tokenizer(
                    cand, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self._model.device)
                input_ids = self._torch.cat([prompt_ids, cand_ids], dim=1)
                logits = self._model(input_ids=input_ids).logits
                # Logits at position t predict token t+1, so the candidate's own
                # tokens are read from [prompt_len-1 : prompt_len-1+cand_len].
                start = prompt_ids.shape[1] - 1
                cand_len = cand_ids.shape[1]
                relevant = logits[0, start : start + cand_len, :]
                log_probs = self._torch.log_softmax(relevant.float(), dim=-1)
                token_scores = log_probs.gather(
                    -1, cand_ids[0].unsqueeze(-1)
                ).squeeze(-1)
                scores[cand] = float(token_scores.sum().item())
        return scores


def _resolve_dtype(torch_mod, name: str):
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch_mod.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch_mod.float16
    if name in {"fp32", "float32"}:
        return torch_mod.float32
    raise ValueError(f"Unsupported torch_dtype: {name!r}")


def _load_adapter(
    base,
    adapter_path: str,
    *,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: tuple[str, ...],
):
    """Attach a LoRA adapter to ``base``, handling the three checkpoint formats."""
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter path does not exist: {adapter_path}")

    config_path = adapter_dir / "adapter_config.json"
    if config_path.is_file():
        # Clean PEFT save. Use PeftModel.from_pretrained.
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        return peft_model.merge_and_unload()

    # ms-swift / verl save format: a merged-shape state_dict with base_model.model.*
    # prefixes and lora_{A,B}.default.weight tensors, no adapter_config.json.
    from peft import LoraConfig, get_peft_model

    state_dict = _load_state_dict_from_dir(adapter_dir)
    if not any(k.startswith("base_model.model.") for k in state_dict):
        raise RuntimeError(
            f"Unknown adapter format at {adapter_dir}: no adapter_config.json and "
            "no 'base_model.model.' prefix on state_dict keys."
        )
    lcfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lcfg)
    peft_model.load_state_dict(state_dict, strict=False)
    return peft_model.merge_and_unload()


def _load_state_dict_from_dir(path: Path) -> dict:
    """Load a state_dict from either ``pytorch_model.bin`` or sharded safetensors."""
    import torch
    from safetensors.torch import load_file

    bin_path = path / "pytorch_model.bin"
    if bin_path.is_file():
        return torch.load(str(bin_path), map_location="cpu")

    # Try sharded safetensors.
    safetensors_files = sorted(path.glob("*.safetensors"))
    if safetensors_files:
        combined: dict = {}
        for shard in safetensors_files:
            combined.update(load_file(str(shard)))
        return combined

    raise FileNotFoundError(
        f"No state_dict found under {path} "
        "(looked for pytorch_model.bin and *.safetensors)"
    )

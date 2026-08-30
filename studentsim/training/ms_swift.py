"""ms-swift CLI wrapper.

The ms-swift trainer is a CLI tool (``swift sft ...``); :func:`build_ms_swift_command`
materializes a complete command line from a :class:`TrainingConfig` plus the
location of the ``swift`` binary (resolved via
:func:`studentsim.core.paths.swift_bin`).

Side-effect-free at module import: actually launching the subprocess is the
responsibility of :class:`Stage1Trainer` / :class:`Stage2Trainer`, and they
expose a ``runner`` injection point for tests.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from dataclasses import dataclass

from studentsim.training.config import TrainingConfig


@dataclass(frozen=True, slots=True)
class MsSwiftCommand:
    """A fully materialized ms-swift command, ready to be executed."""

    argv: Sequence[str]

    def shell(self) -> str:
        """Render as a shell-escaped command line (for logging / docs)."""
        return " ".join(shlex.quote(a) for a in self.argv)


def build_ms_swift_command(
    config: TrainingConfig,
    *,
    swift_binary: str,
) -> MsSwiftCommand:
    """Construct the ``swift sft`` command corresponding to ``config``.

    Plumbing of LoRA hyperparams, optimizer settings, schedule, batch sizes,
    sequence budget, precision, gradient checkpointing, multi-turn ratio,
    initial-adapter loading (for Stage 2), and seeds. The output is a list of
    argv tokens; callers run it via :class:`subprocess.run` or an injected
    fake runner.
    """
    argv: list[str] = [swift_binary, "sft"]

    argv += ["--model", config.base_model]
    argv += ["--dataset", config.data_path]
    argv += ["--output_dir", config.output_dir]

    # LoRA.
    argv += ["--train_type", "lora"]
    argv += ["--lora_rank", str(config.lora.rank)]
    argv += ["--lora_alpha", str(config.lora.alpha)]
    argv += ["--lora_dropout", str(config.lora.dropout)]
    argv += ["--target_modules", *config.lora.target_modules]

    # Optimizer + schedule.
    argv += ["--optim", config.optimizer.name]
    argv += ["--learning_rate", str(config.optimizer.learning_rate)]
    argv += ["--adam_beta1", str(config.optimizer.beta1)]
    argv += ["--adam_beta2", str(config.optimizer.beta2)]
    argv += ["--weight_decay", str(config.optimizer.weight_decay)]
    argv += ["--max_grad_norm", str(config.optimizer.grad_clip)]
    argv += ["--lr_scheduler_type", config.optimizer.schedule]
    argv += ["--warmup_steps", str(config.optimizer.warmup_steps)]

    # Batch + sequence.
    argv += ["--per_device_train_batch_size", str(config.per_device_batch)]
    argv += ["--gradient_accumulation_steps", str(config.gradient_accumulation)]
    argv += ["--max_length", str(config.max_seq_len)]
    argv += ["--num_train_epochs", str(config.epochs)]

    # Precision.
    if config.precision == "bf16":
        argv += ["--bf16", "true"]
    elif config.precision == "fp16":
        argv += ["--fp16", "true"]
    if config.gradient_checkpointing:
        argv += ["--gradient_checkpointing", "true"]

    # A model that reads images needs the vision side pinned down.
    if config.freeze_vision_tower:
        argv += ["--freeze_vit", "true"]
    if config.max_pixels is not None:
        argv += ["--max_pixels", str(config.max_pixels)]

    # Loss is taken over the response only, with empty think blocks masked out.
    argv += ["--loss_scale", "ignore_empty_think"]

    # Save / eval cadence. Neither stage validates during training.
    argv += ["--save_steps", str(config.save_freq)]
    argv += ["--save_total_limit", str(config.save_total_limit)]
    argv += ["--eval_strategy", "no"]

    # Seeds.
    argv += ["--seed", str(config.trainer_seed)]
    argv += ["--data_seed", str(config.data_sampler_seed)]

    # Stage 2: load Stage-1 adapter.
    if config.stage == 2 and config.initial_adapter is not None:
        argv += ["--adapters", config.initial_adapter]
        # Don't re-init args from the saved adapter; the YAML for Stage 2 is canonical.
        argv += ["--load_args", "false"]

    return MsSwiftCommand(argv=tuple(argv))

"""Training configuration dataclasses.

The training recipe is captured here in three nested dataclasses
:class:`LoRAConfig`, :class:`OptimizerConfig`, and :class:`TrainingConfig`.
Values for each domain and stage are stored as YAMLs under
``configs/training/`` and loaded via :meth:`TrainingConfig.from_yaml`. The code
path is the same in every domain; only these values differ.

The defaults are the shared recipe, and the per-domain YAMLs override only
what the data forces. All three domains read the same dataclass, so changing a
default changes every domain at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from studentsim.core.seeds import TRAINER_SEED


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    """LoRA adapter geometry shared by Stage 1 + Stage 2.

    Defaults: rank 128 with alpha 256, a 2x scaling, dropout 0.05, all linear projections.
    """

    rank: int = 128
    alpha: int = 256
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """AdamW with a cosine schedule, warmup counted in steps rather than epochs."""

    name: str = "adamw_torch_fused"
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    schedule: str = "cosine"
    warmup_steps: int = 100

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Top-level config for one training stage (Stage 1 or Stage 2).

    The combination of (per_device_batch * gradient_accumulation * world_size)
    must equal :attr:`effective_batch` (validated at construction).
    """

    base_model: str
    output_dir: str
    stage: int                             # 1 or 2
    data_path: str                         # training records for this stage

    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    precision: str = "bf16"
    gradient_checkpointing: bool = True
    per_device_batch: int = 4
    gradient_accumulation: int = 8
    world_size: int = 8
    effective_batch: int = 256
    max_seq_len: int = 4096
    epochs: float = 1.0
    multi_turn_ratio: float = 0.20
    save_freq: int = 200
    save_total_limit: int = 3

    # ms-swift --seed / --data_seed. Varying both together makes a measured
    # spread cover which records were drawn as well as what order they came in;
    # CROSS_SEED_RUNS holds three such pairs.
    trainer_seed: int = TRAINER_SEED
    data_sampler_seed: int = TRAINER_SEED

    initial_adapter: str | None = None       # for Stage 2: path to Stage-1 adapter

    # Set when the model reads images as well as text. Freezing the vision
    # tower keeps the run to the language side, which is what the tutor SFT
    # does; the pixel cap bounds how large a rendered board may arrive.
    freeze_vision_tower: bool = False
    max_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {self.stage}")
        if self.precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"precision must be bf16/fp16/fp32, got {self.precision}")
        if self.effective_batch != (
            self.per_device_batch * self.gradient_accumulation * self.world_size
        ):
            raise ValueError(
                f"effective_batch ({self.effective_batch}) must equal "
                f"per_device_batch ({self.per_device_batch}) * "
                f"gradient_accumulation ({self.gradient_accumulation}) * "
                f"world_size ({self.world_size})"
            )
        if not (0.0 <= self.multi_turn_ratio <= 1.0):
            raise ValueError(
                f"multi_turn_ratio must be in [0, 1], got {self.multi_turn_ratio}"
            )
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.stage == 2 and self.initial_adapter is None:
            raise ValueError("Stage 2 training requires initial_adapter (path to Stage-1 adapter)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        """Load a config from a YAML file with nested ``lora`` / ``optimizer`` blocks."""
        with Path(path).open() as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingConfig:
        """Construct from a plain dict (after YAML load or as test fixtures)."""
        data = dict(raw)
        if "lora" in data and isinstance(data["lora"], dict):
            lora_data = dict(data["lora"])
            if "target_modules" in lora_data:
                lora_data["target_modules"] = tuple(lora_data["target_modules"])
            data["lora"] = LoRAConfig(**lora_data)
        if "optimizer" in data and isinstance(data["optimizer"], dict):
            data["optimizer"] = OptimizerConfig(**data["optimizer"])
        return cls(**data)

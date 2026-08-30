"""Settings for the chess tutor RL loop.

:class:`RLConfig` holds what every run shares: batch sizes, the KL terms, how
many rollouts a prompt gets, and how the GPUs are divided between training,
rollout generation and the reward. :class:`RewardConfig` holds what the runs
differ by, which is what the reward is made of. :class:`TutorRlConfig` is the
pair, and is what a YAML file under ``configs/tutor_rl/`` deserializes to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from studentsim.core.seeds import TRAINER_SEED


@dataclass(frozen=True, slots=True)
class GpuLayout:
    """How one node's GPUs are divided between the parts of a run.

    Four train the actor under FSDP, one generates its rollouts through vLLM,
    and one holds the frozen reward simulator. A style gate reads a head on
    that same simulator, so it needs a GPU of its own only when it is served
    separately.
    """

    n_trainer_gpus: int = 4
    n_rollout_gpus: int = 1
    n_reward_sim_gpus: int = 1
    n_style_gate_gpus: int = 0

    def total(self) -> int:
        return (
            self.n_trainer_gpus
            + self.n_rollout_gpus
            + self.n_reward_sim_gpus
            + self.n_style_gate_gpus
        )


@dataclass(frozen=True, slots=True)
class RLConfig:
    """Hyperparameters shared by every tutor RL run."""

    actor_sft_base: str                    # path to merged SFT chess tutor checkpoint
    output_dir: str = "tutor_rl/run"
    playground_train: str = "tutor_rl/playground_train.parquet"
    playground_val: str = "tutor_rl/playground_val.parquet"

    # Per-step optimisation.
    train_batch_size: int = 112
    ppo_mini_batch_size: int = 112     # one update per batch
    ppo_micro_batch_size_per_gpu: int = 2
    advantage_estimator: str = "grpo"  # critic-free; no value model is trained
    use_kl_in_reward: bool = False
    use_kl_loss: bool = True
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.01
    entropy_coeff: float = 0.0

    actor_optimizer: str = "adamw"
    actor_learning_rate: float = 1e-6

    # Rollout settings.
    rollout_engine: Literal["vllm"] = "vllm"
    rollouts_per_prompt: int = 4       # the group GRPO compares within
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_top_k: int = -1                 # disabled
    max_prompt_length: int = 2048
    max_response_length: int = 512

    total_steps: int = 20
    save_freq: int = 5
    test_freq: int = 5
    eval_checkpoint_step: int = 20     # the run's last step

    # Adapter settings for the actor, read only when the rank is non-zero.
    actor_lora_rank: int = 0          # 0 trains the policy in full
    actor_lora_alpha: int = 64        # unused while the rank is 0
    actor_lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    seed: int = TRAINER_SEED
    gpu_layout: GpuLayout = field(default_factory=GpuLayout)

    def __post_init__(self) -> None:
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.ppo_mini_batch_size > self.train_batch_size:
            raise ValueError(
                f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be "
                f"<= train_batch_size ({self.train_batch_size})"
            )
        if self.actor_learning_rate <= 0:
            raise ValueError("actor_learning_rate must be positive")
        if not (0.0 < self.rollout_top_p <= 1.0):
            raise ValueError("rollout_top_p must be in (0, 1]")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.eval_checkpoint_step > self.total_steps:
            raise ValueError(
                f"eval_checkpoint_step ({self.eval_checkpoint_step}) > "
                f"total_steps ({self.total_steps})"
            )


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """What one run's reward is made of.

    Everything else about two runs is held fixed, so this is where they
    differ, and what they differ in is who plays the student. Give
    ``reward_simulator_path`` and the two head files for a trained simulator,
    or ``student_model`` for a model prompted to play one.

    The gates only apply to the first. They read the trained simulator's own
    backbone, and a model reached through an API exposes none to read, so
    ``style_gate_preferred`` belongs with the simulator or with neither.
    """

    reward_table_path: str                 # move quality per (position, move)
    reward_simulator_path: str = ""        # frozen Stage-1 chess sim
    heads_checkpoint_path: str = ""        # the style and perception heads
    heads_metrics_path: str = ""           # per-class F1, which weights the gate
    student_model: str = ""                # a model prompted to play the student
    reward_device: str = "cuda:0"          # the GPU GpuLayout reserves for it
    style_gate_preferred: str | None = None
    alpha_style: float = 2.0               # how sharply the style gate decays
    alpha_perception: float = 3.0          # how sharply the perception gate decays
    mate_clip_cp: int = 1500
    scale_cp: int = 500
    illegal_reward: float = -1.0

    def __post_init__(self) -> None:
        if bool(self.student_model) == bool(self.reward_simulator_path):
            raise ValueError(
                "the reward needs exactly one student: set reward_simulator_path "
                "for a trained simulator or student_model for a prompted one"
            )
        if self.student_model:
            if self.style_gate_preferred:
                raise ValueError(
                    "style_gate_preferred needs a trained simulator to read; a "
                    "prompted student exposes no backbone for the head"
                )
            if self.heads_checkpoint_path or self.heads_metrics_path:
                raise ValueError("heads belong to a trained simulator, not a prompted one")
        elif not (self.heads_checkpoint_path and self.heads_metrics_path):
            raise ValueError("a trained simulator needs both head files to gate with")


@dataclass(frozen=True, slots=True)
class TutorRlConfig:
    """One run: what it trains with, and what it is rewarded by.

    Loaded from a YAML under ``configs/tutor_rl/``.
    """

    training: RLConfig
    reward: RewardConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> TutorRlConfig:
        with Path(path).open() as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TutorRlConfig:
        data = dict(raw)
        training_raw = dict(data.get("training", {}))
        if "gpu_layout" in training_raw and isinstance(training_raw["gpu_layout"], dict):
            training_raw["gpu_layout"] = GpuLayout(**training_raw["gpu_layout"])
        if "actor_lora_target_modules" in training_raw:
            training_raw["actor_lora_target_modules"] = tuple(
                training_raw["actor_lora_target_modules"]
            )
        training = RLConfig(**training_raw)
        reward = RewardConfig(**data.get("reward", {}))
        return cls(training=training, reward=reward)

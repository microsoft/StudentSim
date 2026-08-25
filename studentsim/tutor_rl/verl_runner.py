"""verl launcher.

Builds the verl training command from :class:`RLConfig` + :class:`RewardConfig`
and (optionally) spawns it via subprocess. The launcher is the chess-tutor
analogue of :func:`studentsim.training.ms_swift.build_ms_swift_command` for SFT.

The launcher runs on a single node, split as :class:`GpuLayout` says; a cluster
needs this wrapped in whatever submits its jobs.

The overrides below are written against verl 0.8-0.9, which is where each key
sits today. verl has moved several of them between releases -- LoRA from the
actor to the model, the reward function under a ``reward`` section -- so an
older verl needs the names adjusted, which is what ``extra_overrides`` is for.
verl also builds its models with FlashAttention 2 by default, so a box without
that package wants
``actor_rollout_ref.model.override_config.attn_implementation=sdpa``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from studentsim.tutor_rl.rl_config import RewardConfig, RLConfig

_REWARD = "reward.custom_reward_function"
"""Where verl keeps the custom reward function in its config tree."""

RunnerCallable = Callable[[Sequence[str]], int]


def _default_runner(argv: Sequence[str]) -> int:
    # verl asks Ray for no GPUs on its reward workers, on the assumption that
    # scoring a rollout is arithmetic. Ours runs the student simulator, so it
    # needs the device GpuLayout set aside for it, and Ray hides every GPU from
    # a worker that asked for none unless told otherwise.
    env = {**os.environ, "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
    return subprocess.run(list(argv), check=False, env=env).returncode


@dataclass(frozen=True, slots=True)
class VerlCommand:
    """A fully materialized verl command."""

    argv: Sequence[str]

    def shell(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


def build_verl_command(
    *,
    training: RLConfig,
    reward: RewardConfig,
    reward_entry_module: str | None = None,
    extra_overrides: Sequence[str] = (),
) -> VerlCommand:
    """Construct the verl command corresponding to (``training``, ``reward``).

    ``reward_entry_module`` is a Python dotted path to a module exporting a
    ``compute_score(...)`` function, which verl imports at training time. Left
    unset, it follows the reward: a trained simulator goes to
    :mod:`studentsim.tutor_rl._reward_entry` and a prompted student to
    :mod:`studentsim.tutor_rl._prompted_reward_entry`.

    ``extra_overrides`` are appended last and so win over everything above.
    They are for what the machine dictates rather than the experiment -- an
    attention implementation the box can actually run, a verl release that
    spells a key differently.
    """
    argv: list[str] = [
        "python", "-m", "verl.trainer.main_ppo",
    ]

    # Data / model paths.
    argv += [f"data.train_files={training.playground_train}"]
    argv += [f"data.val_files={training.playground_val}"]
    argv += [f"actor_rollout_ref.model.path={training.actor_sft_base}"]
    argv += [f"trainer.default_local_dir={training.output_dir}"]

    # Batch sizes and the optimisation settings verl reads them with.
    argv += [f"data.train_batch_size={training.train_batch_size}"]
    argv += [f"actor_rollout_ref.actor.ppo_mini_batch_size={training.ppo_mini_batch_size}"]
    micro = training.ppo_micro_batch_size_per_gpu
    argv += [f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro}"]
    # The rollout and reference passes score sequences they did not generate,
    # and verl asks for their batch size separately from the update's.
    argv += [f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro}"]
    argv += [f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro}"]
    argv += [f"algorithm.adv_estimator={training.advantage_estimator}"]
    argv += [f"algorithm.use_kl_in_reward={str(training.use_kl_in_reward).lower()}"]
    argv += [f"actor_rollout_ref.actor.use_kl_loss={str(training.use_kl_loss).lower()}"]
    if training.use_kl_loss:
        argv += [f"actor_rollout_ref.actor.kl_loss_type={training.kl_loss_type}"]
        argv += [f"actor_rollout_ref.actor.kl_loss_coef={training.kl_loss_coef}"]
    argv += [f"actor_rollout_ref.actor.entropy_coeff={training.entropy_coeff}"]

    # Optimizers.
    argv += [f"actor_rollout_ref.actor.optim.lr={training.actor_learning_rate}"]

    # Rollout.
    argv += [f"actor_rollout_ref.rollout.name={training.rollout_engine}"]
    argv += [f"actor_rollout_ref.rollout.n={training.rollouts_per_prompt}"]
    argv += [f"actor_rollout_ref.rollout.temperature={training.rollout_temperature}"]
    argv += [f"actor_rollout_ref.rollout.top_p={training.rollout_top_p}"]
    argv += [f"actor_rollout_ref.rollout.top_k={training.rollout_top_k}"]
    argv += [f"data.max_prompt_length={training.max_prompt_length}"]
    argv += [f"data.max_response_length={training.max_response_length}"]

    # LoRA on the actor, when there is any. The policy starts from the merged
    # tutor SFT checkpoint, and training it in full is what verl spells as
    # rank 0; the adapter settings are then left unsaid rather than sent as
    # zeros. These belong to the model rather than the optimizer that trains
    # it, which is where verl keeps them.
    argv += [f"actor_rollout_ref.model.lora_rank={training.actor_lora_rank}"]
    if training.actor_lora_rank:
        argv += [f"actor_rollout_ref.model.lora_alpha={training.actor_lora_alpha}"]
        argv += [
            "actor_rollout_ref.model.target_modules=["
            + ",".join(training.actor_lora_target_modules)
            + "]"
        ]

    # Training length.
    argv += [f"trainer.total_training_steps={training.total_steps}"]
    argv += [f"trainer.save_freq={training.save_freq}"]
    argv += [f"trainer.test_freq={training.test_freq}"]
    # The seed verl exposes is the one that orders the data.
    argv += [f"data.seed={training.seed}"]

    # verl logs to a hosted tracker as well as the console by default, and that
    # fails outright without an account for it. Other trackers go back through
    # --set trainer.logger=[...].
    argv += ["trainer.logger=[console]"]

    # GPU layout.
    argv += [f"trainer.n_gpus_per_node={training.gpu_layout.n_trainer_gpus}"]
    argv += ["trainer.nnodes=1"]

    # Reward function.
    # Unprefixed, verl reads the path as a file on disk and cannot find it; the
    # prefix is what tells it to import an installed module instead.
    # A reward worker holding a student simulator needs a GPU to itself, so the
    # count follows the GPUs reserved for it; verl defaults to eight, which would
    # put eight simulators on one card. A prompted student holds no simulator and
    # reserves no GPU, and then this is one worker making the calls in turn.
    argv += [f"reward.num_workers={max(1, training.gpu_layout.n_reward_sim_gpus)}"]
    entry = reward_entry_module or (
        "studentsim.tutor_rl._prompted_reward_entry" if reward.student_model
        else "studentsim.tutor_rl._reward_entry"
    )
    argv += [f"{_REWARD}.path=pkg://{entry}"]
    argv += [f"{_REWARD}.name=compute_score"]
    # These names are the ones compute_score reads; they have to match it
    # exactly, since verl passes them straight through as keyword arguments.
    # verl declares no keys under reward_kwargs, so each is appended rather
    # than overridden.
    kwargs = {
        "reward_table": reward.reward_table_path,
        "mate_clip_cp": reward.mate_clip_cp,
        "scale_cp": reward.scale_cp,
        "illegal_reward": reward.illegal_reward,
    }
    if reward.student_model:
        # A prompted student needs no GPU and has no backbone to gate on, so
        # neither the device nor the head files travel with it.
        kwargs["student_model"] = reward.student_model
    else:
        kwargs.update(
            adapter_path=reward.reward_simulator_path,
            heads_checkpoint=reward.heads_checkpoint_path,
            heads_metrics=reward.heads_metrics_path,
            device=reward.reward_device,
            # Both gates read their decay rate from here.
            alpha_style=reward.alpha_style,
            alpha_perception=reward.alpha_perception,
        )
        if reward.style_gate_preferred:
            kwargs["style_preferred"] = reward.style_gate_preferred
    argv += [f"+{_REWARD}.reward_kwargs.{name}={value}" for name, value in kwargs.items()]

    argv += list(extra_overrides)
    return VerlCommand(argv=tuple(argv))


@dataclass
class VerlRunner:
    """Build and execute a verl command.

    Production usage::

        runner = VerlRunner(training=..., reward=...)
        exit_code = runner.run()

    Tests inject a fake ``runner`` to avoid spawning verl::

        runner = VerlRunner(training=..., reward=..., runner=lambda argv: 0)
    """

    training: RLConfig
    reward: RewardConfig
    runner: RunnerCallable = _default_runner
    reward_entry_module: str | None = None
    extra_overrides: tuple[str, ...] = ()

    def build_command(self) -> VerlCommand:
        return build_verl_command(
            training=self.training,
            reward=self.reward,
            reward_entry_module=self.reward_entry_module,
            extra_overrides=self.extra_overrides,
        )

    def run(self) -> int:
        command = self.build_command()
        code = self.runner(command.argv)
        if code != 0:
            raise RuntimeError(
                f"verl training failed (exit {code}); command was: {command.shell()}"
            )
        return code

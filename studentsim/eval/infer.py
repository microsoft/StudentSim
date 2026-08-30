"""Batched decoding through the ms-swift CLI.

Evaluation decodes with ``swift infer`` against the base model plus a trained
adapter, writing one result row per input record. Shuffling is off on both the
dataset and the validation dataset so that result row *i* corresponds to input
row *i*; the per-mode breakdown in :mod:`studentsim.eval.scoring` relies on
that alignment.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from studentsim.core.paths import swift_bin

RunnerCallable = Callable[[Sequence[str]], int]


def _default_runner(argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), check=False).returncode


@dataclass(frozen=True)
class InferCommand:
    """A fully materialized ``swift infer`` invocation."""

    argv: tuple[str, ...]

    def shell(self) -> str:
        import shlex

        return " ".join(shlex.quote(a) for a in self.argv)


def build_infer_command(
    *,
    base_model: str,
    adapter_dir: str | Path,
    dataset: str | Path,
    result_path: str | Path,
    max_new_tokens: int,
    max_batch_size: int,
    repetition_penalty: float = 1.0,
    temperature: float = 0.0,
    max_samples: int = 0,
    swift_binary: str | None = None,
) -> InferCommand:
    """Construct the ``swift infer`` command for one evaluation pass."""
    argv: list[str] = [swift_binary or swift_bin(), "infer"]
    argv += ["--model", str(base_model)]
    argv += ["--ckpt_dir", str(adapter_dir)]
    argv += ["--infer_backend", "pt"]
    argv += ["--max_new_tokens", str(max_new_tokens)]
    argv += ["--temperature", str(temperature)]
    argv += ["--repetition_penalty", str(repetition_penalty)]
    argv += ["--stream", "false"]
    argv += ["--val_dataset", str(dataset)]
    argv += ["--dataset_shuffle", "false"]
    argv += ["--val_dataset_shuffle", "false"]
    argv += ["--max_batch_size", str(max_batch_size)]
    argv += ["--result_path", str(result_path)]
    if max_samples > 0:
        argv += ["--val_dataset_sample", str(max_samples)]
    return InferCommand(argv=tuple(argv))


def run_infer(command: InferCommand, *, runner: RunnerCallable = _default_runner) -> None:
    """Run ``command``, raising if it fails or writes nothing."""
    code = runner(command.argv)
    if code != 0:
        raise RuntimeError(f"swift infer failed (exit {code}); command was: {command.shell()}")

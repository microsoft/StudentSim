"""Per-student driver: loops Stage-2 over a roster of student ids.

One run per student, in roster order, the same way in all three domains. What
differs between domains is the roster and the :class:`TrainingConfig` built for
each student, both of which are handed in.

A :class:`PerStudentDriver` is constructed with the Stage-1 adapter path and
a callable that materializes a :class:`TrainingConfig` for each student;
:meth:`run_all` iterates the roster and dispatches to
:class:`Stage2Trainer`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from studentsim.training.config import TrainingConfig
from studentsim.training.stage1 import RunnerCallable, _default_runner
from studentsim.training.stage2 import Stage2Trainer

ConfigBuilder = Callable[[str], TrainingConfig]
"""Callable returning a stage-2 :class:`TrainingConfig` for one student id."""

logger = logging.getLogger(__name__)


@dataclass
class PerStudentDriver:
    """Iterates Stage-2 training over the Stage-2 student roster.

    Parameters
    ----------
    config_builder
        Callable that, given a ``student_id``, returns a fully-populated
        :class:`TrainingConfig` for that student. The caller is responsible
        for setting per-student paths (``data_path``, ``output_dir``).
    runner
        Subprocess runner; defaults to ``subprocess.run``. Tests inject fakes.
    swift_binary
        Path to the ``swift`` binary; ``None`` defers to env-var lookup.
    """

    roster: Sequence[str]
    config_builder: ConfigBuilder
    runner: RunnerCallable = _default_runner
    swift_binary: str | None = None
    completed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    def run_all(self, *, fail_fast: bool = False) -> None:
        """Train every Stage-2 student in roster order.

        Parameters
        ----------
        fail_fast
            If ``True``, raise on the first failed student. Otherwise log and
            continue; failures are recorded in :attr:`failed`.
        """
        roster = list(self.roster)
        if not roster:
            logger.warning("Stage-2 roster is empty; nothing to train.")
            return
        for student_id in roster:
            try:
                self._run_one(student_id)
                self.completed.append(student_id)
                logger.info("Stage-2 done: %s", student_id)
            except Exception as e:
                logger.error("Stage-2 failed for %s: %s", student_id, e)
                self.failed[student_id] = str(e)
                if fail_fast:
                    raise

    def _run_one(self, student_id: str) -> None:
        config = self.config_builder(student_id)
        trainer = Stage2Trainer(
            config=config,
            student_id=student_id,
            swift_binary=self.swift_binary,
            runner=self.runner,
        )
        trainer.run()

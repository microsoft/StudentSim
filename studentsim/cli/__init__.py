"""Command-line entry points.

Each module exposes a ``main(argv: Sequence[str] | None = None) -> int`` that
is wired to a ``studentsim-<x>`` console script in ``pyproject.toml``.

Subcommands
-----------
- :mod:`studentsim.cli.train` -- run Stage 1 or Stage 2 training
- :mod:`studentsim.cli.eval` -- score a trained simulator on both metrics
- :mod:`studentsim.cli.tutor_rl` -- launch the chess tutor RL run
"""

__all__: list[str] = []

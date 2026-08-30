"""Centralized environment-variable-driven path resolution.

Every absolute path is resolved through this module rather than written into
the code that needs it, so a different filesystem layout is a matter of setting
a few environment variables rather than editing files.

Schema:

============================  =====================================================  ==================================================
Variable                       Purpose                                                Default (local development)
============================  =====================================================  ==================================================
STUDENTSIM_DATA_DIR            Root for the built records and source-corpus extracts   ``./data``
STUDENTSIM_CKPT_DIR            Root for trained LoRA adapters and merged checkpoints  ``./checkpoints``
STUDENTSIM_MODEL_CACHE         HuggingFace / ModelScope model cache                   ``~/.cache/studentsim/models``
STUDENTSIM_RUN_DIR             Per-run output (logs, predictions, metrics)            ``./runs``
STUDENTSIM_MS_SWIFT_BIN        ms-swift ``swift`` binary path                         ``shutil.which("swift")``
STUDENTSIM_STOCKFISH_BIN       Stockfish binary path                                  ``shutil.which("stockfish")``
AZURE_OPENAI_ENDPOINT          Azure OpenAI endpoint (e.g., for GPT-5.4 baseline)     ``None`` (required when calling Azure backends)
AZURE_OPENAI_API_KEY           Azure OpenAI API key                                   ``None`` (required when calling Azure backends)
============================  =====================================================  ==================================================

Absolute paths must not appear in any code file. If you catch one, replace it
with a call to a function defined here.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Final

_DEFAULT_DATA_DIR: Final = Path("data")
_DEFAULT_CKPT_DIR: Final = Path("checkpoints")
_DEFAULT_RUN_DIR: Final = Path("runs")
_DEFAULT_MODEL_CACHE: Final = Path.home() / ".cache" / "studentsim" / "models"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def data_dir() -> Path:
    """Root directory for the built records and source-corpus extracts."""
    return _env_path("STUDENTSIM_DATA_DIR", _DEFAULT_DATA_DIR)


def ckpt_dir() -> Path:
    """Root directory for trained LoRA adapters and merged checkpoints."""
    return _env_path("STUDENTSIM_CKPT_DIR", _DEFAULT_CKPT_DIR)


def model_cache() -> Path:
    """HuggingFace / ModelScope model cache."""
    return _env_path("STUDENTSIM_MODEL_CACHE", _DEFAULT_MODEL_CACHE)


def run_dir() -> Path:
    """Per-run output (logs, predictions, metrics)."""
    return _env_path("STUDENTSIM_RUN_DIR", _DEFAULT_RUN_DIR)


def domain_data_dir(domain: str) -> Path:
    """Per-domain subdirectory of the data root."""
    return data_dir() / domain


def stage_ckpt_dir(domain: str, stage: int, student_id: str | None = None) -> Path:
    """Where Stage-1 or Stage-2 adapters land.

    Stage-1 adapters are one per domain (e.g., ``checkpoints/chess/stage1``); Stage-2
    adapters are per-student under ``checkpoints/<domain>/stage2/<student_id>``.
    """
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    root = ckpt_dir() / domain / f"stage{stage}"
    return root / student_id if student_id else root


def swift_bin() -> str:
    """Locate the ``swift`` CLI binary."""
    env = os.environ.get("STUDENTSIM_MS_SWIFT_BIN")
    if env:
        return env
    resolved = shutil.which("swift")
    if not resolved:
        raise FileNotFoundError(
            "Could not locate the ms-swift CLI. Set STUDENTSIM_MS_SWIFT_BIN or install ms-swift."
        )
    return resolved


def stockfish_bin() -> str:
    """Locate the Stockfish binary."""
    env = os.environ.get("STUDENTSIM_STOCKFISH_BIN")
    if env:
        return env
    resolved = shutil.which("stockfish")
    if not resolved:
        raise FileNotFoundError(
            "Could not locate the Stockfish binary. Set STUDENTSIM_STOCKFISH_BIN or install stockfish."
        )
    return resolved


def require_env(name: str) -> str:
    """Read an env var, raising if absent. For credentials that must be set explicitly."""
    value = os.environ.get(name)
    if not value:
        raise OSError(
            f"Environment variable {name!r} is required for this operation. "
            f"See studentsim.core.paths for the full env-var schema."
        )
    return value

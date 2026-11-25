"""Scoring normalizers, one per response space.

Both the prediction and the reference answer pass through the same normalizer
before they are compared, so a score is an exact match between two normalized
strings. Chess reads a UCI move out of the decoded text; L2 compares answer
text after unescaping and collapsing whitespace; math drops reasoning blocks,
edge punctuation and markup and compares lowercased text.

``strip_think`` handles decodes that open a ``<think>`` block: only the text
after the closing tag is searched, so a move or answer mentioned inside the
reasoning trace cannot shadow the final one. Chess scores with it on, L2 with
it off. The math normalizer removes every complete ``<think>`` block whatever
the flag says.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Final

UCI_PATTERN: Final = re.compile(r"\b([a-h][1-8][a-h][1-8])\b")
THINK_END_TAG: Final = re.compile(r"</think>\s*", re.DOTALL)
THINK_BLOCK: Final = re.compile(r"<think>.*?</think>", re.DOTALL)
MARKUP: Final = re.compile(r"<[^>]+>")
EDGE_PUNCTUATION: Final = " .,;:!?"
WHITESPACE_RE: Final = re.compile(r"\s+")

_DELETE_FORMS: Final = ("(delete)", "( delete )", "( delete)", "(delete )")

Normalizer = Callable[[str, bool], str]


def _after_think(text: str, strip_think: bool) -> str:
    if not strip_think:
        return text
    tag = THINK_END_TAG.search(text)
    return text[tag.end() :] if tag else text


def normalize_move(pred: str, strip_think: bool = False) -> str:
    """Extract a UCI move from a chess decode.

    Falls back to the whole (lowercased, stripped) text when no move-shaped
    substring is present, so an unparseable decode scores as a miss rather
    than as an empty string that could match another empty one.
    """
    pred = str(pred or "").strip().lower()
    match = UCI_PATTERN.search(_after_think(pred, strip_think))
    return match.group(1) if match else pred


def normalize_answer(pred: str, strip_think: bool = False) -> str:
    """Normalize an L2 span correction.

    The delete-marker rule is what lets an L2 correction match: the reference
    answer for a deletion is written ``(delete)`` while the simulator reliably
    emits a bare ``delete``.
    """
    text = _after_think(str(pred or ""), strip_think)
    text = html.unescape(text).strip().lower()
    text = WHITESPACE_RE.sub(" ", text)
    return "delete" if text in _DELETE_FORMS else text


def normalize_math_answer(pred: str, strip_think: bool = True) -> str:
    """Normalize a math answer, which is a letter or a short free-text answer.

    Complete ``<think>`` blocks are removed, then surrounding spaces and the
    punctuation in :data:`EDGE_PUNCTUATION`, then markup tags, and the result
    is lowercased. An empty result never counts as a match; the scorer checks
    that, not this function.
    """
    if not pred:
        return ""
    text = THINK_BLOCK.sub("", str(pred)).strip()
    text = text.strip(EDGE_PUNCTUATION)
    text = MARKUP.sub("", text).strip()
    return text.lower()


def normalizer_for(domain: str) -> Normalizer:
    """Return the normalizer that scores ``domain``."""
    if domain == "chess":
        return normalize_move
    if domain == "l2":
        return normalize_answer
    if domain == "math":
        return normalize_math_answer
    raise ValueError(f"unknown domain: {domain!r}")

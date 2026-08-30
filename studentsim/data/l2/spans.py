"""Correction spans in the EFCAMDAT error-coded subcorpus.

Every teacher correction in the ``text`` column is one ``<change>`` element
carrying what the learner wrote, what it should have been, and an error
category. Each one becomes a multi-turn record, and their categories, counted
over a learner's earlier essays, become the error profile in that learner's
prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CHANGE_RE = re.compile(
    r"<change>"
    r"<selection>(?P<sel>.*?)</selection>"
    r"<tag><symbol>(?P<sym>[^<]*)</symbol><correct>(?P<cor>.*?)</correct></tag>"
    r"</change>",
    re.DOTALL,
)

#: The fifteen EFCAMDAT error categories, and how the prompts name them.
SYMBOL_LABELS = {
    "SP": "spelling",
    "WC": "word choice",
    "D": "extra word (delete)",
    "MW": "missing word",
    "AR": "article",
    "PR": "preposition",
    "VT": "verb tense",
    "PL": "plural",
    "AG": "agreement",
    "IS": "inflection / suffix",
    "WO": "word order",
    "XC": "collocation / lexical",
    "EX": "expression / idiom",
    "CO": "connector",
    "SI": "syntax",
}

#: How much of the surrounding essay each span carries, in characters.
CONTEXT_CHARS = 60


@dataclass(frozen=True, slots=True)
class Span:
    """One teacher correction, with enough context to quote it back."""

    selection: str
    """What the learner wrote. Empty for a missing-word correction."""

    symbol: str
    """One of :data:`SYMBOL_LABELS`."""

    correct: str
    """What it should have been. Empty for a delete correction."""

    before: str
    after: str
    """Up to :data:`CONTEXT_CHARS` of the essay on either side."""


def extract_spans(text: str) -> list[Span]:
    """Pull every correction out of one tagged essay, in reading order.

    Context is measured against the learner's own text, so a span late in a
    heavily corrected essay still quotes the words around it rather than the
    markup.
    """
    plain_parts: list[str] = []
    pending: list[tuple[str, str, str, int]] = []
    pos = 0

    for m in CHANGE_RE.finditer(text):
        plain_parts.append(text[pos : m.start()])
        selection = m.group("sel") or ""
        plain_parts.append(selection)
        pending.append(
            (
                selection,
                (m.group("sym") or "").strip(),
                m.group("cor") or "",
                len("".join(plain_parts)),
            )
        )
        pos = m.end()

    plain_parts.append(text[pos:])
    plain = "".join(plain_parts)

    spans = []
    for selection, symbol, correct, end in pending:
        start = end - len(selection)
        spans.append(
            Span(
                selection=selection,
                symbol=symbol,
                correct=correct,
                before=plain[max(0, start - CONTEXT_CHARS) : start],
                after=plain[end : end + CONTEXT_CHARS],
            )
        )
    return spans

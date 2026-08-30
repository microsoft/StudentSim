"""Production LanguageTool wrapper.

Uses the ``language-tool-python`` package, which embeds a LanguageTool JVM /
server. The mapping from LanguageTool's per-match ``ruleIssueType`` field to
the seven buckets is in :data:`_RULE_ISSUE_TYPE_TO_BUCKET`.

Importing this module is optional: users who do not run L2 evaluation never
need a JVM. The import is lazy.
"""

from __future__ import annotations

import re
from typing import Final

from studentsim.domains.l2.fidelity import LT_BUCKETS, IssueCounts

# The seven buckets are the LanguageTool issue types themselves; issue types
# outside the taxonomy are not counted, and so contribute to neither a
# category density nor the total.
_WORD_RE: Final = re.compile(r"\b\w+\b")

# language-tool-python renamed this attribute between releases. Reading the
# wrong name would silently score every text as error-free.
_ISSUE_TYPE_ATTRS: Final = ("rule_issue_type", "ruleIssueType")


def _issue_type(match: object) -> str:
    """The LanguageTool issue type of one match, or ``""`` if unavailable."""
    for attr in _ISSUE_TYPE_ATTRS:
        value = getattr(match, attr, None)
        if value:
            return str(value)
    raise AttributeError(
        "LanguageTool match exposes no issue type under "
        f"{_ISSUE_TYPE_ATTRS}; L2 fidelity cannot be scored"
    )


class LanguageToolCounter:
    """Production :class:`IssueCounter` impl backed by ``language-tool-python``.

    Parameters
    ----------
    locale
        LanguageTool locale, default ``"en-US"``. The L2 corpus is in
        English; other locales are accepted but untested here.
    """

    def __init__(self, *, locale: str = "en-US") -> None:
        # Lazy import so users who never run L2 evaluation don't need a JVM.
        import language_tool_python

        self._lt = language_tool_python.LanguageTool(locale)

    def count(self, text: str) -> IssueCounts:
        word_count = len(_WORD_RE.findall(text or ""))
        per_bucket: dict[str, int] = {b: 0 for b in LT_BUCKETS}
        if not text.strip():
            return IssueCounts(per_bucket=per_bucket, word_count=word_count)
        for match in self._lt.check(text):
            issue_type = _issue_type(match)
            if issue_type in per_bucket:
                per_bucket[issue_type] += 1
        return IssueCounts(per_bucket=per_bucket, word_count=word_count)

    def close(self) -> None:
        """Release the LanguageTool server process."""
        self._lt.close()

    def __enter__(self) -> LanguageToolCounter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

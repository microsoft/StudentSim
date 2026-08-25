"""Repairing the answer key before anything reads it.

A spreadsheet along the way coerced fractions into dates, so a problem whose
answer is ``5/6`` arrives as ``6-May``. The mapping back is exact, and it runs
before selection, splitting, or record building so that no later step sees a
corrupted key.
"""

from __future__ import annotations

import re

_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

_DAY_MONTH = re.compile(r"^(\d+)-(" + "|".join(_MONTHS) + r")$")
_MONTH_DAY = re.compile(r"^(" + "|".join(_MONTHS) + r")-(\d+)$")


def repair_fraction(value: str) -> str | None:
    """The fraction a date-shaped answer came from, or ``None`` if it is fine.

    ``6-May`` is how a spreadsheet stores ``5/6``, and ``Jan-40`` is how it
    stores ``1/40``.
    """
    text = (value or "").strip()
    match = _DAY_MONTH.match(text)
    if match:
        return f"{_MONTHS[match.group(2)]}/{match.group(1)}"
    match = _MONTH_DAY.match(text)
    if match:
        return f"{_MONTHS[match.group(1)]}/{match.group(2)}"
    return None

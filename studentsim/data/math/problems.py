"""Turning a problem as the platform stores it into a problem a model can read.

Problem bodies arrive as HTML with MathML inside. Fractions become ``a/b`` and
everything else is flattened to text, with block tags kept as line breaks so
that a multi-part question does not collapse into one line.
"""

from __future__ import annotations

import html
import re

from bs4 import BeautifulSoup

_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"}

#: What the platform expects the answer to look like, said in words for the
#: simulator.
ANSWER_TYPE_HINTS = {
    "Numeric": "Answer with a number.",
    "Algebraic Expression": "Answer with an algebraic expression.",
    "Exact Match": "Answer with the exact text.",
    "Exact Fraction": "Answer with a fraction.",
    "Numeric Expression": "Answer with a numeric expression.",
    "Ordering": "Answer by listing items in order, comma-separated.",
    "Drop Down": "Answer with one of the dropdown options.",
}


def clean_text(markup: str) -> str:
    """Plain text for a problem body, keeping its paragraph structure."""
    if not markup:
        return ""
    soup = BeautifulSoup(markup, "html.parser")

    for math in soup.find_all("math"):
        fraction = math.find("mfrac")
        if fraction:
            numbers = fraction.find_all("mn")
            if len(numbers) == 2:
                math.replace_with(f"{numbers[0].text}/{numbers[1].text}")
                continue
        math.replace_with(math.get_text())

    for tag in soup.find_all(True):
        if tag.name in _BLOCK_TAGS:
            tag.insert_before("\n")
            tag.insert_after("\n")

    for image in soup.find_all("img"):
        image.replace_with(image.get("alt", ""))

    text = html.unescape(soup.get_text(separator=" "))
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    return re.sub(r"[ \t]{2,}", " ", text).strip()

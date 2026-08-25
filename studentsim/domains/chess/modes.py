"""Chess guidance modes.

Four pedagogical styles, one per tutor message in the multi-turn corpus:

============================  ==================================================================
Mode                          What the tutor message says
============================  ==================================================================
``error_remediation``         Names the mistake and states the correction explicitly.
``comparative``               Contrasts the played move with the engine's top alternatives.
``strategic``                 Foregrounds long-term plan considerations (no explicit correction).
``socratic``                  Asks one or two leading questions and withholds the answer.
============================  ==================================================================

The ``CHESS_MODES`` tuple fixes the iteration order used by
the chess mode list. Per-mode breakdown
per-mode tables render columns in this order.

The names appear verbatim in the multi-turn record schema and as the style
head's labels, so they are load-bearing identifiers rather than display text;
renaming one silently stops matching the records that carry it.
"""

from __future__ import annotations

from typing import Final

from studentsim.core.records import GuidanceMode

CHESS_DOMAIN_NAME: Final = "chess"

ERROR_REMEDIATION: Final = GuidanceMode(name="error_remediation", domain=CHESS_DOMAIN_NAME)
COMPARATIVE: Final = GuidanceMode(name="comparative", domain=CHESS_DOMAIN_NAME)
STRATEGIC: Final = GuidanceMode(name="strategic", domain=CHESS_DOMAIN_NAME)
SOCRATIC: Final = GuidanceMode(name="socratic", domain=CHESS_DOMAIN_NAME)

CHESS_MODES: Final[tuple[GuidanceMode, ...]] = (
    ERROR_REMEDIATION,
    COMPARATIVE,
    STRATEGIC,
    SOCRATIC,
)

CHESS_MODE_NAMES: Final[tuple[str, ...]] = tuple(m.name for m in CHESS_MODES)



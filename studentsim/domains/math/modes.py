"""Math guidance modes.

Three pedagogical styles:

================  =====================================================
Mode               What the tutor message says
================  =====================================================
``error_remediation``  Pinpoints the mistake and gives the correct
                       answer with a short explanation.
``socratic``           Asks one or two guiding questions; withholds the answer.
``conceptual``         Connects the present error to a pattern of past
                       same-skill errors and teaches the principle without
                       giving the answer.
================  =====================================================

The order here is error remediation, socratic, conceptual; per-mode
tables render columns in this order.
"""

from __future__ import annotations

from typing import Final

from studentsim.core.records import GuidanceMode

MATH_DOMAIN_NAME: Final = "math"

ERROR_REMEDIATION: Final = GuidanceMode(name="error_remediation", domain=MATH_DOMAIN_NAME)
SOCRATIC: Final = GuidanceMode(name="socratic", domain=MATH_DOMAIN_NAME)
CONCEPTUAL: Final = GuidanceMode(name="conceptual", domain=MATH_DOMAIN_NAME)

MATH_MODES: Final[tuple[GuidanceMode, ...]] = (
    ERROR_REMEDIATION,
    SOCRATIC,
    CONCEPTUAL,
)
MATH_MODE_NAMES: Final[tuple[str, ...]] = tuple(m.name for m in MATH_MODES)



"""L2 guidance modes.

Two surface-form templates for the tutor message:

================  ============================================================
Mode               What the tutor message looks like
================  ============================================================
``point_based``    States that an error of a given kind is present in the
                   highlighted span and asks the student to fix it.
``rule_based``     States the underlying linguistic rule first and then asks
                   the student to apply it to the highlighted span.
================  ============================================================

The independent EFCAMDAT-side error category (grammar / lexical / structural)
is a *property of the span*, not the tutor message surface; it lives on the
multi-turn record's ``meta`` and powers :meth:`L2Guidance.per_category_score`.
"""

from __future__ import annotations

from typing import Final

from studentsim.core.records import GuidanceMode

L2_DOMAIN_NAME: Final = "l2"

POINT_BASED: Final = GuidanceMode(name="point_based", domain=L2_DOMAIN_NAME)
RULE_BASED: Final = GuidanceMode(name="rule_based", domain=L2_DOMAIN_NAME)

L2_MODES: Final[tuple[GuidanceMode, ...]] = (POINT_BASED, RULE_BASED)
L2_MODE_NAMES: Final[tuple[str, ...]] = tuple(m.name for m in L2_MODES)


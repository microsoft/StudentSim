"""Second-language English writing domain (EFCAMDAT).

Source corpus: EFCAMDAT second release, accessed under the Cambridge Research
Lab academic user agreement. Response space: short essay text (~50-150 words).
Guidance modes: point-based and rule-based correction templates.
"""

from studentsim.domains.l2.fidelity import (
    LT_BUCKETS,
    IssueCounter,
    IssueCounts,
    L2Fidelity,
)
from studentsim.domains.l2.modes import (
    L2_DOMAIN_NAME,
    L2_MODE_NAMES,
    L2_MODES,
    POINT_BASED,
    RULE_BASED,
)

__all__ = [
    "IssueCounter",
    "IssueCounts",
    "L2Fidelity",
    "L2_DOMAIN_NAME",
    "L2_MODES",
    "L2_MODE_NAMES",
    "LT_BUCKETS",
    "POINT_BASED",
    "RULE_BASED",
]

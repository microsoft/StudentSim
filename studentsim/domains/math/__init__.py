"""Mathematics domain (FoundationalASSIST / ASSISTments).

Source corpus: FoundationalASSIST 6-8 grade Illustrative Mathematics records,
accessed under the upstream Hugging Face Responsible Use Agreement. Response
space: a single letter A / B / C / D over a deterministic four-way multiple-
choice form. Guidance modes: error remediation, socratic, conceptual.
"""

from studentsim.domains.math.fidelity import (
    MATH_LETTERS,
    MathFidelity,
    argmax_letter,
    renormalized_log_likelihood,
)
from studentsim.domains.math.modes import (
    CONCEPTUAL,
    ERROR_REMEDIATION,
    MATH_DOMAIN_NAME,
    MATH_MODE_NAMES,
    MATH_MODES,
    SOCRATIC,
)

__all__ = [
    "CONCEPTUAL",
    "ERROR_REMEDIATION",
    "MATH_DOMAIN_NAME",
    "MATH_LETTERS",
    "MATH_MODES",
    "MATH_MODE_NAMES",
    "MathFidelity",
    "SOCRATIC",
    "argmax_letter",
    "renormalized_log_likelihood",
]

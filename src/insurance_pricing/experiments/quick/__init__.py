from __future__ import annotations

import warnings

from . import dualtrack, gap_diagnosis, tail_recovery, tail_selection

warnings.warn(
    "insurance_pricing.experiments.quick is archived (read-only) and "
    "kept only for transitional compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "gap_diagnosis",
    "dualtrack",
    "tail_recovery",
    "tail_selection",
]

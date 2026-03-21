from __future__ import annotations

import warnings

warnings.warn(
    "insurance_pricing.experiments.quick is archived (read-only) and "
    "kept only for transitional compatibility.",
    DeprecationWarning,
    stacklevel=2,
)

from . import dualtrack, gap_diagnosis, tail_recovery, tail_selection  # noqa: E402

__all__ = [
    "gap_diagnosis",
    "dualtrack",
    "tail_recovery",
    "tail_selection",
]

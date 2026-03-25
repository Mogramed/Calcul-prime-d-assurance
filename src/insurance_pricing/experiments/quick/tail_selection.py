from __future__ import annotations

import warnings
from typing import Any

from insurance_pricing.legacy.quick import tail_selection_impl as _impl

warnings.warn(
    "insurance_pricing.experiments.quick.tail_selection is archived "
    "(read-only) and should not be used for new development.",
    DeprecationWarning,
    stacklevel=2,
)


def train_run(config_path: str) -> dict[str, Any]:
    return _impl.train_run(config_path)


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = getattr(_impl, "__all__", [n for n in dir(_impl) if not n.startswith("_")])

from __future__ import annotations

from typing import Any, cast

from insurance_pricing.legacy import training_fulltrain_impl as _impl


def fit_full_predict(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return cast(tuple[Any, ...], _impl.fit_full_predict(*args, **kwargs))


def fit_full_predict_fulltrain(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return cast(tuple[Any, ...], _impl.fit_full_predict_fulltrain(*args, **kwargs))


def fit_full_two_part_predict(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return cast(tuple[Any, ...], _impl.fit_full_two_part_predict(*args, **kwargs))


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = [
    "fit_full_predict",
    "fit_full_predict_fulltrain",
    "fit_full_two_part_predict",
]

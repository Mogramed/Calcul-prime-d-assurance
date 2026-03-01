from __future__ import annotations

from typing import Any, Mapping, Tuple

import pandas as pd

from src.insurance_pricing.legacy import training_benchmark_impl as _impl


def make_run_id(df: pd.DataFrame) -> pd.Series:
    return _impl.make_run_id(df)


def run_benchmark(
    spec: Mapping[str, Any],
    *,
    bundle: Any,
    splits: Mapping[str, Mapping[int, tuple]],
    seed: int,
    collect_predictions: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ = collect_predictions  # Compatibility arg kept for notebook/API stability.
    return _impl.run_benchmark(
        spec=spec,
        bundle=bundle,
        splits=splits,
        seed=seed,
    )


def _fit_predict_fold_v2(*args: Any, **kwargs: Any) -> tuple:
    return _impl._fit_predict_fold_v2(*args, **kwargs)


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


__all__ = [
    "make_run_id",
    "run_benchmark",
    "_fit_predict_fold_v2",
]

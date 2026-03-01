from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from src.insurance_pricing.cv.splits import build_split_registry, validate_folds_disjoint
SplitRegistry = Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]


def build_splits(train: pd.DataFrame, config: Any) -> SplitRegistry:
    return build_split_registry(
        train,
        n_blocks_time=int(config.n_blocks_time),
        n_splits_group=int(config.n_splits_group),
        group_col=str(config.group_col),
    )


def validate_split_integrity(
    splits: Mapping[str, Mapping[int, Tuple[np.ndarray, np.ndarray]]],
    *,
    train: pd.DataFrame,
    group_col: str = "id_client",
) -> dict:
    report = {}
    for split_name, folds in splits.items():
        try:
            validate_folds_disjoint(
                folds,
                check_full_coverage=(split_name in {"secondary_group", "aux_blocked5"}),
                n_rows=len(train),
            )
            ok = True
            msg = "ok"
        except Exception as exc:  # pragma: no cover - defensive
            ok = False
            msg = str(exc)
        report[split_name] = {"ok": ok, "message": msg, "n_folds": len(folds)}

    if "secondary_group" in splits and group_col in train.columns:
        leak = False
        for _, (tr_idx, va_idx) in splits["secondary_group"].items():
            tr_groups = set(train.iloc[tr_idx][group_col].tolist())
            va_groups = set(train.iloc[va_idx][group_col].tolist())
            if tr_groups.intersection(va_groups):
                leak = True
                break
        report["secondary_group"]["group_leak"] = bool(leak)
        report["secondary_group"]["ok"] = report["secondary_group"]["ok"] and (not leak)
    return report

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from insurance_pricing._typing import IntArray, SplitIndices
from insurance_pricing.data.io import ensure_dir
from insurance_pricing.data.schema import DEFAULT_V2_DIR, INDEX_COL


def build_primary_time_folds(
    train: pd.DataFrame,
    *,
    n_blocks: int = 5,
    index_col: str = INDEX_COL,
) -> dict[int, SplitIndices]:
    order = np.argsort(train[index_col].to_numpy())
    blocks = np.array_split(order, n_blocks)
    folds: dict[int, SplitIndices] = {}
    for fold in range(1, n_blocks):
        tr = np.concatenate(blocks[:fold]).astype(int)
        va = blocks[fold].astype(int)
        folds[fold] = (tr, va)
    return folds


def build_secondary_group_folds(
    train: pd.DataFrame,
    *,
    n_splits: int = 5,
    group_col: str = "id_client",
) -> dict[int, SplitIndices]:
    gkf = GroupKFold(n_splits=n_splits)
    groups = train[group_col].to_numpy()
    out: dict[int, SplitIndices] = {}
    for i, (tr, va) in enumerate(gkf.split(train, groups=groups), start=1):
        out[i] = (tr.astype(int), va.astype(int))
    return out


def validate_folds_disjoint(
    folds: Mapping[int, SplitIndices],
    *,
    check_full_coverage: bool = False,
    n_rows: int | None = None,
) -> None:
    valid_union: list[int] = []
    for fold, (tr, va) in folds.items():
        inter = set(map(int, tr)).intersection(set(map(int, va)))
        if inter:
            raise AssertionError(f"Fold {fold} has train/valid overlap")
        valid_union.extend(list(map(int, va)))
    if check_full_coverage:
        if n_rows is None:
            raise ValueError("n_rows is required when check_full_coverage=True")
        expected = set(range(n_rows))
        got = set(valid_union)
        if expected != got:
            raise AssertionError(f"Coverage mismatch. Missing={len(expected - got)}")


def folds_to_frame(
    folds: Mapping[int, SplitIndices],
    *,
    split_name: str,
    n_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, (tr, va) in folds.items():
        rows.extend(
            {"split": split_name, "fold_id": int(fold), "row_idx": int(i), "role": "train"}
            for i in tr
        )
        rows.extend(
            {"split": split_name, "fold_id": int(fold), "row_idx": int(i), "role": "valid"}
            for i in va
        )
    df = pd.DataFrame(rows)
    df["n_rows_total"] = int(n_rows)
    return df


def export_fold_artifacts(
    *,
    train: pd.DataFrame,
    primary_folds: Mapping[int, SplitIndices],
    secondary_folds: Mapping[int, SplitIndices],
    output_dir: str | Path = "artifacts",
) -> None:
    out = ensure_dir(output_dir)
    folds_to_frame(primary_folds, split_name="primary_time", n_rows=len(train)).to_parquet(
        out / "folds_primary.parquet", index=False
    )
    folds_to_frame(secondary_folds, split_name="secondary_group", n_rows=len(train)).to_parquet(
        out / "folds_secondary.parquet", index=False
    )


def build_aux_blocked_folds(
    train: pd.DataFrame, *, n_blocks: int = 5, index_col: str = INDEX_COL
) -> dict[int, SplitIndices]:
    order = np.argsort(train[index_col].to_numpy())
    blocks = np.array_split(order, n_blocks)
    out: dict[int, SplitIndices] = {}
    for i, va in enumerate(blocks, start=1):
        tr = np.concatenate([b for j, b in enumerate(blocks) if j != (i - 1)]).astype(int)
        out[i] = (tr, va.astype(int))
    return out


def build_split_registry(
    train: pd.DataFrame,
    *,
    n_blocks_time: int = 5,
    n_splits_group: int = 5,
    group_col: str = "id_client",
) -> dict[str, dict[int, SplitIndices]]:
    return {
        "primary_time": build_primary_time_folds(train, n_blocks=n_blocks_time),
        "secondary_group": build_secondary_group_folds(
            train, n_splits=n_splits_group, group_col=group_col
        ),
        "aux_blocked5": build_aux_blocked_folds(train, n_blocks=5),
    }


def validate_group_disjoint(
    folds: Mapping[int, SplitIndices],
    groups: pd.Series | IntArray,
) -> None:
    g = pd.Series(groups).astype(str).to_numpy()
    for fold_id, (tr, va) in folds.items():
        if set(g[tr]).intersection(set(g[va])):
            raise AssertionError(f"Group overlap in fold {fold_id}")


def export_split_artifacts_v2(
    *,
    train: pd.DataFrame,
    splits: Mapping[str, Mapping[int, SplitIndices]],
    output_dir: str | Path = DEFAULT_V2_DIR,
) -> None:
    out = ensure_dir(output_dir)
    names = {
        "primary_time": "folds_primary_time.parquet",
        "secondary_group": "folds_secondary_group.parquet",
        "aux_blocked5": "folds_aux_blocked5.parquet",
    }
    for split_name, folds in splits.items():
        if split_name not in names:
            continue
        folds_to_frame(folds, split_name=split_name, n_rows=len(train)).to_parquet(
            out / names[split_name], index=False
        )

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from insurance_pricing.data.schema import INDEX_COL

from .metrics import rmse
from .run_id import make_run_id_from_df

def compute_prediction_distribution_audit(
    pred: np.ndarray,
    *,
    y_true: Optional[np.ndarray] = None,
    run_id: Optional[str] = None,
    split: Optional[str] = None,
    sample: Optional[str] = None,
    collapse_q99_q90_ratio: float = 1.03,
    collapse_identical_ratio: float = 0.15,
) -> Dict[str, Any]:
    p = np.asarray(pred, dtype=float)
    p = np.maximum(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    n = int(len(p))
    if n == 0:
        base = {
            "run_id": run_id,
            "split": split,
            "sample": sample,
            "n": 0,
            "pred_mean": float("nan"),
            "pred_std": float("nan"),
            "pred_q50": float("nan"),
            "pred_q90": float("nan"),
            "pred_q95": float("nan"),
            "pred_q99": float("nan"),
            "pred_max": float("nan"),
            "pred_share_zero": float("nan"),
            "pred_identical_share": float("nan"),
            "pred_q99_q90_ratio": float("nan"),
            "distribution_collapse_flag": 1,
        }
        if y_true is not None:
            base["rmse_pred"] = float("nan")
            base["q99_ratio_to_true"] = float("nan")
        return base

    q50, q90, q95, q99 = np.nanquantile(p, [0.50, 0.90, 0.95, 0.99])
    pred_max = float(np.nanmax(p))
    pred_mean = float(np.nanmean(p))
    pred_std = float(np.nanstd(p))
    p_round = np.round(p, 6)
    _, counts = np.unique(p_round, return_counts=True)
    identical_share = float(np.max(counts) / n) if len(counts) else 1.0
    share_zero = float(np.mean(p_round <= 0))
    share_nonzero = float(np.mean(p_round > 0))
    q99_q90_ratio = float(q99 / max(q90, 1e-9))
    p_nonzero = p[p_round > 0]
    pred_n_nonzero = int(len(p_nonzero))
    if pred_n_nonzero:
        p_nz_round = np.round(p_nonzero, 6)
        _, counts_nz = np.unique(p_nz_round, return_counts=True)
        identical_share_nonzero = float(np.max(counts_nz) / pred_n_nonzero) if len(counts_nz) else 1.0
        q90_nz, q99_nz = np.nanquantile(p_nonzero, [0.90, 0.99])
        q99_q90_ratio_nonzero = float(q99_nz / max(q90_nz, 1e-9))
    else:
        identical_share_nonzero = float("nan")
        q90_nz = float("nan")
        q99_nz = float("nan")
        q99_q90_ratio_nonzero = float("nan")

    # Zero-inflated targets naturally create many identical zeros; collapse should be judged on non-zero support when available.
    use_nonzero_for_collapse = bool((share_zero >= 0.05) and (pred_n_nonzero >= max(50, int(0.01 * n))))
    collapse_ratio_eval = q99_q90_ratio_nonzero if use_nonzero_for_collapse and np.isfinite(q99_q90_ratio_nonzero) else q99_q90_ratio
    collapse_identical_eval = (
        identical_share_nonzero if use_nonzero_for_collapse and np.isfinite(identical_share_nonzero) else identical_share
    )
    collapse_flag = int(
        (collapse_ratio_eval <= collapse_q99_q90_ratio)
        or (collapse_identical_eval >= collapse_identical_ratio)
    )

    out: Dict[str, Any] = {
        "run_id": run_id,
        "split": split,
        "sample": sample,
        "n": n,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_q50": float(q50),
        "pred_q90": float(q90),
        "pred_q95": float(q95),
        "pred_q99": float(q99),
        "pred_max": pred_max,
        "pred_share_zero": share_zero,
        "pred_share_nonzero": share_nonzero,
        "pred_identical_share": identical_share,
        "pred_identical_share_nonzero": identical_share_nonzero,
        "pred_q99_q90_ratio": q99_q90_ratio,
        "pred_n_nonzero": pred_n_nonzero,
        "pred_q90_nonzero": float(q90_nz) if np.isfinite(q90_nz) else float("nan"),
        "pred_q99_nonzero": float(q99_nz) if np.isfinite(q99_nz) else float("nan"),
        "pred_q99_q90_ratio_nonzero": q99_q90_ratio_nonzero,
        "collapse_use_nonzero_support": int(use_nonzero_for_collapse),
        "distribution_collapse_flag": collapse_flag,
    }

    if y_true is not None:
        y = np.asarray(y_true, dtype=float)
        y = np.maximum(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        q99_true = float(np.nanquantile(y, 0.99))
        out["rmse_pred"] = rmse(y, p)
        out["q99_ratio_to_true"] = float(q99 / max(q99_true, 1e-9))
        y_nonzero = y[y > 0]
        if len(y_nonzero) and pred_n_nonzero:
            out["q99_ratio_to_true_nonzero"] = float(
                float(np.nanquantile(p_nonzero, 0.99)) / max(float(np.nanquantile(y_nonzero, 0.99)), 1e-9)
            )
        else:
            out["q99_ratio_to_true_nonzero"] = float("nan")
    return out

def build_prediction_distribution_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    d = pred_df.copy()
    if "run_id" not in d.columns:
        d["run_id"] = make_run_id_from_df(d)
    rows: List[Dict[str, Any]] = []
    for (run_id, split, is_test), g in d.groupby(["run_id", "split", "is_test"], dropna=False):
        g = g.copy()
        sample = "test" if int(is_test) == 1 else "oof"
        p = g["pred_prime"].to_numpy(dtype=float)
        if sample == "oof" and "y_sev" in g.columns:
            yy = g["y_sev"].to_numpy(dtype=float)
            valid = np.isfinite(yy)
            row = compute_prediction_distribution_audit(
                p[valid], y_true=yy[valid], run_id=str(run_id), split=str(split), sample=sample
            )
        else:
            row = compute_prediction_distribution_audit(
                p, y_true=None, run_id=str(run_id), split=str(split), sample=sample
            )
        rows.append(row)
    return pd.DataFrame(rows)

def compute_ood_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    cat_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if cat_cols is None:
        cat_cols = [c for c in train.columns if train[c].dtype == "object" and c in test.columns]
    rows: List[Dict[str, Any]] = []
    for c in cat_cols:
        tr_u = set(train[c].astype(str).dropna().unique())
        te_u = set(test[c].astype(str).dropna().unique())
        unseen = te_u - tr_u
        rows.append(
            {
                "diagnostic_type": "ood",
                "feature": c,
                "train_unique": int(len(tr_u)),
                "test_unique": int(len(te_u)),
                "unseen_test_levels": int(len(unseen)),
                "unseen_ratio_on_levels": float(len(unseen) / max(len(te_u), 1)),
            }
        )
    return pd.DataFrame(rows)

def compute_segment_bias_from_oof(
    train: pd.DataFrame,
    oof_df: pd.DataFrame,
    *,
    run_id: str,
    split_name: str = "primary_time",
    segment_cols: Optional[Sequence[str]] = None,
    min_count: int = 150,
) -> pd.DataFrame:
    if segment_cols is None:
        segment_cols = [
            "utilisation",
            "type_contrat",
            "cp2",
            "cp3",
            "marque_vehicule",
            "modele_vehicule",
        ]
    d = oof_df.copy()
    if "run_id" not in d.columns:
        d["run_id"] = make_run_id_from_df(d)
    d = d[(d["is_test"] == 0) & (d["split"] == split_name) & (d["run_id"] == run_id)].copy()
    d = d.sort_values("row_idx")
    valid = d["pred_prime"].notna().to_numpy()
    tr = train.reset_index(drop=True).loc[valid].copy()
    dd = d.loc[valid].copy().reset_index(drop=True)
    tr["y_true"] = dd["y_sev"].to_numpy()
    tr["pred_prime"] = dd["pred_prime"].to_numpy()
    tr["error"] = tr["pred_prime"] - tr["y_true"]
    rows: List[Dict[str, Any]] = []
    for c in segment_cols:
        if c not in tr.columns:
            continue
        grp = (
            tr.groupby(c)
            .agg(
                n=("y_true", "size"),
                y_mean=("y_true", "mean"),
                p_mean=("pred_prime", "mean"),
                bias=("error", "mean"),
            )
            .reset_index()
        )
        grp = grp[grp["n"] >= min_count]
        for _, r in grp.iterrows():
            rows.append(
                {
                    "diagnostic_type": "segment_bias",
                    "feature": c,
                    "segment": str(r[c]),
                    "n": int(r["n"]),
                    "y_mean": float(r["y_mean"]),
                    "p_mean": float(r["p_mean"]),
                    "bias": float(r["bias"]),
                }
            )
    return pd.DataFrame(rows)

def build_model_cards(run_registry: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rr = run_registry.copy()
    if "run_id" not in rr.columns:
        rr["run_id"] = make_run_id_from_df(rr)
    sel = selected[["run_id"]].drop_duplicates().copy()
    card = rr[rr["run_id"].isin(set(sel["run_id"]))].copy()
    cols = [
        "run_id",
        "feature_set",
        "engine",
        "family",
        "tweedie_power",
        "config_id",
        "seed",
        "severity_mode",
        "calibration",
        "tail_mapper",
        "split",
        "rmse_prime",
        "auc_freq",
        "brier_freq",
        "rmse_sev_pos",
        "q99_ratio_pos",
    ]
    cols = [c for c in cols if c in card.columns]
    return card[cols].sort_values(["run_id", "split"]).reset_index(drop=True)

def simulate_public_private_shakeup(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    n_sim: int = 2000,
    public_ratio: float = 1.0 / 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    n_pub = int(round(n * public_ratio))
    idx = np.arange(n)
    rows: List[Dict[str, float]] = []
    for s in range(n_sim):
        rng.shuffle(idx)
        pub = idx[:n_pub]
        pri = idx[n_pub:]
        pub_rmse = rmse(y[pub], p[pub])
        pri_rmse = rmse(y[pri], p[pri])
        rows.append(
            {
                "sim_id": s,
                "rmse_public": pub_rmse,
                "rmse_private": pri_rmse,
                "gap_public_minus_private": pub_rmse - pri_rmse,
            }
        )
    return pd.DataFrame(rows)

def simulate_public_private_shakeup_v2(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    n_sim: int = 2000,
    public_ratio: float = 1.0 / 3.0,
    seed: int = 42,
    stratified_tail: bool = False,
    tail_quantile: float = 0.9,
    tail_public_share: float = 0.5,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    n_pub = int(round(n * public_ratio))
    idx = np.arange(n)
    rows: List[Dict[str, float]] = []

    if stratified_tail:
        thr = float(np.nanquantile(y, tail_quantile))
        tail_idx = idx[y >= thr]
        body_idx = idx[y < thr]

    for s in range(n_sim):
        if stratified_tail:
            rng.shuffle(tail_idx)
            rng.shuffle(body_idx)
            n_tail_pub = min(len(tail_idx), int(round(n_pub * tail_public_share)))
            n_body_pub = n_pub - n_tail_pub
            pub = np.concatenate([tail_idx[:n_tail_pub], body_idx[:n_body_pub]])
            pri = np.setdiff1d(idx, pub, assume_unique=False)
        else:
            rng.shuffle(idx)
            pub = idx[:n_pub]
            pri = idx[n_pub:]
        pub_rmse = rmse(y[pub], p[pub])
        pri_rmse = rmse(y[pri], p[pri])
        rows.append(
            {
                "sim_id": s,
                "rmse_public": pub_rmse,
                "rmse_private": pri_rmse,
                "gap_public_minus_private": pub_rmse - pri_rmse,
                "stratified_tail": int(stratified_tail),
            }
        )
    return pd.DataFrame(rows)


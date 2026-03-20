from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from insurance_pricing.evaluation.metrics import rmse
from insurance_pricing.evaluation.run_id import make_run_id_from_df

def optimize_non_negative_weights(pred_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    p = np.asarray(pred_matrix, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n_models = p.shape[1]
    x0 = np.full(n_models, 1.0 / n_models)
    bounds = [(0.0, 1.0)] * n_models
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        return rmse(y, p @ w)

    r = minimize(objective, x0=x0, bounds=bounds, constraints=constraints)
    if not r.success:
        return x0
    w = np.clip(r.x, 0.0, 1.0)
    s = w.sum()
    return x0 if s <= 0 else w / s

def pick_top_configs(
    run_registry: pd.DataFrame,
    *,
    split_name: str = "primary_time",
    top_k_per_engine: int = 2,
) -> Dict[str, List[str]]:
    rr = run_registry.copy()
    rr = rr[(rr["level"] == "run") & (rr["split"] == split_name)]
    rr = rr.sort_values(["engine", "rmse_prime", "brier_freq"])
    out: Dict[str, List[str]] = {}
    for engine, g in rr.groupby("engine"):
        out[engine] = g["config_id"].drop_duplicates().head(top_k_per_engine).tolist()
    return out

def select_final_models(
    run_registry: pd.DataFrame,
    risk_policy: str | Mapping[str, Any] = "stability_private",
    *,
    return_report: bool = False,
) -> pd.DataFrame:
    if isinstance(risk_policy, str):
        rp = risk_policy.lower()
        if rp == "stability_private":
            policy = {
                "split_weights": {
                    "primary_time": 0.65,
                    "secondary_group": 0.20,
                    "aux_blocked5": 0.15,
                },
                "max_secondary_gap": 1.0,
                "max_aux_gap": 1.0,
                "max_models": 6,
                "min_q99_ratio": 0.30,
                "gap_penalty_weight": 1.0,
                "dispersion_penalty_weight": 0.75,
                "tail_penalty_weight": 2.5,
                "collapse_penalty_weight": 3.5,
                "tail_dispersion_penalty_weight": 2.0,
                "max_distribution_collapse_flag": 0,
                "max_tail_dispersion_flag": 0,
            }
        elif rp == "balanced":
            policy = {
                "split_weights": {
                    "primary_time": 0.55,
                    "secondary_group": 0.25,
                    "aux_blocked5": 0.20,
                },
                "max_secondary_gap": 1.5,
                "max_aux_gap": 1.5,
                "max_models": 6,
                "min_q99_ratio": 0.20,
                "gap_penalty_weight": 0.8,
                "dispersion_penalty_weight": 0.55,
                "tail_penalty_weight": 2.0,
                "collapse_penalty_weight": 2.0,
                "tail_dispersion_penalty_weight": 1.2,
                "max_distribution_collapse_flag": 1,
                "max_tail_dispersion_flag": 1,
            }
        else:
            policy = {
                "split_weights": {
                    "primary_time": 0.5,
                    "secondary_group": 0.25,
                    "aux_blocked5": 0.25,
                },
                "max_secondary_gap": 3.0,
                "max_aux_gap": 3.0,
                "max_models": 6,
                "min_q99_ratio": 0.10,
                "gap_penalty_weight": 0.5,
                "dispersion_penalty_weight": 0.35,
                "tail_penalty_weight": 1.0,
                "collapse_penalty_weight": 1.0,
                "tail_dispersion_penalty_weight": 0.5,
                "max_distribution_collapse_flag": 1,
                "max_tail_dispersion_flag": 1,
            }
    else:
        policy = dict(risk_policy)

    rr = run_registry.copy()
    if "run_id" not in rr.columns:
        rr["run_id"] = make_run_id_from_df(rr)
    rr = rr[rr["level"] == "run"].copy()
    if rr.empty:
        return pd.DataFrame()

    piv_rmse = rr.pivot_table(index="run_id", columns="split", values="rmse_prime", aggfunc="mean")
    piv_q99 = rr.pivot_table(index="run_id", columns="split", values="q99_ratio_pos", aggfunc="mean")
    meta_cols_all = [
        "feature_set",
        "engine",
        "family",
        "tweedie_power",
        "config_id",
        "seed",
        "severity_mode",
        "calibration",
        "tail_mapper",
    ]
    meta_cols = [c for c in meta_cols_all if c in rr.columns]
    meta = rr.groupby("run_id")[meta_cols].first()
    flags = rr.groupby("run_id").agg(
        tail_dispersion_flag=("tail_dispersion_flag", "max")
        if "tail_dispersion_flag" in rr.columns
        else ("rmse_prime", "size"),
        distribution_collapse_flag=("distribution_collapse_flag", "max")
        if "distribution_collapse_flag" in rr.columns
        else ("rmse_prime", "size"),
        rmse_gap_secondary=("rmse_gap_secondary", "mean")
        if "rmse_gap_secondary" in rr.columns
        else ("rmse_prime", "size"),
        rmse_gap_aux=("rmse_gap_aux", "mean")
        if "rmse_gap_aux" in rr.columns
        else ("rmse_prime", "size"),
    )
    if "tail_dispersion_flag" not in rr.columns:
        flags["tail_dispersion_flag"] = 0.0
    if "distribution_collapse_flag" not in rr.columns:
        flags["distribution_collapse_flag"] = 0.0
    if "rmse_gap_secondary" not in rr.columns:
        flags["rmse_gap_secondary"] = np.nan
    if "rmse_gap_aux" not in rr.columns:
        flags["rmse_gap_aux"] = np.nan

    out = meta.join(piv_rmse.add_prefix("rmse_"), how="left").join(
        piv_q99.add_prefix("q99_"), how="left"
    )
    out = out.join(flags, how="left")
    out = out.reset_index()
    for c in ["rmse_primary_time", "rmse_secondary_group", "rmse_aux_blocked5", "q99_primary_time"]:
        if c not in out.columns:
            out[c] = np.nan

    if out["rmse_gap_secondary"].isna().all():
        out["rmse_gap_secondary"] = out["rmse_secondary_group"] - out["rmse_primary_time"]
    if out["rmse_gap_aux"].isna().all():
        out["rmse_gap_aux"] = out["rmse_aux_blocked5"] - out["rmse_primary_time"]

    split_weights = dict(policy.get("split_weights", {}))
    if not split_weights:
        split_weights = {"primary_time": 1.0}
    w_primary = float(split_weights.get("primary_time", 0.0))
    w_secondary = float(split_weights.get("secondary_group", 0.0))
    w_aux = float(split_weights.get("aux_blocked5", 0.0))

    num = (
        w_primary * out["rmse_primary_time"].fillna(0.0)
        + w_secondary * out["rmse_secondary_group"].fillna(0.0)
        + w_aux * out["rmse_aux_blocked5"].fillna(0.0)
    )
    den = (
        w_primary * out["rmse_primary_time"].notna().astype(float)
        + w_secondary * out["rmse_secondary_group"].notna().astype(float)
        + w_aux * out["rmse_aux_blocked5"].notna().astype(float)
    ).clip(lower=1e-9)
    out["rmse_weighted"] = num / den
    out["rmse_split_std"] = out[
        ["rmse_primary_time", "rmse_secondary_group", "rmse_aux_blocked5"]
    ].std(axis=1, ddof=0)
    out["gap_penalty"] = (
        np.maximum(out["rmse_gap_secondary"].fillna(0.0), 0.0)
        + np.maximum(out["rmse_gap_aux"].fillna(0.0), 0.0)
    )
    out["tail_penalty"] = (1.0 - out["q99_primary_time"].fillna(0.0)).abs()
    out["selection_score"] = (
        out["rmse_weighted"].fillna(np.inf)
        + float(policy.get("gap_penalty_weight", 1.0)) * out["gap_penalty"]
        + float(policy.get("dispersion_penalty_weight", 0.0)) * out["rmse_split_std"].fillna(0.0)
        + float(policy.get("tail_penalty_weight", 0.0)) * out["tail_penalty"]
        + float(policy.get("collapse_penalty_weight", 0.0))
        * out["distribution_collapse_flag"].fillna(0.0).astype(float)
        + float(policy.get("tail_dispersion_penalty_weight", 0.0))
        * out["tail_dispersion_flag"].fillna(0.0).astype(float)
    )

    out["accepted_secondary"] = out["rmse_gap_secondary"].fillna(0.0) <= float(
        policy.get("max_secondary_gap", np.inf)
    )
    out["accepted_aux"] = out["rmse_gap_aux"].fillna(0.0) <= float(
        policy.get("max_aux_gap", np.inf)
    )
    out["accepted_tail"] = out["q99_primary_time"].fillna(0.0) >= float(
        policy.get("min_q99_ratio", 0.0)
    )
    out["accepted_collapse"] = out["distribution_collapse_flag"].fillna(0.0) <= float(
        policy.get("max_distribution_collapse_flag", np.inf)
    )
    out["accepted_dispersion"] = out["tail_dispersion_flag"].fillna(0.0) <= float(
        policy.get("max_tail_dispersion_flag", np.inf)
    )
    out["accepted"] = (
        out["accepted_secondary"]
        & out["accepted_aux"]
        & out["accepted_tail"]
        & out["accepted_collapse"]
        & out["accepted_dispersion"]
    )

    def _build_reason(r: pd.Series) -> str:
        reasons: List[str] = []
        if not bool(r["accepted_secondary"]):
            reasons.append("secondary_gap")
        if not bool(r["accepted_aux"]):
            reasons.append("aux_gap")
        if not bool(r["accepted_tail"]):
            reasons.append("tail_q99")
        if not bool(r["accepted_collapse"]):
            reasons.append("distribution_collapse")
        if not bool(r["accepted_dispersion"]):
            reasons.append("tail_dispersion")
        return "accepted" if not reasons else ",".join(reasons)

    out["decision_reason"] = out.apply(_build_reason, axis=1)
    out = out.sort_values(["accepted", "selection_score"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=int)

    if return_report:
        return out

    sel = out[out["accepted"]].head(int(policy["max_models"]))
    if sel.empty:
        sel = out.head(int(policy["max_models"])).copy()
        sel["decision_reason"] = sel["decision_reason"].astype(str) + "|fallback"
    return sel.reset_index(drop=True)


def select_best_run(
    run_df: pd.DataFrame,
    *,
    split: str = "primary_time",
) -> Mapping[str, Any]:
    if run_df is None or run_df.empty:
        raise ValueError("run_df is empty.")
    d = run_df.copy()
    if "level" in d.columns:
        d = d[d["level"].astype(str) == "run"]
    if "split" in d.columns:
        d = d[d["split"].astype(str) == split]
    if d.empty:
        raise ValueError(f"No rows for split={split}.")
    d = d.sort_values("rmse_prime", na_position="last")
    return d.iloc[0].to_dict()


def score_multi_split(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df is None or run_df.empty:
        return pd.DataFrame()
    d = run_df.copy()
    if "level" in d.columns:
        d = d[d["level"].astype(str) == "run"]
    keys = [
        c
        for c in [
            "run_id",
            "engine",
            "family",
            "config_id",
            "seed",
            "feature_set",
            "severity_mode",
            "calibration",
            "tail_mapper",
        ]
        if c in d.columns
    ]
    piv = d.pivot_table(index=keys, columns="split", values="rmse_prime", aggfunc="first").reset_index()
    piv["rmse_primary_time"] = pd.to_numeric(piv.get("primary_time"), errors="coerce")
    piv["rmse_secondary_group"] = pd.to_numeric(piv.get("secondary_group"), errors="coerce")
    piv["rmse_aux_blocked5"] = pd.to_numeric(piv.get("aux_blocked5"), errors="coerce")
    piv["rmse_gap_secondary"] = piv["rmse_secondary_group"] - piv["rmse_primary_time"]
    piv["rmse_gap_aux"] = piv["rmse_aux_blocked5"] - piv["rmse_primary_time"]
    arr = np.vstack(
        [
            piv["rmse_primary_time"].to_numpy(dtype=float),
            piv["rmse_secondary_group"].to_numpy(dtype=float),
            piv["rmse_aux_blocked5"].to_numpy(dtype=float),
        ]
    ).T
    piv["rmse_split_std"] = np.nanstd(arr, axis=1)
    return piv.sort_values("rmse_primary_time", na_position="last").reset_index(drop=True)


from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from insurance_pricing._typing import FloatArray
from insurance_pricing.evaluation import diagnostics as v2diag

from .quality import _rmse


def compute_error_by_deciles(
    y_true: FloatArray,
    y_pred: FloatArray,
    n_bins: int = 10,
    mode: str = "qcut_all",
    zero_aware: bool | None = None,
) -> pd.DataFrame:
    if zero_aware is not None:
        mode = "zero_aware" if bool(zero_aware) else "qcut_all"

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return pd.DataFrame()

    def _agg(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return pd.DataFrame(
                columns=[
                    "bin",
                    "n",
                    "y_mean",
                    "pred_mean",
                    "bias",
                    "mae",
                    "rmse",
                ]
            )
        dd = df_in.copy()
        dd["err"] = dd["y_pred"] - dd["y_true"]
        dd["abs_err"] = dd["err"].abs()
        return (
            dd.groupby("bin")
            .agg(
                n=("y_true", "size"),
                y_mean=("y_true", "mean"),
                pred_mean=("y_pred", "mean"),
                bias=("err", "mean"),
                mae=("abs_err", "mean"),
                rmse=("err", lambda e: float(np.sqrt(np.mean(np.square(e))))),
            )
            .reset_index()
        )

    if mode == "zero_aware":
        d = pd.DataFrame({"y_true": y, "y_pred": p})
        d["bin_type"] = "positive_decile"
        d["bin_order"] = np.nan

        zero_mask = d["y_true"] <= 0
        out_parts: list[pd.DataFrame] = []

        if zero_mask.any():
            dz = d.loc[zero_mask, ["y_true", "y_pred"]].copy()
            dz["bin"] = "zero"
            zagg = _agg(dz)
            if not zagg.empty:
                zagg["bin_type"] = "zero"
                zagg["bin_order"] = 0
                out_parts.append(zagg)

        dpos = d.loc[~zero_mask, ["y_true", "y_pred"]].copy()
        if not dpos.empty:
            n_unique_pos = int(pd.Series(dpos["y_true"]).nunique(dropna=True))
            q = max(1, min(n_bins, n_unique_pos))
            if q == 1:
                dpos["bin"] = "positive_all"
            else:
                dpos["bin"] = pd.qcut(dpos["y_true"], q=q, duplicates="drop").astype(str)
            pagg = _agg(dpos)
            if not pagg.empty:
                pagg = pagg.sort_values("y_mean").reset_index(drop=True)
                pagg["bin_type"] = "positive_decile"
                pagg["bin_order"] = np.arange(1, len(pagg) + 1)
                out_parts.append(pagg)

        if not out_parts:
            return pd.DataFrame()
        out = pd.concat(out_parts, ignore_index=True, sort=False)
        cols_front = ["bin_type", "bin_order", "bin"]
        other_cols = [c for c in out.columns if c not in cols_front]
        return out[cols_front + other_cols]

    # default legacy mode
    q = max(1, min(n_bins, len(np.unique(y))))
    rank = (
        pd.Series(["all"] * len(y))
        if q == 1
        else pd.qcut(pd.Series(y), q=q, duplicates="drop")
    )
    d = pd.DataFrame({"y_true": y, "y_pred": p, "bin": rank.astype(str)})
    out = _agg(d)
    if not out.empty:
        out["bin_type"] = "qcut_all"
        out["bin_order"] = np.arange(1, len(out) + 1)
        cols_front = ["bin_type", "bin_order", "bin"]
        out = out[cols_front + [c for c in out.columns if c not in cols_front]]
    return out


def compute_calibration_table(
    y_true: FloatArray, p_pred: FloatArray, n_bins: int = 10
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)
    if len(y) == 0:
        return pd.DataFrame()
    bins = pd.qcut(pd.Series(p), q=min(n_bins, len(np.unique(p))), duplicates="drop")
    d = pd.DataFrame({"y_true": y, "p_pred": p, "bin": bins.astype(str)})
    out = (
        d.groupby("bin")
        .agg(
            n=("y_true", "size"),
            p_mean=("p_pred", "mean"),
            y_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    out["calibration_gap"] = out["p_mean"] - out["y_rate"]
    return out


def compute_oof_model_diagnostics(
    oof_df: pd.DataFrame,
    run_id: str,
    split: str = "primary_time",
    decile_mode: str = "zero_aware",
) -> dict[str, pd.DataFrame]:
    d = oof_df.copy()
    if "run_id" not in d.columns:
        # fallback legacy
        cols = [
            "feature_set",
            "engine",
            "family",
            "config_id",
            "seed",
            "severity_mode",
            "calibration",
            "tail_mapper",
        ]
        cols = [c for c in cols if c in d.columns]
        if cols:
            rid = d[cols[0]].astype(str)
            for c in cols[1:]:
                rid = rid + "|" + d[c].astype(str)
            d["run_id"] = rid
    d = d[
        (d.get("is_test", 0) == 0)
        & (d["split"] == split)
        & (d["run_id"].astype(str) == str(run_id))
    ].copy()
    if d.empty:
        return {
            "metrics": pd.DataFrame(),
            "error_by_decile_true": pd.DataFrame(),
            "error_by_decile_true_zero_aware": pd.DataFrame(),
            "error_by_decile_pos_only": pd.DataFrame(),
            "error_by_decile_pred": pd.DataFrame(),
            "calibration_freq": pd.DataFrame(),
            "residuals": pd.DataFrame(),
            "residuals_segment_summary": pd.DataFrame(),
            "extreme_cases_summary": pd.DataFrame(),
            "distribution": pd.DataFrame(),
        }
    d = d[d["y_sev"].notna() & d["pred_prime"].notna()].copy()
    y = d["y_sev"].to_numpy(dtype=float)
    pred_prime = d["pred_prime"].to_numpy(dtype=float)
    pred_freq = (
        d["pred_freq"].to_numpy(dtype=float)
        if "pred_freq" in d.columns
        else np.full(len(d), np.nan)
    )
    pred_sev = (
        d["pred_sev"].to_numpy(dtype=float) if "pred_sev" in d.columns else np.full(len(d), np.nan)
    )
    y_freq = d["y_freq"].to_numpy(dtype=float) if "y_freq" in d.columns else (y > 0).astype(float)
    pos = y > 0

    metrics = {
        "run_id": str(run_id),
        "split": split,
        "n": int(len(d)),
        "n_nonzero": int(np.sum(y > 0)),
        "share_nonzero": float(np.mean(y > 0)),
        "rmse_prime": _rmse(y, pred_prime),
        "mae_prime": float(mean_absolute_error(y, pred_prime)),
        "r2_prime": float(r2_score(y, pred_prime)),
        "auc_freq": float(roc_auc_score(y_freq, pred_freq))
        if len(np.unique(y_freq)) > 1
        else np.nan,
        "gini_freq": float(2 * roc_auc_score(y_freq, pred_freq) - 1)
        if len(np.unique(y_freq)) > 1
        else np.nan,
        "brier_freq": float(brier_score_loss(y_freq.astype(int), np.clip(pred_freq, 0, 1)))
        if np.isfinite(pred_freq).any()
        else np.nan,
        "logloss_freq": float(log_loss(y_freq.astype(int), np.clip(pred_freq, 1e-6, 1 - 1e-6)))
        if np.isfinite(pred_freq).any() and len(np.unique(y_freq)) > 1
        else np.nan,
        "pr_auc_freq": float(average_precision_score(y_freq.astype(int), pred_freq))
        if len(np.unique(y_freq)) > 1
        else np.nan,
        "rmse_sev_pos": _rmse(y[pos], pred_sev[pos])
        if pos.any() and np.isfinite(pred_sev[pos]).any()
        else np.nan,
        "mae_sev_pos": float(mean_absolute_error(y[pos], pred_sev[pos]))
        if pos.any() and np.isfinite(pred_sev[pos]).any()
        else np.nan,
        "q95_ratio_pos": (
            float(np.quantile(pred_sev[pos], 0.95)) / max(float(np.quantile(y[pos], 0.95)), 1e-9)
        )
        if pos.any() and np.isfinite(pred_sev[pos]).any()
        else np.nan,
        "q99_ratio_pos": (
            float(np.quantile(pred_sev[pos], 0.99)) / max(float(np.quantile(y[pos], 0.99)), 1e-9)
        )
        if pos.any() and np.isfinite(pred_sev[pos]).any()
        else np.nan,
    }
    if pos.any():
        metrics["mae_prime_nonzero"] = float(mean_absolute_error(y[pos], pred_prime[pos]))
        metrics["r2_prime_nonzero"] = (
            float(r2_score(y[pos], pred_prime[pos])) if int(np.sum(pos)) > 1 else np.nan
        )
    else:
        metrics["mae_prime_nonzero"] = np.nan
        metrics["r2_prime_nonzero"] = np.nan
    q99_true_all = float(np.quantile(y, 0.99)) if len(y) else np.nan
    mask_top1 = y >= q99_true_all if np.isfinite(q99_true_all) else np.zeros(len(y), dtype=bool)
    metrics["rmse_prime_top1pct"] = (
        _rmse(y[mask_top1], pred_prime[mask_top1]) if mask_top1.any() else np.nan
    )
    metrics_df = pd.DataFrame([metrics])

    residuals = d[["row_idx"]].copy() if "row_idx" in d.columns else pd.DataFrame(index=d.index)
    residuals["y_true"] = y
    residuals["y_freq"] = y_freq
    residuals["pred_freq"] = pred_freq
    residuals["pred_sev"] = pred_sev
    residuals["pred_prime"] = pred_prime
    residuals["residual"] = pred_prime - y
    residuals["abs_error"] = np.abs(residuals["residual"])
    residuals["sq_error"] = residuals["residual"] ** 2
    residuals["is_extreme_true_top1"] = (y >= np.quantile(y, 0.99)).astype(int)
    residuals["is_nonzero_true"] = (y > 0).astype(int)
    residuals["decile_true_zero_aware_bucket"] = None

    err_true_zero_aware = compute_error_by_deciles(
        y_true=y, y_pred=pred_prime, n_bins=10, mode="zero_aware"
    )
    err_true = compute_error_by_deciles(y_true=y, y_pred=pred_prime, n_bins=10, mode=decile_mode)
    err_true.insert(0, "decile_basis", "y_true")
    if not err_true_zero_aware.empty:
        err_true_zero_aware.insert(0, "decile_basis", "y_true_zero_aware")
        if {"bin", "bin_type"}.issubset(err_true_zero_aware.columns):
            # assign per-row bucket for downstream residual summaries
            y_series = pd.Series(y)
            zero_mask_arr = y <= 0
            if zero_mask_arr.any():
                residuals.loc[zero_mask_arr, "decile_true_zero_aware_bucket"] = "zero:zero"
            pos_mask_arr = y > 0
            if pos_mask_arr.any():
                try:
                    y_pos_series = y_series[pos_mask_arr]
                    qpos = max(1, min(10, int(y_pos_series.nunique())))
                    if qpos == 1:
                        pos_bins = pd.Series(
                            ["positive_decile:positive_all"] * int(np.sum(pos_mask_arr)),
                            index=y_pos_series.index,
                        )
                    else:
                        pos_bin_labels = pd.qcut(y_pos_series, q=qpos, duplicates="drop").astype(
                            str
                        )
                        pos_bins = "positive_decile:" + pos_bin_labels
                    residuals.loc[pos_mask_arr, "decile_true_zero_aware_bucket"] = pos_bins.values
                except Exception:
                    residuals.loc[pos_mask_arr, "decile_true_zero_aware_bucket"] = (
                        "positive_decile:unknown"
                    )
    err_true_pos_only = (
        compute_error_by_deciles(y_true=y[pos], y_pred=pred_prime[pos], n_bins=10, mode="qcut_all")
        if pos.any()
        else pd.DataFrame()
    )
    if not err_true_pos_only.empty:
        err_true_pos_only.insert(0, "decile_basis", "y_true_positive_only")

    # predicted deciles
    try:
        bins_pred = pd.qcut(
            pd.Series(pred_prime), q=min(10, len(np.unique(pred_prime))), duplicates="drop"
        )
        tmp_pred = pd.DataFrame({"y_true": y, "y_pred": pred_prime, "bin": bins_pred.astype(str)})
        tmp_pred["err"] = tmp_pred["y_pred"] - tmp_pred["y_true"]
        tmp_pred["abs_err"] = tmp_pred["err"].abs()
        err_pred = (
            tmp_pred.groupby("bin")
            .agg(
                n=("y_true", "size"),
                y_mean=("y_true", "mean"),
                pred_mean=("y_pred", "mean"),
                bias=("err", "mean"),
                mae=("abs_err", "mean"),
                rmse=("err", lambda e: float(np.sqrt(np.mean(np.square(e))))),
            )
            .reset_index()
        )
    except Exception:
        err_pred = pd.DataFrame()
    if not err_pred.empty:
        err_pred.insert(0, "decile_basis", "y_pred")

    cal = compute_calibration_table(y_true=y_freq, p_pred=pred_freq, n_bins=10)
    dist = v2diag.build_prediction_distribution_table(d.assign(is_test=0))
    if not dist.empty:
        dist = dist.assign(run_id=str(run_id), split=split)
    residuals_segment_parts = []
    for seg_col in ["is_nonzero_true", "is_extreme_true_top1", "decile_true_zero_aware_bucket"]:
        if seg_col not in residuals.columns:
            continue
        g = (
            residuals.groupby(seg_col, dropna=False)
            .agg(
                n=("y_true", "size"),
                y_mean=("y_true", "mean"),
                pred_mean=("pred_prime", "mean"),
                bias=("residual", "mean"),
                mae=("abs_error", "mean"),
                rmse=("residual", lambda e: float(np.sqrt(np.mean(np.square(e))))),
            )
            .reset_index()
            .rename(columns={seg_col: "segment_value"})
        )
        g.insert(0, "segment_col", seg_col)
        residuals_segment_parts.append(g)
    residuals_segment_summary = (
        pd.concat(residuals_segment_parts, ignore_index=True, sort=False)
        if residuals_segment_parts
        else pd.DataFrame()
    )

    extreme_cases_rows = []
    if len(residuals):
        for label, mask_arr in [
            ("all", np.ones(len(residuals), dtype=bool)),
            ("true_top1pct", residuals["is_extreme_true_top1"].to_numpy(dtype=int) == 1),
            ("nonzero_only", residuals["is_nonzero_true"].to_numpy(dtype=int) == 1),
        ]:
            if not np.any(mask_arr):
                continue
            rr = residuals.loc[mask_arr]
            extreme_cases_rows.append(
                {
                    "subset": label,
                    "n": int(len(rr)),
                    "share": float(len(rr) / max(len(residuals), 1)),
                    "y_mean": float(rr["y_true"].mean()),
                    "pred_mean": float(rr["pred_prime"].mean()),
                    "mae": float(rr["abs_error"].mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(rr["residual"]))))
                    if len(rr)
                    else np.nan,
                    "q95_abs_error": float(rr["abs_error"].quantile(0.95)) if len(rr) else np.nan,
                    "max_abs_error": float(rr["abs_error"].max()) if len(rr) else np.nan,
                }
            )
    extreme_cases_summary = pd.DataFrame(extreme_cases_rows)
    return {
        "metrics": metrics_df,
        "error_by_decile_true": err_true,
        "error_by_decile_true_zero_aware": err_true_zero_aware,
        "error_by_decile_pos_only": err_true_pos_only,
        "error_by_decile_pred": err_pred,
        "calibration_freq": cal,
        "residuals": residuals,
        "residuals_segment_summary": residuals_segment_summary,
        "extreme_cases_summary": extreme_cases_summary,
        "distribution": dist,
    }

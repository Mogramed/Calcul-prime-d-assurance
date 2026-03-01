from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.insurance_pricing import analytics as ds
from src.insurance_pricing import training as v2

from . import dualtrack_impl as w23
from . import gap_diagnosis_impl as w22
from .common import (
    rmse as _rmse,
    safe_float as _safe_float,
    safe_read_csv as _safe_read_csv,
    safe_read_parquet as _safe_read_parquet,
)


ARTIFACT_V24_DIR = Path("artifacts") / "v2_4_tail_recovery"
EPS = 1e-9
DEFAULT_TE_COLS = ["code_postal", "cp3", "modele_vehicule", "marque_modele"]


def ensure_v24_dir(root: str | Path = ".") -> Path:
    return v2.ensure_dir(Path(root) / ARTIFACT_V24_DIR)


def _rmse_subset(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return float("nan")
    return _rmse(np.asarray(y_true)[m], np.asarray(y_pred)[m])


def _make_jsonable_value(v: Any) -> Any:
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_make_jsonable_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _make_jsonable_value(val) for k, val in v.items()}
    return v


def _drop_array_cols_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    drop_cols = []
    for c in df.columns:
        s = df[c]
        if s.dtype == "object":
            try:
                sample = next((x for x in s.tolist() if x is not None and not (isinstance(x, float) and np.isnan(x))), None)
            except Exception:
                sample = None
            if isinstance(sample, (np.ndarray, list, tuple, dict)):
                drop_cols.append(c)
    return df.drop(columns=drop_cols, errors="ignore")


def _ensure_run_id(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "run_id" not in out.columns:
        out["run_id"] = v2.make_run_id(out)
    return out


def load_tail_recovery_context(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    ctx = w23.load_existing_artifacts(root)
    out_dir = ensure_v24_dir(root)
    ctx["artifact_v24"] = out_dir

    a_v23 = root / "artifacts" / "v2_3_dualtrack_quick"
    ctx["v23_overfit_summary"] = json.loads((a_v23 / "overfit_ood_queue_summary.json").read_text(encoding="utf-8")) if (a_v23 / "overfit_ood_queue_summary.json").exists() else {}
    ctx["v23_overfit_diag"] = _safe_read_csv(a_v23 / "overfit_ood_queue_diagnosis.csv")
    ctx["v23_scale_sweep"] = _safe_read_csv(a_v23 / "direct_tweedie_scale_sweep.csv")
    ctx["v23_blend_sweep"] = _safe_read_csv(a_v23 / "direct_tweedie_blend_sweep.csv")
    ctx["v23_direct_registry"] = _safe_read_csv(a_v23 / "direct_tweedie_cv_registry.csv")
    ctx["v23_oof_compare_bridge"] = _safe_read_parquet(a_v23 / "oof_compare_bridge.parquet")
    ctx["v23_submission_robust"] = _safe_read_csv(a_v23 / "submission_v2_3_robust.csv")
    ctx["v23_submission_lb_challenger"] = _safe_read_csv(a_v23 / "submission_v2_3_lb_challenger.csv")
    ctx["v23_decision_md"] = (a_v23 / "submission_decision_v2_3_dualtrack.md").read_text(encoding="utf-8") if (a_v23 / "submission_decision_v2_3_dualtrack.md").exists() else ""
    return ctx


def _extract_run_row_from_v2(ctx: Mapping[str, Any], run_id: str, split: str = "primary_time") -> Optional[dict[str, Any]]:
    rr = pd.DataFrame(ctx.get("v2_run_registry", pd.DataFrame()))
    if rr.empty:
        return None
    return w22.extract_row_from_run_table(rr, run_id)


def extract_base_run_predictions(
    ctx: Mapping[str, Any],
    run_id: Optional[str] = None,
    split: str = "primary_time",
) -> dict[str, Any]:
    v2_sel = pd.DataFrame(ctx.get("v2_selected", pd.DataFrame()))
    if run_id is None:
        if not v2_sel.empty and "run_id" in v2_sel.columns:
            run_id = str(v2_sel.iloc[0]["run_id"])
        else:
            rr = pd.DataFrame(ctx.get("v2_run_registry", pd.DataFrame()))
            best = rr[(rr.get("level", "run").astype(str) == "run") & (rr.get("split", split).astype(str) == split)].sort_values("rmse_prime").head(1)
            if best.empty:
                raise ValueError("No base V2 run found in artifacts.")
            run_id = str(best.iloc[0]["run_id"]) if "run_id" in best.columns else str(v2.make_run_id(best.iloc[[0]]).iloc[0])

    v2_oof = _ensure_run_id(pd.DataFrame(ctx.get("v2_oof", pd.DataFrame())))
    if v2_oof.empty:
        raise ValueError("Missing artifacts/v2/oof_predictions_v2.parquet")

    pred_all = v2_oof[v2_oof["run_id"].astype(str) == str(run_id)].copy()
    if pred_all.empty:
        raise ValueError(f"Run not found in v2_oof: {run_id}")

    pred_by_split: dict[str, pd.DataFrame] = {}
    for sp in ["primary_time", "secondary_group", "aux_blocked5"]:
        d = pred_all[pred_all["split"].astype(str) == sp].copy()
        if not d.empty:
            pred_by_split[sp] = d

    base_row = _extract_run_row_from_v2(ctx, str(run_id), split=split) or {}
    if base_row:
        base_row["run_id"] = str(run_id)

    ds_diag = {}
    try:
        ds_diag = ds.compute_oof_model_diagnostics(v2_oof, run_id=str(run_id), split=split, decile_mode="zero_aware")
    except Exception:
        ds_diag = {}

    return {
        "base_run_id": str(run_id),
        "base_run_row": base_row,
        "pred_all": pred_all,
        "pred_by_split": pred_by_split,
        "primary_split": split,
        "ds_diag": ds_diag,
    }


def fit_tail_rank_scaler(
    oof_df: pd.DataFrame,
    *,
    threshold_q: float,
    lambda_: float,
    gamma_: float,
    on: str = "pred_sev",
) -> dict:
    d = oof_df.copy()
    col = str(on)
    if col not in d.columns or "y_sev" not in d.columns:
        return {"kind": "identity", "reason": "missing_columns"}
    pos = (pd.to_numeric(d["y_sev"], errors="coerce") > 0) & pd.to_numeric(d[col], errors="coerce").notna()
    x = pd.to_numeric(d.loc[pos, col], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) < 50:
        return {"kind": "identity", "reason": "insufficient_positive_samples"}
    rank_ref = np.sort(x)
    thr_q = float(threshold_q)
    thr_val = float(np.quantile(rank_ref, thr_q))
    return {
        "kind": "tail_rank_scaler",
        "on": col,
        "threshold_q": thr_q,
        "threshold_val": thr_val,
        "lambda": float(lambda_),
        "gamma": float(gamma_),
        "rank_ref_sorted": rank_ref,
        "n_ref": int(len(rank_ref)),
    }


def apply_tail_rank_scaler(
    pred: np.ndarray,
    *,
    rank_ref: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    p = np.maximum(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if str(params.get("kind", "identity")) != "tail_rank_scaler":
        return p
    ref = np.sort(np.asarray(rank_ref, dtype=float))
    ref = ref[np.isfinite(ref) & (ref >= 0)]
    if len(ref) == 0:
        return p
    thr_q = float(params.get("threshold_q", 0.95))
    lam = float(params.get("lambda", 0.0))
    gam = float(params.get("gamma", 1.0))
    rank = np.searchsorted(ref, p, side="right") / max(len(ref), 1)
    z = np.clip((rank - thr_q) / max(1.0 - thr_q, 1e-9), 0.0, 1.0)
    mult = 1.0 + lam * np.power(z, gam)
    out = np.maximum(p * mult, 0.0)
    return out


def fit_tail_mapper_thresholded(
    oof_pred_pos: np.ndarray,
    y_pos: np.ndarray,
    *,
    threshold_q: float,
    mode: str = "piecewise_monotone",
    min_samples_tail: int = 150,
) -> dict:
    x = np.asarray(oof_pred_pos, dtype=float)
    y = np.asarray(y_pos, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    x = x[mask]
    y = y[mask]
    if len(x) < max(min_samples_tail, 50):
        return {"kind": "identity", "reason": "insufficient_samples"}
    thr_q = float(threshold_q)
    x_thr = float(np.quantile(x, thr_q))
    tail = x >= x_thr
    if int(np.sum(tail)) < int(min_samples_tail):
        return {"kind": "identity", "reason": "insufficient_tail_samples", "threshold_q": thr_q, "x_threshold": x_thr}

    base_mapper = v2.fit_tail_mapper_safe(x[tail], y[tail], min_samples=min_samples_tail)
    if str(base_mapper.get("kind", "identity")) == "identity":
        return {"kind": "identity", "reason": "base_mapper_identity", "threshold_q": thr_q, "x_threshold": x_thr}

    mapped_thr = float(v2.apply_tail_mapper_safe(base_mapper, np.array([x_thr], dtype=float))[0])
    return {
        "kind": "tail_mapper_thresholded",
        "mode": str(mode),
        "threshold_q": thr_q,
        "x_threshold": x_thr,
        "mapped_threshold": mapped_thr,
        "base_mapper": base_mapper,
        "min_samples_tail": int(min_samples_tail),
    }


def apply_tail_mapper_thresholded(pred_pos: np.ndarray, mapper: Mapping[str, Any]) -> np.ndarray:
    p = np.asarray(pred_pos, dtype=float)
    p = np.maximum(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if str(mapper.get("kind", "identity")) != "tail_mapper_thresholded":
        return p
    x_thr = float(mapper.get("x_threshold", 0.0))
    mapped_thr = float(mapper.get("mapped_threshold", x_thr))
    base_mapper = mapper.get("base_mapper", {"kind": "identity"})
    out = p.copy()
    hi = p > x_thr
    if not np.any(hi):
        return out
    raw_hi = v2.apply_tail_mapper_safe(base_mapper, p[hi])
    # Anchor continuity at threshold and enforce no reduction above threshold.
    adj_hi = x_thr + np.maximum(raw_hi - mapped_thr, 0.0)
    out[hi] = np.maximum(adj_hi, p[hi])
    return np.maximum(out, 0.0)


def _apply_tail_transform_to_pred_df(
    pred_df: pd.DataFrame,
    *,
    candidate_id: str,
    candidate_family: str,
    transform_kind: str,
    transform_params: Mapping[str, Any],
    base_run_id: str,
) -> pd.DataFrame:
    d = pred_df.copy()
    if d.empty:
        return d
    if "pred_sev" not in d.columns or "pred_prime" not in d.columns:
        raise ValueError("pred_df must contain pred_sev and pred_prime")

    sev = pd.to_numeric(d["pred_sev"], errors="coerce").to_numpy(dtype=float)
    sev = np.maximum(np.nan_to_num(sev, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if transform_kind == "tail_rank_scaler":
        rank_ref = np.asarray(transform_params.get("rank_ref_sorted", []), dtype=float)
        sev_new = apply_tail_rank_scaler(sev, rank_ref=rank_ref, params=transform_params)
    elif transform_kind == "tail_mapper_thresholded":
        sev_new = apply_tail_mapper_thresholded(sev, transform_params)
    else:
        sev_new = sev.copy()

    if "pred_freq" in d.columns and d["pred_freq"].notna().any():
        pf = pd.to_numeric(d["pred_freq"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
        prime_new = np.maximum(pf * sev_new, 0.0)
    else:
        prime_new = np.maximum(sev_new, 0.0)

    d["pred_sev"] = sev_new
    d["pred_prime"] = prime_new
    d["candidate_id"] = str(candidate_id)
    d["candidate_family"] = str(candidate_family)
    d["base_run_id"] = str(base_run_id)
    d["tail_transform_kind"] = str(transform_kind)
    d["run_id"] = str(candidate_id)
    return d


def _score_one_split_candidate(
    pred_df_split: pd.DataFrame,
    *,
    candidate_id: str,
    split: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "candidate_id": candidate_id,
        "split": split,
        "rmse_prime": np.nan,
        "rmse_prime_top1pct": np.nan,
        "q95_ratio_pos": np.nan,
        "q99_ratio_pos": np.nan,
        "body_rmse_proxy": np.nan,
        "tail_rmse_proxy": np.nan,
        "n_oof": 0,
        "n_oof_nonzero": 0,
    }
    d = pred_df_split.copy()
    d = d[(d["is_test"] == 0) & d["y_sev"].notna() & d["pred_prime"].notna()].copy()
    if d.empty:
        return out
    run_id_tmp = f"{candidate_id}|{split}"
    d["run_id"] = run_id_tmp
    d["split"] = split

    diag = ds.compute_oof_model_diagnostics(d, run_id=run_id_tmp, split=split, decile_mode="zero_aware")
    m = diag.get("metrics", pd.DataFrame())
    if not m.empty:
        row = m.iloc[0]
        out.update(
            {
                "rmse_prime": _safe_float(row.get("rmse_prime")),
                "rmse_prime_top1pct": _safe_float(row.get("rmse_prime_top1pct")),
                "q95_ratio_pos": _safe_float(row.get("q95_ratio_pos")),
                "q99_ratio_pos": _safe_float(row.get("q99_ratio_pos")),
                "n_oof": int(row.get("n", 0)),
                "n_oof_nonzero": int(row.get("n_nonzero", 0)),
            }
        )

    y = d["y_sev"].to_numpy(dtype=float)
    p = d["pred_prime"].to_numpy(dtype=float)
    q90_true = float(np.quantile(y, 0.90))
    q99_true = float(np.quantile(y, 0.99))
    out["body_rmse_proxy"] = _rmse_subset(y, p, y <= q90_true)
    out["tail_rmse_proxy"] = _rmse_subset(y, p, y >= q99_true)
    out["diag_tables"] = diag
    return out


def score_tail_candidate_multi_split(
    *,
    candidate_id: str,
    base_run_id: str,
    candidate_family: str,
    pred_df_all_splits: pd.DataFrame,
    transform_kind: str,
    transform_params: Mapping[str, Any],
) -> dict[str, Any]:
    d = pred_df_all_splits.copy()
    if d.empty:
        return {"registry_rows": pd.DataFrame(), "pred_df": pd.DataFrame(), "dist_df": pd.DataFrame(), "params": dict(transform_params)}

    scored_pred = _apply_tail_transform_to_pred_df(
        d,
        candidate_id=candidate_id,
        candidate_family=candidate_family,
        transform_kind=transform_kind,
        transform_params=transform_params,
        base_run_id=base_run_id,
    )

    split_rows: list[dict[str, Any]] = []
    for sp in ["primary_time", "secondary_group", "aux_blocked5"]:
        dspl = scored_pred[scored_pred["split"].astype(str) == sp].copy()
        if dspl.empty:
            continue
        srow = _score_one_split_candidate(dspl, candidate_id=candidate_id, split=sp)
        srow["candidate_family"] = candidate_family
        srow["base_run_id"] = base_run_id
        split_rows.append(srow)

    split_df = pd.DataFrame(split_rows)
    dist_df = v2.build_prediction_distribution_table(scored_pred) if not scored_pred.empty else pd.DataFrame()

    agg = {
        "candidate_id": candidate_id,
        "base_run_id": base_run_id,
        "candidate_family": candidate_family,
        "split": "multi",
        "rmse_prime": np.nan,
        "rmse_gap_secondary": np.nan,
        "rmse_gap_aux": np.nan,
        "rmse_split_std": np.nan,
        "rmse_prime_top1pct": np.nan,
        "q95_ratio_pos": np.nan,
        "q99_ratio_pos": np.nan,
        "body_rmse_proxy": np.nan,
        "tail_rmse_proxy": np.nan,
        "distribution_alignment_penalty": np.nan,
        "tail_overcorrection_flag": np.nan,
        "tail_undercoverage_flag": np.nan,
        "selection_score_tail": np.nan,
        "selection_status": "candidate",
        "pred_df_candidate": scored_pred,
    }
    if not split_df.empty:
        piv = split_df.set_index("split")
        rmse_primary = _safe_float(piv["rmse_prime"].get("primary_time"))
        rmse_sec = _safe_float(piv["rmse_prime"].get("secondary_group"))
        rmse_aux = _safe_float(piv["rmse_prime"].get("aux_blocked5"))
        agg["rmse_prime"] = rmse_primary
        agg["rmse_gap_secondary"] = rmse_sec - rmse_primary if np.isfinite(rmse_sec) and np.isfinite(rmse_primary) else np.nan
        agg["rmse_gap_aux"] = rmse_aux - rmse_primary if np.isfinite(rmse_aux) and np.isfinite(rmse_primary) else np.nan
        rmses = np.asarray([rmse_primary, rmse_sec, rmse_aux], dtype=float)
        agg["rmse_split_std"] = float(np.nanstd(rmses, ddof=0)) if np.sum(np.isfinite(rmses)) >= 2 else np.nan
        agg["rmse_prime_top1pct"] = _safe_float(piv["rmse_prime_top1pct"].get("primary_time"))
        agg["q95_ratio_pos"] = _safe_float(piv["q95_ratio_pos"].get("primary_time"))
        agg["q99_ratio_pos"] = _safe_float(piv["q99_ratio_pos"].get("primary_time"))
        agg["body_rmse_proxy"] = _safe_float(piv["body_rmse_proxy"].get("primary_time"))
        agg["tail_rmse_proxy"] = _safe_float(piv["tail_rmse_proxy"].get("primary_time"))

    if not dist_df.empty:
        # Use primary_time alignment as main reference
        align = w22.compute_distribution_alignment_from_dist(dist_df, run_id=str(candidate_id), split="primary_time")
        agg.update({k: align.get(k) for k in align.keys()})
        agg["distribution_alignment_penalty"] = _safe_float(align.get("distribution_alignment_penalty"))

    q99 = _safe_float(agg.get("q99_ratio_pos"))
    tail_under_penalty = 15.0 * max(0.0, 0.45 - q99) if np.isfinite(q99) else 15.0
    tail_over_penalty = 20.0 * max(0.0, q99 - 0.85) if np.isfinite(q99) else 0.0
    gap_sec_pos = max(0.0, _safe_float(agg.get("rmse_gap_secondary"), 0.0))
    gap_aux_pos = max(0.0, _safe_float(agg.get("rmse_gap_aux"), 0.0))
    align_pen = _safe_float(agg.get("distribution_alignment_penalty"), 0.0)
    base_rmse = _safe_float(agg.get("rmse_prime"))
    if not np.isfinite(base_rmse):
        base_rmse = 1e9
    agg["tail_undercoverage_flag"] = int(np.isfinite(q99) and q99 < 0.10)
    agg["tail_overcorrection_flag"] = int(np.isfinite(q99) and q99 > 0.85)
    agg["selection_score_tail"] = float(base_rmse + 0.5 * gap_sec_pos + 0.5 * gap_aux_pos + tail_under_penalty + tail_over_penalty + (align_pen if np.isfinite(align_pen) else 0.0))

    registry_rows = split_df.copy()
    for c, v in agg.items():
        if c not in ["pred_df_candidate"] and c not in registry_rows.columns:
            if c in {"candidate_id", "base_run_id", "candidate_family", "selection_status"}:
                registry_rows[c] = v
    multi_row = {k: v for k, v in agg.items() if k != "pred_df_candidate"}
    registry_rows = pd.concat([registry_rows, pd.DataFrame([multi_row])], ignore_index=True, sort=False)
    return {
        "registry_rows": registry_rows,
        "pred_df": scored_pred,
        "dist_df": dist_df,
        "params": dict(transform_params),
        "multi_row": multi_row,
    }


def _load_train_test_and_feature_sets(ctx: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, v2.DatasetBundle], dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]:
    train_raw, test_raw = v2.load_train_test(Path(ctx.get("data_dir", Path(".") / "data")))
    feature_sets = v2.prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
    splits = v2.build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")
    return train_raw, test_raw, feature_sets, splits


def fit_severity_weighted_tail_variant(
    ctx: Mapping[str, Any],
    *,
    base_run_row: Mapping[str, Any],
    seed: int = 42,
    feature_set_override: Optional[str] = None,
    sev_params_override: Optional[Mapping[str, Any]] = None,
    out_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    # Phase B wrapper: rerun the same base spec but force severity_mode='weighted_tail'
    train_raw, test_raw, feature_sets, splits = _load_train_test_and_feature_sets(ctx)
    cfg_lookup = w22.build_cfg_lookup()
    row = dict(base_run_row)
    if "run_id" not in row:
        row["run_id"] = str(base_run_row.get("run_id", "base_row_missing"))
    row["severity_mode"] = "weighted_tail"
    if feature_set_override is not None:
        row["feature_set"] = str(feature_set_override)
    spec = w22.build_spec_from_row(
        row,
        cfg_lookup=cfg_lookup,
        te_cols=DEFAULT_TE_COLS,
        feature_set_override=str(feature_set_override) if feature_set_override else None,
    )
    if sev_params_override:
        spec["sev_params"] = {**dict(spec.get("sev_params", {})), **dict(sev_params_override)}
    fold_df, run_df, pred_df = v2.run_benchmark(spec=spec, bundle=feature_sets, splits=splits, seed=seed)
    dist_df = v2.build_prediction_distribution_table(pred_df) if not pred_df.empty else pd.DataFrame()
    scored_df = w22.score_quick_runs(run_df, pred_df, dist_df, seed=seed, n_sim_shakeup=0, run_shakeup=False) if not run_df.empty else pd.DataFrame()
    if scored_df.empty:
        return pd.DataFrame()
    scored_df = scored_df.copy()
    scored_df["candidate_family"] = "sev_retrain_weighted_tail"
    scored_df["base_run_id"] = str(base_run_row.get("run_id"))
    scored_df["candidate_id"] = "sev_retrain_weighted_tail"
    scored_df["split"] = "multi"
    scored_df["rmse_prime"] = pd.to_numeric(scored_df.get("rmse_primary_time"), errors="coerce")
    scored_df["rmse_prime_top1pct"] = pd.to_numeric(scored_df.get("rmse_prime_top1pct"), errors="coerce")
    scored_df["selection_score_tail"] = pd.to_numeric(scored_df.get("selection_score_quick"), errors="coerce")
    if out_dir is not None:
        out = v2.ensure_dir(out_dir)
        _drop_array_cols_for_csv(scored_df).to_csv(out / "weighted_tail_variant_registry.csv", index=False)
        if not pred_df.empty:
            pred_df.to_parquet(out / "weighted_tail_variant_oof_predictions.parquet", index=False)
    return scored_df


def build_tail_pareto_front(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()
    d = candidates_df.copy()
    if "split" in d.columns:
        d = d[d["split"].astype(str) == "multi"].copy()
    if d.empty:
        return pd.DataFrame()
    d["pareto_tail_distance"] = (pd.to_numeric(d.get("q99_ratio_pos", np.nan), errors="coerce") - 0.60).abs()
    for c in ["rmse_prime", "rmse_split_std", "rmse_prime_top1pct", "pareto_tail_distance"]:
        if c not in d.columns:
            d[c] = np.nan
    d = d[pd.to_numeric(d["rmse_prime"], errors="coerce").notna()].copy()
    if d.empty:
        return pd.DataFrame()
    vals = d[["rmse_prime", "rmse_split_std", "rmse_prime_top1pct", "pareto_tail_distance"]].to_numpy(dtype=float)
    finite_fill = np.nanmax(np.where(np.isfinite(vals), vals, np.nan), axis=0)
    finite_fill = np.where(np.isfinite(finite_fill), finite_fill, 1e9)
    vals = np.where(np.isfinite(vals), vals, finite_fill + 1.0)
    keep = np.ones(len(d), dtype=bool)
    for i in range(len(d)):
        if not keep[i]:
            continue
        vi = vals[i]
        dominated = np.all(vals <= vi + 1e-12, axis=1) & np.any(vals < vi - 1e-12, axis=1)
        if np.any(dominated):
            keep[i] = False
    out = d.loc[keep].copy()
    out = out.sort_values(["rmse_prime", "pareto_tail_distance", "rmse_split_std"], na_position="last").reset_index(drop=True)
    return out


def select_tail_recovery_submissions(
    candidates_df: pd.DataFrame,
    policy: str = "balanced_tail",
) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame()
    d = candidates_df.copy()
    if "split" in d.columns:
        d = d[d["split"].astype(str) == "multi"].copy()
    if d.empty:
        return pd.DataFrame()

    if "selection_score_tail" not in d.columns:
        # Backfill a simple balanced-tail score
        q99 = pd.to_numeric(d.get("q99_ratio_pos", np.nan), errors="coerce")
        rmse_primary = pd.to_numeric(d.get("rmse_prime", d.get("rmse_primary_time", np.nan)), errors="coerce")
        gap_sec = np.maximum(pd.to_numeric(d.get("rmse_gap_secondary", np.nan), errors="coerce").fillna(0.0), 0.0)
        gap_aux = np.maximum(pd.to_numeric(d.get("rmse_gap_aux", np.nan), errors="coerce").fillna(0.0), 0.0)
        dist_pen = pd.to_numeric(d.get("distribution_alignment_penalty", np.nan), errors="coerce").fillna(0.0)
        d["selection_score_tail"] = (
            rmse_primary.fillna(1e9)
            + 0.5 * gap_sec
            + 0.5 * gap_aux
            + 15.0 * np.maximum(0.0, 0.45 - q99.fillna(0.0))
            + 20.0 * np.maximum(0.0, q99.fillna(0.0) - 0.85)
            + dist_pen
        )

    q99 = pd.to_numeric(d.get("q99_ratio_pos", np.nan), errors="coerce")
    d["tail_undercoverage_flag"] = d.get("tail_undercoverage_flag", (q99 < 0.10).astype(int))
    d["tail_overcorrection_flag"] = d.get("tail_overcorrection_flag", (q99 > 0.85).astype(int))
    d["passes_guardrails"] = (
        (pd.to_numeric(d.get("rmse_gap_secondary", np.nan), errors="coerce").fillna(0.0) <= 1.0)
        & (pd.to_numeric(d.get("rmse_gap_aux", np.nan), errors="coerce").fillna(0.0) <= 1.0)
        & (pd.to_numeric(d.get("distribution_alignment_penalty", np.nan), errors="coerce").fillna(0.0) < 5.0)
        & (pd.to_numeric(d.get("tail_overcorrection_flag", 0), errors="coerce").fillna(0.0) <= 0)
        & (q99.fillna(0.0) >= 0.10)
    )

    robust_pool = d[d["passes_guardrails"]].copy()
    if robust_pool.empty:
        robust_pool = d.sort_values(["selection_score_tail", "rmse_prime"], na_position="last").head(1).copy()
    robust_row = robust_pool.sort_values(["selection_score_tail", "rmse_prime"], na_position="last").iloc[0].to_dict() if not robust_pool.empty else None

    challenger_pool = d.copy()
    # Prefer candidates with stronger tail than base but without overcorrection.
    challenger_pool["tail_strength_rank"] = -(pd.to_numeric(challenger_pool.get("q99_ratio_pos", np.nan), errors="coerce").fillna(0.0))
    challenger_pool = challenger_pool.sort_values(["tail_strength_rank", "rmse_prime", "selection_score_tail"], na_position="last")
    challenger_row = challenger_pool.iloc[0].to_dict() if not challenger_pool.empty else None

    rows = []
    if robust_row is not None:
        robust_row["role"] = "robust"
        robust_row["selection_status"] = "selected_robust"
        robust_row["risk_tag"] = "robust"
        rows.append(robust_row)
    if challenger_row is not None and (robust_row is None or str(challenger_row.get("candidate_id")) != str(robust_row.get("candidate_id"))):
        challenger_row["role"] = "lb_challenger"
        challenger_row["selection_status"] = "selected_challenger"
        challenger_row["risk_tag"] = "public_private_risk"
        rows.append(challenger_row)
    return pd.DataFrame(rows)


def _fit_base_fulltrain_components(
    ctx: Mapping[str, Any],
    *,
    base_run_row: Mapping[str, Any],
    seed_default: int = 42,
) -> dict[str, Any]:
    train_raw, test_raw, feature_sets, _ = _load_train_test_and_feature_sets(ctx)
    cfg_lookup = w22.build_cfg_lookup()
    te_cols = DEFAULT_TE_COLS
    r = dict(base_run_row)
    run_id = str(r["run_id"])
    fs_name = str(r.get("feature_set", "base_v2"))
    seed = int(float(r.get("seed", seed_default)))
    family = str(r.get("family", "two_part_classic"))
    calibration = str(r.get("calibration", "none"))
    tail_mapper_name = str(r.get("tail_mapper", "none"))
    spec = w22.build_spec_from_row(r, cfg_lookup=cfg_lookup, te_cols=te_cols)
    bundle = feature_sets[fs_name]
    out = v2.fit_full_predict_fulltrain(spec=spec, bundle=bundle, seed=seed, complexity={})
    test_freq = out["test_freq"].copy()
    test_sev = out["test_sev"].copy()

    v2_oof = _ensure_run_id(pd.DataFrame(ctx.get("v2_oof", pd.DataFrame())))
    oo = v2_oof[
        (v2_oof["is_test"] == 0)
        & (v2_oof["split"].astype(str) == "primary_time")
        & (v2_oof["run_id"].astype(str) == run_id)
    ].copy()

    if calibration != "none" and len(oo):
        ok = oo["pred_freq"].notna() & oo["y_freq"].notna()
        if ok.any():
            cal = v2.fit_calibrator(
                oo.loc[ok, "pred_freq"].to_numpy(),
                oo.loc[ok, "y_freq"].to_numpy(),
                method=calibration,
            )
            test_freq = v2.apply_calibrator(cal, test_freq, method=calibration)

    if tail_mapper_name != "none" and family != "direct_tweedie" and len(oo):
        pos = (oo["y_freq"] == 1) & oo["pred_sev"].notna() & oo["y_sev"].notna()
        if int(pos.sum()) >= 80:
            mapper = v2.fit_tail_mapper_safe(oo.loc[pos, "pred_sev"].to_numpy(), oo.loc[pos, "y_sev"].to_numpy())
            sev_before = test_sev.copy()
            test_sev = v2.apply_tail_mapper_safe(mapper, test_sev)
            std_ratio = float(np.std(test_sev) / max(np.std(sev_before), 1e-9))
            q99_oof = float(np.nanquantile(oo.loc[pos, "pred_sev"].to_numpy(), 0.99))
            q99_test = float(np.nanquantile(test_sev, 0.99))
            if (std_ratio < 0.70) or (q99_test < 0.60 * q99_oof):
                test_sev = sev_before

    pred = np.maximum(out["test_prime"], 0.0) if family == "direct_tweedie" else np.maximum(test_freq * test_sev, 0.0)
    sub = v2.build_submission(test_raw["index"], pred)
    return {
        "run_id": run_id,
        "train_raw": train_raw,
        "test_raw": test_raw,
        "test_freq": test_freq,
        "test_sev": test_sev,
        "test_prime": pred,
        "sub": sub,
    }


def _apply_candidate_transform_to_test_components(
    test_freq: np.ndarray,
    test_sev: np.ndarray,
    *,
    candidate_row: Mapping[str, Any],
    transform_store: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    cid = str(candidate_row.get("candidate_id"))
    payload = transform_store.get(cid, {})
    kind = str(payload.get("transform_kind", "identity"))
    params = payload.get("transform_params", {"kind": "identity"})
    sev = np.maximum(np.asarray(test_sev, dtype=float), 0.0)
    if kind == "tail_rank_scaler":
        rank_ref = np.asarray(params.get("rank_ref_sorted", []), dtype=float)
        sev = apply_tail_rank_scaler(sev, rank_ref=rank_ref, params=params)
    elif kind == "tail_mapper_thresholded":
        sev = apply_tail_mapper_thresholded(sev, params)
    pred = np.maximum(np.asarray(test_freq, dtype=float) * sev, 0.0)
    return pred


def _save_submission_from_pred(index_values: pd.Series | np.ndarray, pred: np.ndarray, path: str | Path) -> Path:
    sub = v2.build_submission(pd.Series(index_values), np.maximum(np.asarray(pred, dtype=float), 0.0))
    p = Path(path)
    v2.ensure_dir(p.parent)
    sub.to_csv(p, index=False)
    return p


def write_tail_decision_report(
    *,
    base_info: Mapping[str, Any],
    candidates_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    out_dir: str | Path = ARTIFACT_V24_DIR,
) -> Path:
    out = v2.ensure_dir(out_dir)
    lines: list[str] = []
    lines.append("# Submission decision V2.4 Tail Recovery")
    lines.append("")
    lines.append("## 1) Contexte")
    lines.append(f"- Base run V2 (ancre robuste): `{base_info.get('base_run_id')}`")
    lines.append("- Objectif: corriger la queue (q95/q99) sans casser RMSE global ni la stabilité inter-splits.")
    lines.append("")
    lines.append("## 2) Résumé candidats")
    if candidates_df is not None and not candidates_df.empty:
        d = candidates_df.copy()
        if "split" in d.columns:
            d = d[d["split"].astype(str) == "multi"]
        cols = [c for c in ["candidate_id", "candidate_family", "rmse_prime", "rmse_gap_secondary", "rmse_gap_aux", "q95_ratio_pos", "q99_ratio_pos", "rmse_prime_top1pct", "selection_score_tail"] if c in d.columns]
        if not d.empty and cols:
            lines.append("")
            lines.append(d[cols].sort_values(["selection_score_tail", "rmse_prime"], na_position="last").to_markdown(index=False))
            lines.append("")
    else:
        lines.append("- Aucun candidat évalué")
        lines.append("")
    lines.append("## 3) Front Pareto")
    if pareto_df is not None and not pareto_df.empty:
        cols = [c for c in ["candidate_id", "candidate_family", "rmse_prime", "q99_ratio_pos", "rmse_split_std", "rmse_prime_top1pct"] if c in pareto_df.columns]
        lines.append("")
        lines.append(pareto_df[cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- Front Pareto indisponible")
        lines.append("")
    lines.append("## 4) Sélection finale")
    if selected_df is not None and not selected_df.empty:
        cols = [c for c in ["role", "candidate_id", "candidate_family", "risk_tag", "rmse_prime", "q99_ratio_pos", "selection_score_tail"] if c in selected_df.columns]
        lines.append("")
        lines.append(selected_df[cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- Fallback baseline (aucun candidat retenu)")
        lines.append("")
    lines.append("## 5) Rappel méthode")
    lines.append("- Pas de scaling global seul pour corriger la queue.")
    lines.append("- Priorité aux corrections tail-only au-dessus d'un seuil, avec garde-fous RMSE + stabilité.")
    lines.append("- Overfitting CV non conclu sans preuve inter-splits.")
    lines.append("")
    path = out / "submission_decision_v2_4_tail.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def materialize_tail_recovery_submissions(
    ctx: Mapping[str, Any],
    *,
    base_run_row: Mapping[str, Any],
    selected_df: pd.DataFrame,
    transform_store: Mapping[str, Mapping[str, Any]],
    out_dir: str | Path = ARTIFACT_V24_DIR,
) -> dict[str, Any]:
    out = v2.ensure_dir(out_dir)
    base_payload = _fit_base_fulltrain_components(ctx, base_run_row=base_run_row)
    test_index = base_payload["test_raw"]["index"]
    test_freq = base_payload["test_freq"]
    test_sev = base_payload["test_sev"]

    submission_paths: dict[str, Path] = {}
    pred_audits: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []

    if selected_df is None or selected_df.empty:
        # fallback to base
        pred_base = np.maximum(np.asarray(base_payload["test_prime"], dtype=float), 0.0)
        p = _save_submission_from_pred(test_index, pred_base, out / "submission_v2_4_robust.csv")
        submission_paths["robust"] = p
        pred_audits.append({"role": "robust", **v2.compute_prediction_distribution_audit(pred_base, run_id="v2_4_fallback_base", split="test", sample="test")})
        return {"base_payload": base_payload, "submission_paths": submission_paths, "pred_audits": pd.DataFrame(pred_audits), "generated_rows": pd.DataFrame(generated_rows)}

    for _, row in selected_df.iterrows():
        role = str(row.get("role"))
        cid = str(row.get("candidate_id"))
        pred = _apply_candidate_transform_to_test_components(
            test_freq=test_freq,
            test_sev=test_sev,
            candidate_row=row,
            transform_store=transform_store,
        )
        if role == "robust":
            fname = "submission_v2_4_robust.csv"
        elif role == "lb_challenger":
            fname = "submission_v2_4_lb_challenger.csv"
        else:
            fname = f"submission_{role}.csv"
        path = _save_submission_from_pred(test_index, pred, out / fname)
        submission_paths[role] = path
        pred_audits.append({"role": role, "candidate_id": cid, **v2.compute_prediction_distribution_audit(pred, run_id=cid, split="test", sample="test")})
        generated_rows.append({"role": role, "candidate_id": cid, "file": str(path), "n": int(len(pred)), "pred_mean": float(np.mean(pred)), "pred_q99": float(np.quantile(pred, 0.99))})

    pred_audits_df = pd.DataFrame(pred_audits)
    if not pred_audits_df.empty:
        pred_audits_df.to_csv(out / "pred_distribution_compare_v2_4.csv", index=False)
    return {
        "base_payload": base_payload,
        "submission_paths": submission_paths,
        "pred_audits": pred_audits_df,
        "generated_rows": pd.DataFrame(generated_rows),
    }


def train_run(config_path: str) -> dict:
    from src.insurance_pricing import train_run as _train_run

    return _train_run(config_path)

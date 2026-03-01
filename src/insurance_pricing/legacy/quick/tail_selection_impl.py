from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from src.insurance_pricing import training as v2

from . import gap_diagnosis_impl as w22
from . import tail_recovery_impl as tr
from .common import (
    safe_read_csv as _safe_read_csv,
    safe_read_json as _safe_read_json,
)


ARTIFACT_V24_DIR = Path("artifacts") / "v2_4_tail_recovery"
ARTIFACT_V241_DIR = Path("artifacts") / "v2_4_1_tail_recovery"


def ensure_v241_dir(root: str | Path = ".") -> Path:
    return v2.ensure_dir(Path(root) / ARTIFACT_V241_DIR)


def _coerce_num(df: pd.DataFrame, col: str, fill: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), fill), index=df.index, dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    if np.isnan(fill):
        return s
    return s.fillna(fill)


def load_v24_outputs(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    out_dir = ensure_v241_dir(root)
    ctx = tr.load_tail_recovery_context(root)
    v24_dir = root / ARTIFACT_V24_DIR

    tail_registry = _safe_read_csv(v24_dir / "tail_candidates_registry.csv")
    tail_pareto = _safe_read_csv(v24_dir / "tail_pareto_front.csv")
    tail_diag = _safe_read_csv(v24_dir / "tail_diagnostics_by_split.csv")
    tail_transform_params = _safe_read_json(v24_dir / "tail_transform_params.json")
    decision_md = (
        (v24_dir / "submission_decision_v2_4_tail.md").read_text(encoding="utf-8")
        if (v24_dir / "submission_decision_v2_4_tail.md").exists()
        else ""
    )

    base = tr.extract_base_run_predictions(ctx, run_id=None, split="primary_time")
    multi = (
        tail_registry[tail_registry["split"].astype(str) == "multi"].copy()
        if not tail_registry.empty and "split" in tail_registry.columns
        else tail_registry.copy()
    )

    ctx.update(
        {
            "root": root,
            "artifact_v24": v24_dir,
            "artifact_v241": out_dir,
            "tail_candidates_registry_v24": tail_registry,
            "tail_candidates_multi_v24": multi,
            "tail_pareto_v24": tail_pareto,
            "tail_diag_v24": tail_diag,
            "tail_transform_store_v24": tail_transform_params,
            "tail_decision_md_v24": decision_md,
            "base_v24": base,
        }
    )
    return ctx


def mark_identity_and_duplicate_candidates(
    df: pd.DataFrame,
    pred_store: Mapping[str, Any],
    tol: float = 1e-10,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    if "candidate_id" not in d.columns:
        raise ValueError("candidate_id is required in candidate dataframe.")

    if "split" in d.columns:
        d = d[d["split"].astype(str) == "multi"].copy()

    metric_cols = [
        "rmse_prime",
        "rmse_prime_top1pct",
        "q95_ratio_pos",
        "q99_ratio_pos",
        "pred_q90_oof",
        "pred_q99_oof",
    ]
    for c in metric_cols:
        if c not in d.columns:
            d[c] = np.nan

    if "candidate_family" not in d.columns:
        d["candidate_family"] = "unknown"

    # Baseline identity row
    if (d["candidate_id"].astype(str) == "baseline_identity").any():
        baseline_row = d[d["candidate_id"].astype(str) == "baseline_identity"].iloc[0]
    else:
        baseline_row = d.sort_values("rmse_prime", na_position="last").iloc[0]
    baseline_id = str(baseline_row["candidate_id"])

    d["is_identity_candidate"] = 0
    d["identity_reason"] = ""
    d["is_duplicate_candidate"] = 0
    d["duplicate_of"] = ""

    baseline_vals = {c: float(pd.to_numeric(pd.Series([baseline_row[c]]), errors="coerce").iloc[0]) for c in metric_cols}

    # First pass identity
    for idx, row in d.iterrows():
        cid = str(row["candidate_id"])
        cfamily = str(row.get("candidate_family", ""))
        payload = pred_store.get(cid, {})
        kind = ""
        if isinstance(payload, Mapping):
            kind = str(payload.get("transform_kind", ""))
            if not kind and isinstance(payload.get("transform_params"), Mapping):
                kind = str(payload["transform_params"].get("kind", ""))

        reason = ""
        if cid == baseline_id or cid == "baseline_identity":
            reason = "baseline_identity"
        elif cfamily == "baseline_identity":
            reason = "baseline_family"
        elif kind == "identity":
            reason = "transform_identity"
        else:
            all_close = True
            for c in metric_cols:
                v = float(pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0])
                b = baseline_vals[c]
                if np.isfinite(v) and np.isfinite(b):
                    if abs(v - b) > max(tol, 1e-8):
                        all_close = False
                        break
                else:
                    all_close = False
                    break
            if all_close:
                reason = "metrics_equal_to_baseline"

        if reason:
            d.at[idx, "is_identity_candidate"] = 1
            d.at[idx, "identity_reason"] = reason

    # Duplicate detection by metric signature, then optional pred-store arrays
    seen_signatures: Dict[tuple, str] = {}
    seen_arrays: Dict[str, np.ndarray] = {}
    order = d.sort_values(["is_identity_candidate", "rmse_prime"], na_position="last").index.tolist()
    for idx in order:
        row = d.loc[idx]
        cid = str(row["candidate_id"])
        if int(row.get("is_identity_candidate", 0)) == 1:
            continue

        sig = tuple(
            round(float(pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]), 10)
            if np.isfinite(float(pd.to_numeric(pd.Series([row.get(c)]), errors="coerce").iloc[0]))
            else np.nan
            for c in metric_cols
        )

        if sig in seen_signatures:
            d.at[idx, "is_duplicate_candidate"] = 1
            d.at[idx, "duplicate_of"] = seen_signatures[sig]
            continue

        payload = pred_store.get(cid, {})
        arr = None
        if isinstance(payload, Mapping) and isinstance(payload.get("pred_test"), np.ndarray):
            arr = np.asarray(payload["pred_test"], dtype=float)
        elif isinstance(payload, np.ndarray):
            arr = np.asarray(payload, dtype=float)

        duplicate = False
        duplicate_of = ""
        if arr is not None and arr.size > 0:
            for prev_cid, prev_arr in seen_arrays.items():
                if prev_arr.shape == arr.shape and np.nanmax(np.abs(prev_arr - arr)) <= tol:
                    duplicate = True
                    duplicate_of = prev_cid
                    break
            if not duplicate:
                seen_arrays[cid] = arr

        if duplicate:
            d.at[idx, "is_duplicate_candidate"] = 1
            d.at[idx, "duplicate_of"] = duplicate_of
        else:
            seen_signatures[sig] = cid

    return d.reset_index(drop=True)


def compute_tail_guardrails(
    df_multi: pd.DataFrame,
    baseline_id: str = "baseline_identity",
    rmse_tol: float = 0.15,
    q99_low: float = 0.45,
    q99_high: float = 0.70,
) -> pd.DataFrame:
    if df_multi is None or df_multi.empty:
        return pd.DataFrame()

    d = df_multi.copy()
    if "candidate_id" not in d.columns:
        raise ValueError("candidate_id is required in candidate dataframe.")

    if (d["candidate_id"].astype(str) == baseline_id).any():
        b = d[d["candidate_id"].astype(str) == baseline_id].iloc[0]
    else:
        b = d.sort_values("rmse_prime", na_position="last").iloc[0]
        baseline_id = str(b["candidate_id"])

    rmse_baseline = float(pd.to_numeric(pd.Series([b.get("rmse_prime")]), errors="coerce").iloc[0])
    top1_baseline = float(pd.to_numeric(pd.Series([b.get("rmse_prime_top1pct")]), errors="coerce").iloc[0])

    d["baseline_id"] = baseline_id
    d["rmse_baseline"] = rmse_baseline
    d["top1pct_baseline"] = top1_baseline
    d["rmse_delta_vs_baseline"] = _coerce_num(d, "rmse_prime") - rmse_baseline
    d["top1pct_delta_vs_baseline"] = _coerce_num(d, "rmse_prime_top1pct") - top1_baseline

    d["meets_rmse_tol"] = (_coerce_num(d, "rmse_prime") <= (rmse_baseline + rmse_tol)).astype(int)
    d["meets_gap_secondary"] = (_coerce_num(d, "rmse_gap_secondary", fill=0.0) <= 1.0).astype(int)
    d["meets_gap_aux"] = (_coerce_num(d, "rmse_gap_aux", fill=0.0) <= 1.0).astype(int)
    d["meets_q99_low"] = (_coerce_num(d, "q99_ratio_pos") >= q99_low).astype(int)
    d["meets_q99_high"] = (_coerce_num(d, "q99_ratio_pos") <= q99_high).astype(int)
    d["meets_dist_penalty"] = (_coerce_num(d, "distribution_alignment_penalty", fill=np.inf) <= 3.0).astype(int)
    d["meets_tail_over"] = (_coerce_num(d, "tail_overcorrection_flag", fill=0.0) == 0.0).astype(int)
    d["meets_not_identity"] = (_coerce_num(d, "is_identity_candidate", fill=0.0) == 0.0).astype(int)
    d["meets_not_duplicate"] = (_coerce_num(d, "is_duplicate_candidate", fill=0.0) == 0.0).astype(int)

    guard_cols = [
        "meets_rmse_tol",
        "meets_gap_secondary",
        "meets_gap_aux",
        "meets_q99_low",
        "meets_q99_high",
        "meets_dist_penalty",
        "meets_tail_over",
        "meets_not_identity",
        "meets_not_duplicate",
    ]
    d["hard_admissible"] = d[guard_cols].astype(int).min(axis=1).astype(int)

    def _fail_reasons(row: pd.Series) -> str:
        reasons = []
        if int(row["meets_rmse_tol"]) == 0:
            reasons.append("rmse_tol")
        if int(row["meets_gap_secondary"]) == 0:
            reasons.append("gap_secondary")
        if int(row["meets_gap_aux"]) == 0:
            reasons.append("gap_aux")
        if int(row["meets_q99_low"]) == 0:
            reasons.append("q99_too_low")
        if int(row["meets_q99_high"]) == 0:
            reasons.append("q99_too_high")
        if int(row["meets_dist_penalty"]) == 0:
            reasons.append("dist_penalty")
        if int(row["meets_tail_over"]) == 0:
            reasons.append("tail_over")
        if int(row["meets_not_identity"]) == 0:
            reasons.append("identity")
        if int(row["meets_not_duplicate"]) == 0:
            reasons.append("duplicate")
        return ",".join(reasons)

    d["guardrail_fail_reasons"] = d.apply(_fail_reasons, axis=1)
    return d


def compute_selection_score_v241(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()

    if "top1pct_baseline" not in d.columns or d["top1pct_baseline"].isna().all():
        if (d["candidate_id"].astype(str) == "baseline_identity").any():
            b = d[d["candidate_id"].astype(str) == "baseline_identity"].iloc[0]
        else:
            b = d.sort_values("rmse_prime", na_position="last").iloc[0]
        d["top1pct_baseline"] = float(pd.to_numeric(pd.Series([b.get("rmse_prime_top1pct")]), errors="coerce").iloc[0])

    rmse_prime = _coerce_num(d, "rmse_prime", fill=1e9)
    rmse_top1 = _coerce_num(d, "rmse_prime_top1pct", fill=np.inf)
    top1_base = _coerce_num(d, "top1pct_baseline", fill=np.nan)
    split_std = _coerce_num(d, "rmse_split_std", fill=2.0)
    dist_pen = _coerce_num(d, "distribution_alignment_penalty", fill=10.0)
    q99 = _coerce_num(d, "q99_ratio_pos", fill=0.0)

    top1_term = np.maximum(0.0, rmse_top1 - top1_base) / 100.0
    q99_term = np.abs(q99 - 0.575)

    d["selection_score_v241"] = (
        rmse_prime
        + 0.25 * top1_term
        + 0.25 * split_std
        + 0.20 * dist_pen
        + 3.0 * q99_term
    )
    return d


def select_robust_and_challenger_v241(df_guarded: pd.DataFrame) -> pd.DataFrame:
    if df_guarded is None or df_guarded.empty:
        return pd.DataFrame()
    d = df_guarded.copy()

    if (d["candidate_id"].astype(str) == "baseline_identity").any():
        baseline = d[d["candidate_id"].astype(str) == "baseline_identity"].iloc[0]
    else:
        baseline = d.sort_values("rmse_prime", na_position="last").iloc[0]
    rmse_baseline = float(pd.to_numeric(pd.Series([baseline.get("rmse_prime")]), errors="coerce").iloc[0])

    sort_cols = ["selection_score_v241", "rmse_prime"]
    for c in sort_cols:
        if c not in d.columns:
            d[c] = np.nan

    robust_pool = d[d.get("hard_admissible", 0).astype(int) == 1].copy()
    if robust_pool.empty:
        robust_row = baseline.to_dict()
        robust_row["selection_reason"] = "fallback_baseline_no_admissible_tail_candidate"
    else:
        robust_row = robust_pool.sort_values(sort_cols, na_position="last").iloc[0].to_dict()
        robust_row["selection_reason"] = "best_admissible_by_selection_score_v241"
    robust_row["role"] = "robust"
    robust_row["selection_status"] = "selected_robust"
    robust_row["risk_tag"] = "robust"

    challenger_pool = d.copy()
    challenger_pool = challenger_pool[
        (_coerce_num(challenger_pool, "is_identity_candidate", fill=0.0) == 0.0)
        & (_coerce_num(challenger_pool, "is_duplicate_candidate", fill=0.0) == 0.0)
        & (_coerce_num(challenger_pool, "rmse_prime", fill=np.inf) <= (rmse_baseline + 1.0))
        & (_coerce_num(challenger_pool, "q99_ratio_pos", fill=0.0) >= 0.45)
    ].copy()
    if "tail_overcorrection_flag" in challenger_pool.columns:
        challenger_pool = challenger_pool[_coerce_num(challenger_pool, "tail_overcorrection_flag", fill=0.0) == 0.0]

    challenger_row = None
    if not challenger_pool.empty:
        challenger_pool["tail_strength_rank"] = -_coerce_num(challenger_pool, "q99_ratio_pos", fill=0.0)
        challenger_pool = challenger_pool.sort_values(
            ["tail_strength_rank", "rmse_prime", "selection_score_v241"],
            na_position="last",
        )
        for _, r in challenger_pool.iterrows():
            if str(r["candidate_id"]) != str(robust_row["candidate_id"]):
                challenger_row = r.to_dict()
                break

    rows = [robust_row]
    if challenger_row is not None:
        challenger_row["role"] = "lb_challenger"
        challenger_row["selection_status"] = "selected_challenger"
        challenger_row["risk_tag"] = "public_private_risk"
        challenger_row["selection_reason"] = "best_tail_candidate_under_soft_rmse_constraint"
        rows.append(challenger_row)
    else:
        fallback = robust_row.copy()
        fallback["role"] = "lb_challenger"
        fallback["selection_status"] = "selected_challenger"
        fallback["risk_tag"] = "fallback_same_as_robust"
        fallback["selection_reason"] = "no_non_identity_challenger_found"
        rows.append(fallback)

    return pd.DataFrame(rows).reset_index(drop=True)


def _run_one_phase_b_variant(
    ctx: Mapping[str, Any],
    *,
    base_run_row: Mapping[str, Any],
    feature_set: str,
    beta: float,
    w_max: float,
    threshold_q: float,
    seed: int,
) -> pd.DataFrame:
    # Existing V2 API does not expose beta/w_max/q90 knobs directly.
    # We keep these values as metadata while running weighted_tail severity mode.
    cfg_lookup = w22.build_cfg_lookup()
    row = dict(base_run_row)
    row["severity_mode"] = "weighted_tail"
    row["feature_set"] = feature_set
    row["calibration"] = str(row.get("calibration", "none"))
    row["tail_mapper"] = str(row.get("tail_mapper", "none"))
    spec = w22.build_spec_from_row(
        row,
        cfg_lookup=cfg_lookup,
        te_cols=tr.DEFAULT_TE_COLS,
        feature_set_override=feature_set,
    )

    train_raw, test_raw = v2.load_train_test(Path(ctx.get("root", ".")) / "data")
    feature_sets = v2.prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
    splits = v2.build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")

    _, run_df, pred_df = v2.run_benchmark(spec=spec, bundle=feature_sets, splits=splits, seed=seed)
    dist_df = v2.build_prediction_distribution_table(pred_df) if not pred_df.empty else pd.DataFrame()
    scored = (
        w22.score_quick_runs(run_df, pred_df, dist_df, seed=seed, n_sim_shakeup=0, run_shakeup=False)
        if not run_df.empty
        else pd.DataFrame()
    )
    if scored.empty:
        return pd.DataFrame()

    r = scored.iloc[0].to_dict()
    candidate_id = f"phaseb_wtail_b{str(beta).replace('.', '')}_w{int(w_max)}_q{str(threshold_q).replace('.', '')}_{feature_set}"
    out = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "candidate_family": "sev_retrain_weighted_tail",
                "split": "multi",
                "base_run_id": str(base_run_row.get("run_id")),
                "engine": str(base_run_row.get("engine")),
                "family": str(base_run_row.get("family")),
                "config_id": str(base_run_row.get("config_id")),
                "feature_set": feature_set,
                "severity_mode": "weighted_tail",
                "calibration": str(base_run_row.get("calibration", "none")),
                "tail_mapper": str(base_run_row.get("tail_mapper", "none")),
                "seed": int(seed),
                "tweedie_power": float(base_run_row.get("tweedie_power", 1.5)),
                "phase_b_beta": float(beta),
                "phase_b_w_max": float(w_max),
                "phase_b_threshold_q": float(threshold_q),
                "rmse_prime": float(r.get("rmse_primary_time", np.nan)),
                "rmse_gap_secondary": float(r.get("rmse_secondary_group", np.nan)) - float(r.get("rmse_primary_time", np.nan)),
                "rmse_gap_aux": float(r.get("rmse_aux_blocked5", np.nan)) - float(r.get("rmse_primary_time", np.nan)),
                "rmse_split_std": float(r.get("rmse_split_std", np.nan)),
                "rmse_prime_top1pct": float(r.get("rmse_prime_top1pct", np.nan)),
                "q95_ratio_pos": float(r.get("q95_ratio_pos", np.nan)),
                "q99_ratio_pos": float(r.get("q99_ratio_pos", np.nan)),
                "distribution_alignment_penalty": float(r.get("distribution_alignment_penalty", np.nan)),
                "tail_overcorrection_flag": int(float(r.get("q99_ratio_pos", np.nan)) > 0.85) if np.isfinite(float(r.get("q99_ratio_pos", np.nan))) else 0,
                "tail_undercoverage_flag": int(float(r.get("q99_ratio_pos", np.nan)) < 0.10) if np.isfinite(float(r.get("q99_ratio_pos", np.nan))) else 1,
                "is_identity_candidate": 0,
                "is_duplicate_candidate": 0,
                "phase_b_generated_from_benchmark": 1,
            }
        ]
    )
    return out


def run_phase_b_if_needed(
    ctx: Mapping[str, Any],
    base_run_row: Mapping[str, Any],
    trigger_no_candidate: bool = True,
) -> pd.DataFrame:
    guarded = pd.DataFrame(ctx.get("guarded_candidates_v241", pd.DataFrame()))
    if trigger_no_candidate and not guarded.empty:
        if (guarded.get("hard_admissible", pd.Series(dtype=float)).fillna(0).astype(int) == 1).any():
            return pd.DataFrame()

    # Two variants max, as requested.
    variants = [
        {"beta": 0.5, "w_max": 10.0, "threshold_q": 0.90, "feature_set": "base_v2"},
        {"beta": 1.0, "w_max": 10.0, "threshold_q": 0.90, "feature_set": "robust_v2"},
    ]

    rows = []
    for vcfg in variants:
        try:
            rows.append(
                _run_one_phase_b_variant(
                    ctx,
                    base_run_row=base_run_row,
                    feature_set=str(vcfg["feature_set"]),
                    beta=float(vcfg["beta"]),
                    w_max=float(vcfg["w_max"]),
                    threshold_q=float(vcfg["threshold_q"]),
                    seed=int(base_run_row.get("seed", 42)),
                )
            )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def materialize_submissions_v241(
    ctx: Mapping[str, Any],
    selected_df: pd.DataFrame,
    transform_store: Mapping[str, Any],
    *,
    out_dir: str | Path = ARTIFACT_V241_DIR,
    base_run_row: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    out = v2.ensure_dir(out_dir)
    if selected_df is None or selected_df.empty:
        return {"submission_paths": {}, "pred_audits": pd.DataFrame(), "generated_rows": pd.DataFrame()}

    if base_run_row is None:
        base = ctx.get("base_v24", {})
        base_run_row = dict(base.get("base_run_row", {}))
    if not base_run_row:
        raise ValueError("base_run_row is required to materialize submissions.")

    base_payload = None
    submission_paths: Dict[str, Path] = {}
    pred_audits: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []

    for _, row in selected_df.iterrows():
        role = str(row.get("role", "candidate"))
        cid = str(row.get("candidate_id"))
        cfamily = str(row.get("candidate_family", ""))

        if cfamily == "sev_retrain_weighted_tail":
            payload = tr._fit_base_fulltrain_components(ctx, base_run_row=row.to_dict())
            test_index = payload["test_raw"]["index"]
            pred = np.maximum(np.asarray(payload["test_prime"], dtype=float), 0.0)
        else:
            if base_payload is None:
                base_payload = tr._fit_base_fulltrain_components(ctx, base_run_row=base_run_row)
            test_index = base_payload["test_raw"]["index"]
            pred = tr._apply_candidate_transform_to_test_components(
                base_payload["test_freq"],
                base_payload["test_sev"],
                candidate_row=row.to_dict(),
                transform_store=transform_store,
            )

        if role == "robust":
            fname = "submission_v2_4_1_robust.csv"
        elif role == "lb_challenger":
            fname = "submission_v2_4_1_lb_challenger.csv"
        else:
            fname = f"submission_v2_4_1_{role}.csv"

        path = out / fname
        sub = v2.build_submission(pd.Series(test_index), np.maximum(np.asarray(pred, dtype=float), 0.0))
        sub.to_csv(path, index=False)
        submission_paths[role] = path

        pred_audits.append(
            {
                "role": role,
                "candidate_id": cid,
                **v2.compute_prediction_distribution_audit(pred, run_id=cid, split="test", sample="test"),
            }
        )
        generated_rows.append(
            {
                "role": role,
                "candidate_id": cid,
                "candidate_family": cfamily,
                "file": str(path),
                "n": int(len(pred)),
                "pred_mean": float(np.mean(pred)),
                "pred_q99": float(np.quantile(pred, 0.99)),
            }
        )

    pred_audits_df = pd.DataFrame(pred_audits)
    generated_rows_df = pd.DataFrame(generated_rows)
    if not pred_audits_df.empty:
        pred_audits_df.to_csv(out / "pred_distribution_compare_v2_4_1.csv", index=False)
    if not generated_rows_df.empty:
        generated_rows_df.to_csv(out / "generated_submissions_v2_4_1.csv", index=False)

    return {
        "submission_paths": submission_paths,
        "pred_audits": pred_audits_df,
        "generated_rows": generated_rows_df,
    }


def write_decision_report_v241(
    *,
    base_info: Mapping[str, Any],
    candidates_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    out_dir: str | Path = ARTIFACT_V241_DIR,
) -> Path:
    out = v2.ensure_dir(out_dir)
    lines: list[str] = []
    lines.append("# Submission decision V2.4.1 Tail Selection Fix")
    lines.append("")
    lines.append("## 1) Context")
    lines.append(f"- Base run V2 anchor: `{base_info.get('base_run_id', '')}`")
    lines.append("- Goal: recover tail without uncontrolled global RMSE regression.")
    lines.append("")
    lines.append("## 2) Hard guardrails")
    lines.append("- rmse_prime <= baseline + 0.15")
    lines.append("- q99_ratio_pos in [0.45, 0.70]")
    lines.append("- secondary/aux gaps <= 1.0")
    lines.append("- distribution_alignment_penalty <= 3.0")
    lines.append("- no identity / duplicate candidates")
    lines.append("")
    lines.append("## 3) Candidate summary")
    if candidates_df is not None and not candidates_df.empty:
        cols = [
            c
            for c in [
                "candidate_id",
                "candidate_family",
                "rmse_prime",
                "rmse_delta_vs_baseline",
                "q99_ratio_pos",
                "rmse_prime_top1pct",
                "hard_admissible",
                "guardrail_fail_reasons",
                "selection_score_v241",
            ]
            if c in candidates_df.columns
        ]
        if cols:
            lines.append("")
            lines.append(candidates_df[cols].sort_values(["hard_admissible", "selection_score_v241"], ascending=[False, True], na_position="last").to_markdown(index=False))
            lines.append("")
    else:
        lines.append("- No candidates available.")
        lines.append("")
    lines.append("## 4) Pareto front")
    if pareto_df is not None and not pareto_df.empty:
        cols = [c for c in ["candidate_id", "candidate_family", "rmse_prime", "q99_ratio_pos", "rmse_split_std", "rmse_prime_top1pct"] if c in pareto_df.columns]
        lines.append("")
        lines.append(pareto_df[cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- Pareto front unavailable.")
        lines.append("")
    lines.append("## 5) Final selection")
    if selected_df is not None and not selected_df.empty:
        cols = [c for c in ["role", "candidate_id", "candidate_family", "risk_tag", "selection_reason", "rmse_prime", "q99_ratio_pos", "selection_score_v241"] if c in selected_df.columns]
        lines.append("")
        lines.append(selected_df[cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- No final selection rows.")
        lines.append("")
    lines.append("## 6) Method note")
    lines.append("- Overfitting is not asserted without inter-split evidence.")
    lines.append("- Public LB gains from global scaling can be unstable on private split.")
    lines.append("- Robust selection remains anchored on strict local guardrails.")
    lines.append("")

    path = out / "submission_decision_v2_4_1.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_v241_cycle(
    root: str | Path = ".",
    *,
    run_phase_b_if_missing: bool = True,
    materialize_submissions: bool = True,
) -> dict[str, Any]:
    ctx = load_v24_outputs(root)
    out_dir = Path(ctx["artifact_v241"])
    base = ctx["base_v24"]

    candidates = ctx.get("tail_candidates_multi_v24", pd.DataFrame()).copy()
    transform_store = ctx.get("tail_transform_store_v24", {})
    marked = mark_identity_and_duplicate_candidates(candidates, transform_store, tol=1e-10)
    guarded = compute_tail_guardrails(marked, baseline_id="baseline_identity", rmse_tol=0.15, q99_low=0.45, q99_high=0.70)
    scored = compute_selection_score_v241(guarded)
    scored["phase_source"] = "phase_a"

    ctx["guarded_candidates_v241"] = scored
    phase_b_df = pd.DataFrame()
    if run_phase_b_if_missing:
        phase_b_df = run_phase_b_if_needed(ctx, base.get("base_run_row", {}), trigger_no_candidate=True)
        if not phase_b_df.empty:
            phase_b_df = compute_tail_guardrails(
                phase_b_df,
                baseline_id=str(scored.loc[scored["candidate_id"].astype(str) == "baseline_identity", "candidate_id"].iloc[0])
                if (scored["candidate_id"].astype(str) == "baseline_identity").any()
                else str(scored.sort_values("rmse_prime", na_position="last").iloc[0]["candidate_id"]),
                rmse_tol=0.15,
                q99_low=0.45,
                q99_high=0.70,
            )
            phase_b_df = compute_selection_score_v241(phase_b_df)
            phase_b_df["phase_source"] = "phase_b"

    all_candidates = pd.concat([scored, phase_b_df], ignore_index=True, sort=False) if not phase_b_df.empty else scored.copy()
    all_candidates = all_candidates.drop_duplicates(subset=["candidate_id"], keep="last")
    selected = select_robust_and_challenger_v241(all_candidates)

    role_map = selected.set_index("candidate_id")["role"].to_dict() if not selected.empty else {}
    all_candidates["selection_status"] = all_candidates["candidate_id"].map(role_map).fillna("rejected")
    pareto = tr.build_tail_pareto_front(all_candidates)

    # Save table artifacts
    tr._drop_array_cols_for_csv(all_candidates).to_csv(out_dir / "tail_candidates_registry_v2_4_1.csv", index=False)
    tr._drop_array_cols_for_csv(pareto).to_csv(out_dir / "tail_pareto_front_v2_4_1.csv", index=False)
    tr._drop_array_cols_for_csv(all_candidates).to_csv(out_dir / "tail_selection_report_v2_4_1.csv", index=False)

    outputs = {"submission_paths": {}, "pred_audits": pd.DataFrame(), "generated_rows": pd.DataFrame()}
    if materialize_submissions:
        outputs = materialize_submissions_v241(
            ctx,
            selected_df=selected,
            transform_store=transform_store,
            out_dir=out_dir,
            base_run_row=base.get("base_run_row", {}),
        )

    decision_path = write_decision_report_v241(
        base_info=base,
        candidates_df=all_candidates,
        pareto_df=pareto,
        selected_df=selected,
        out_dir=out_dir,
    )

    return {
        "ctx": ctx,
        "base": base,
        "candidates": all_candidates,
        "pareto": pareto,
        "selected": selected,
        "phase_b_candidates": phase_b_df,
        "decision_path": decision_path,
        **outputs,
    }


__all__ = [
    "load_v24_outputs",
    "mark_identity_and_duplicate_candidates",
    "compute_tail_guardrails",
    "compute_selection_score_v241",
    "select_robust_and_challenger_v241",
    "run_phase_b_if_needed",
    "materialize_submissions_v241",
    "write_decision_report_v241",
    "run_v241_cycle",
]


def train_run(config_path: str) -> dict:
    from src.insurance_pricing import train_run as _train_run

    return _train_run(config_path)

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.insurance_pricing import analytics as ds
from src.insurance_pricing import training as v2
from .common import (
    safe_read_csv as _safe_read_csv,
    safe_read_json as _safe_read_json,
    safe_read_parquet as _safe_read_parquet,
)


ARTIFACT_QUICK_DIR = Path("artifacts") / "v2_2_quick"


def ensure_quick_dir(root: str | Path = ".") -> Path:
    return v2.ensure_dir(Path(root) / ARTIFACT_QUICK_DIR)


def load_existing_artifacts(root: str | Path = ".") -> Dict[str, Any]:
    root = Path(root)
    a_v1 = root / "artifacts"
    a_v2 = root / v2.DEFAULT_V2_DIR
    a_ds = root / ds.DEFAULT_DS_DIR
    out_dir = ensure_quick_dir(root)

    ctx: Dict[str, Any] = {
        "root": root,
        "data_dir": root / "data",
        "artifact_v1": a_v1,
        "artifact_v2": a_v2,
        "artifact_ds": a_ds,
        "artifact_quick": out_dir,
    }

    # V1
    ctx["v1_run_registry"] = _safe_read_csv(a_v1 / "run_registry.csv")
    ctx["v1_oof"] = _safe_read_parquet(a_v1 / "oof_predictions.parquet")
    ctx["v1_submission"] = _safe_read_csv(a_v1 / "submission_v1.csv")

    # V2
    ctx["v2_run_registry"] = _safe_read_csv(a_v2 / "run_registry_v2.csv")
    ctx["v2_oof"] = _safe_read_parquet(a_v2 / "oof_predictions_v2.parquet")
    ctx["v2_selection_report"] = _safe_read_csv(a_v2 / "selection_report_v2.csv")
    ctx["v2_selected"] = _safe_read_csv(a_v2 / "selected_models_v2.csv")
    ctx["v2_pred_dist"] = _safe_read_csv(a_v2 / "pred_distribution_audit_v2.csv")
    ctx["v2_submission_robust"] = _safe_read_csv(a_v2 / "submission_v2_robust.csv")
    ctx["v2_submission_single"] = _safe_read_csv(a_v2 / "submission_v2_single.csv")
    ctx["v2_submission_audit"] = _safe_read_json(a_v2 / "submission_audit_v2.json")

    # DS
    ctx["ds_metrics"] = _safe_read_csv(a_ds / "oof_model_diagnostics_metrics.csv")
    ctx["ds_dist"] = _safe_read_csv(a_ds / "oof_model_diagnostics_distribution.csv")
    ctx["ds_extremes"] = _safe_read_csv(a_ds / "oof_model_diagnostics_extreme_cases_summary.csv")
    ctx["ds_err_true_za"] = _safe_read_csv(a_ds / "oof_model_diagnostics_error_by_decile_true_zero_aware.csv")
    ctx["ds_drift_num"] = _safe_read_csv(a_ds / "drift_numeric_ks_psi.csv")
    ctx["ds_drift_cat"] = _safe_read_csv(a_ds / "drift_categorical_chi2.csv")
    ctx["ds_seg_resid"] = _safe_read_csv(a_ds / "oof_model_diagnostics_residuals_segment_summary.csv")
    return ctx


def summarize_submission_df(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name, "exists": int(df is not None and len(df) > 0)}
    if df is None or len(df) == 0 or "pred" not in df.columns:
        return out
    s = pd.to_numeric(df["pred"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out.update(
        {
            "n": int(len(df)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "q90": float(s.quantile(0.90)),
            "q95": float(s.quantile(0.95)),
            "q99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "share_zero": float((s <= 0).mean()),
        }
    )
    return out


def build_cfg_lookup() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        engine: {cfg["config_id"]: cfg for cfg in cfgs}
        for engine, cfgs in v2.V2_COARSE_CONFIGS.items()
    }


def build_spec_from_row(
    row: Mapping[str, Any],
    *,
    cfg_lookup: Mapping[str, Mapping[str, Mapping[str, Any]]],
    te_cols: Sequence[str],
    feature_set_override: Optional[str] = None,
) -> Dict[str, Any]:
    engine = str(row["engine"])
    config_id = str(row["config_id"])
    cfg = cfg_lookup.get(engine, {}).get(config_id)
    if cfg is None:
        raise KeyError(f"Config not found in V2_COARSE_CONFIGS: {engine}/{config_id}")

    family = str(row.get("family", "two_part_classic"))
    severity_mode = str(row.get("severity_mode", "classic"))
    calibration = str(row.get("calibration", "none"))
    tail_mapper = str(row.get("tail_mapper", "none"))
    feature_set = str(feature_set_override or row.get("feature_set", "base_v2"))

    tw = row.get("tweedie_power", 1.5)
    tweedie_power = 1.5 if pd.isna(tw) else float(tw)

    return {
        "feature_set": feature_set,
        "engine": engine,
        "family": family,
        "severity_mode": severity_mode,
        "tweedie_power": tweedie_power,
        "config_id": config_id,
        "calibration_methods": [calibration],
        "use_tail_mapper": bool(tail_mapper != "none" and family != "direct_tweedie"),
        "use_target_encoding": True,
        "target_encode_cols": list(te_cols),
        "target_encoding_smoothing": 20.0,
        "freq_params": cfg["freq_params"],
        "sev_params": cfg["sev_params"],
        "direct_params": cfg["direct_params"],
    }


def extract_row_from_run_table(run_df: pd.DataFrame, run_id: str) -> Optional[Dict[str, Any]]:
    if len(run_df) == 0:
        return None
    d = run_df.copy()
    if "run_id" not in d.columns:
        d["run_id"] = v2.make_run_id(d)
    if "level" in d.columns:
        d = d[d["level"].astype(str) == "run"]
    d = d[d["run_id"].astype(str) == str(run_id)]
    if len(d) == 0:
        return None
    p = d[d["split"].astype(str) == "primary_time"] if "split" in d.columns else d
    return (p.iloc[0] if len(p) else d.iloc[0]).to_dict()


def select_quick_candidates(
    selection_report_df: pd.DataFrame,
    run_registry_v2_df: pd.DataFrame,
    *,
    max_runs: int = 4,
) -> pd.DataFrame:
    if len(selection_report_df):
        pool = selection_report_df.copy()
        if "accepted" in pool.columns:
            acc = pool[pool["accepted"].astype(str).str.lower().isin(["true", "1"])]
            if len(acc):
                pool = acc
        sort_cols = [c for c in ["rank", "selection_score", "rmse_primary_time"] if c in pool.columns]
        pool = pool.sort_values(sort_cols if sort_cols else pool.columns.tolist()[:1])
        source = "selection_report_v2"
    else:
        pool = run_registry_v2_df.copy()
        if len(pool):
            if "level" in pool.columns:
                pool = pool[pool["level"].astype(str) == "run"]
            if "split" in pool.columns:
                pool = pool[pool["split"].astype(str) == "primary_time"]
            if "run_id" not in pool.columns:
                pool["run_id"] = v2.make_run_id(pool)
            sort_cols = [c for c in ["rmse_prime"] if c in pool.columns]
            pool = pool.sort_values(sort_cols if sort_cols else pool.columns.tolist()[:1])
        source = "run_registry_v2"

    if len(pool) == 0:
        return pd.DataFrame()
    if "run_id" not in pool.columns:
        pool["run_id"] = v2.make_run_id(pool)
    pool = pool.drop_duplicates(subset=["run_id"]).copy()

    chosen: list[pd.Series] = []
    chosen_ids: set[str] = set()

    for _, r in pool.head(min(2, max_runs)).iterrows():
        rid = str(r["run_id"])
        if rid not in chosen_ids:
            chosen.append(r)
            chosen_ids.add(rid)

    if "engine" in pool.columns and len(chosen) < max_runs:
        for _, grp in pool.groupby("engine", sort=False):
            for _, r in grp.iterrows():
                rid = str(r["run_id"])
                if rid in chosen_ids:
                    continue
                chosen.append(r)
                chosen_ids.add(rid)
                break
            if len(chosen) >= max_runs:
                break

    if len(chosen) < max_runs:
        for _, r in pool.iterrows():
            rid = str(r["run_id"])
            if rid in chosen_ids:
                continue
            chosen.append(r)
            chosen_ids.add(rid)
            if len(chosen) >= max_runs:
                break

    out = pd.DataFrame(chosen).reset_index(drop=True)
    out["candidate_rank_quick"] = np.arange(1, len(out) + 1)
    out["candidate_source"] = source
    return out


def compute_distribution_alignment_from_dist(
    dist_df: pd.DataFrame,
    run_id: str,
    *,
    split: str = "primary_time",
) -> Dict[str, Any]:
    base = {
        "pred_q90_oof": np.nan,
        "pred_q99_oof": np.nan,
        "pred_q90_test": np.nan,
        "pred_q99_test": np.nan,
        "q99_test_over_oof": np.nan,
        "q90_test_over_oof": np.nan,
        "std_test_over_oof": np.nan,
        "distribution_alignment_penalty": np.nan,
        "distribution_alignment_score": np.nan,
    }
    if len(dist_df) == 0:
        return base
    d = dist_df.copy()
    row_oof = d[
        (d["run_id"].astype(str) == str(run_id))
        & (d["split"].astype(str) == str(split))
        & (d["sample"].astype(str) == "oof")
    ]
    row_test = d[
        (d["run_id"].astype(str) == str(run_id))
        & (d["split"].astype(str) == str(split))
        & (d["sample"].astype(str) == "test")
    ]
    if len(row_oof) == 0 or len(row_test) == 0:
        return base
    o = row_oof.iloc[0]
    t = row_test.iloc[0]
    q90_ratio = float(t["pred_q90"] / max(o["pred_q90"], 1e-9))
    q99_ratio = float(t["pred_q99"] / max(o["pred_q99"], 1e-9))
    std_ratio = float(t["pred_std"] / max(o["pred_std"], 1e-9)) if "pred_std" in d.columns else np.nan
    penalty = 10.0 * abs(q99_ratio - 1.0) + 5.0 * abs(q90_ratio - 1.0)
    if np.isfinite(std_ratio):
        penalty += 2.0 * abs(std_ratio - 1.0)
    return {
        "pred_q90_oof": float(o["pred_q90"]),
        "pred_q99_oof": float(o["pred_q99"]),
        "pred_q90_test": float(t["pred_q90"]),
        "pred_q99_test": float(t["pred_q99"]),
        "q99_test_over_oof": q99_ratio,
        "q90_test_over_oof": q90_ratio,
        "std_test_over_oof": std_ratio,
        "distribution_alignment_penalty": float(penalty),
        "distribution_alignment_score": float(-penalty),
    }


def classify_gap_hypothesis(row: Mapping[str, Any]) -> str:
    tail_flag = bool(row.get("tail_undercoverage_flag", 0))
    ood_flag = bool(row.get("ood_risk_flag", 0))
    cv_flag = bool(row.get("cv_instability_flag", 0))
    if cv_flag and not (tail_flag or ood_flag):
        return "Overfitting CV probable"
    if tail_flag and ood_flag:
        return "Mixte (OOD + queue)"
    if tail_flag:
        return "Gap domine par sous-modelisation de la queue"
    if ood_flag:
        return "Gap domine par OOD / shift"
    return "Overfitting non prouve"


def build_bridge_summary(ctx: Mapping[str, Any], *, kaggle_public_rmse: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    v1_run = ctx.get("v1_run_registry", pd.DataFrame()).copy()
    if len(v1_run):
        if "level" in v1_run.columns:
            v1_run = v1_run[v1_run["level"].astype(str) == "run"]
        if "split" in v1_run.columns:
            v1_run = v1_run[v1_run["split"].astype(str) == "primary_time"]
        if len(v1_run):
            best = v1_run.sort_values("rmse_prime").iloc[0]
            rows.append(
                {
                    "model_version": "v1_best_local",
                    "run_id": "|".join(
                        map(
                            str,
                            [
                                best.get("engine"),
                                best.get("config_id"),
                                best.get("seed"),
                                best.get("severity_mode"),
                                best.get("calibration"),
                            ],
                        )
                    ),
                    "rmse_primary_time": float(best.get("rmse_prime", np.nan)),
                    "auc_freq": float(best.get("auc_freq", np.nan)),
                    "brier_freq": float(best.get("brier_freq", np.nan)),
                    "rmse_sev_pos": float(best.get("rmse_sev_pos", np.nan)),
                    "q99_ratio_pos": float(best.get("q99_ratio_pos", np.nan)),
                }
            )

    v2_selected = ctx.get("v2_selected", pd.DataFrame())
    v2_run_df = ctx.get("v2_run_registry", pd.DataFrame())
    if len(v2_selected) and "run_id" in v2_selected.columns:
        rid = str(v2_selected.iloc[0]["run_id"])
        row = extract_row_from_run_table(v2_run_df, rid)
        if row is not None:
            rows.append(
                {
                    "model_version": "v2_selected_top",
                    "run_id": rid,
                    "rmse_primary_time": float(row.get("rmse_prime", row.get("rmse_primary_time", np.nan))),
                    "auc_freq": float(row.get("auc_freq", np.nan)),
                    "brier_freq": float(row.get("brier_freq", np.nan)),
                    "rmse_sev_pos": float(row.get("rmse_sev_pos", np.nan)),
                    "q99_ratio_pos": float(row.get("q99_ratio_pos", np.nan)),
                }
            )

    ds_metrics = ctx.get("ds_metrics", pd.DataFrame())
    if len(ds_metrics):
        m = ds_metrics.iloc[0].to_dict()
        rows.append(
            {
                "model_version": "ds_diagnostic_run",
                "run_id": str(m.get("run_id")),
                "rmse_primary_time": float(m.get("rmse_prime", np.nan)),
                "auc_freq": float(m.get("auc_freq", np.nan)),
                "brier_freq": float(m.get("brier_freq", np.nan)),
                "rmse_sev_pos": float(m.get("rmse_sev_pos", np.nan)),
                "q99_ratio_pos": float(m.get("q99_ratio_pos", np.nan)),
                "rmse_prime_top1pct": float(m.get("rmse_prime_top1pct", np.nan)),
                "kaggle_public_rmse_user": float(kaggle_public_rmse),
            }
        )
    return pd.DataFrame(rows)


def build_submission_distribution_summary(ctx: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub_df = pd.DataFrame(
        [
            summarize_submission_df(ctx.get("v1_submission", pd.DataFrame()), "submission_v1"),
            summarize_submission_df(ctx.get("v2_submission_robust", pd.DataFrame()), "submission_v2_robust"),
            summarize_submission_df(ctx.get("v2_submission_single", pd.DataFrame()), "submission_v2_single"),
        ]
    )
    corr_rows: list[dict[str, Any]] = []
    sub_map = {
        "v1": ctx.get("v1_submission", pd.DataFrame()),
        "v2_robust": ctx.get("v2_submission_robust", pd.DataFrame()),
        "v2_single": ctx.get("v2_submission_single", pd.DataFrame()),
    }
    keys = list(sub_map.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = sub_map[keys[i]]
            b = sub_map[keys[j]]
            if len(a) == 0 or len(b) == 0 or "index" not in a.columns or "index" not in b.columns:
                continue
            m = a.merge(b, on="index", suffixes=("_a", "_b"))
            if len(m) == 0:
                continue
            corr_rows.append(
                {
                    "left": keys[i],
                    "right": keys[j],
                    "corr_pred": float(np.corrcoef(m["pred_a"].to_numpy(), m["pred_b"].to_numpy())[0, 1]),
                }
            )
    return sub_df, pd.DataFrame(corr_rows)


def build_gap_diagnosis(
    ctx: Mapping[str, Any],
    *,
    kaggle_public_rmse: float,
    top_k: int = 10,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    v2_selection_report_df = ctx.get("v2_selection_report", pd.DataFrame())
    v2_run_df = ctx.get("v2_run_registry", pd.DataFrame())
    v2_pred_dist_df = ctx.get("v2_pred_dist", pd.DataFrame())
    v2_oof_df = ctx.get("v2_oof", pd.DataFrame())
    ds_drift_cat_df = ctx.get("ds_drift_cat", pd.DataFrame())

    if len(v2_selection_report_df):
        base_diag = v2_selection_report_df.copy()
    else:
        base_diag = pd.DataFrame()
        if len(v2_run_df):
            rr = v2_run_df.copy()
            if "run_id" not in rr.columns:
                rr["run_id"] = v2.make_run_id(rr)
            if "level" in rr.columns:
                rr = rr[rr["level"].astype(str) == "run"]
            piv_rmse = rr.pivot_table(index="run_id", columns="split", values="rmse_prime", aggfunc="mean")
            piv_q99 = rr.pivot_table(index="run_id", columns="split", values="q99_ratio_pos", aggfunc="mean")
            meta_cols = [
                c
                for c in [
                    "feature_set",
                    "engine",
                    "family",
                    "config_id",
                    "seed",
                    "severity_mode",
                    "calibration",
                    "tail_mapper",
                    "tweedie_power",
                ]
                if c in rr.columns
            ]
            meta = rr.groupby("run_id")[meta_cols].first()
            base_diag = meta.join(piv_rmse.add_prefix("rmse_")).join(piv_q99.add_prefix("q99_")).reset_index()
            for a, b, name in [
                ("secondary_group", "primary_time", "rmse_gap_secondary"),
                ("aux_blocked5", "primary_time", "rmse_gap_aux"),
            ]:
                if f"rmse_{a}" in base_diag.columns and f"rmse_{b}" in base_diag.columns:
                    base_diag[name] = base_diag[f"rmse_{a}"] - base_diag[f"rmse_{b}"]
            if {"rmse_primary_time", "rmse_secondary_group", "rmse_aux_blocked5"}.issubset(base_diag.columns):
                base_diag["rmse_split_std"] = base_diag[
                    ["rmse_primary_time", "rmse_secondary_group", "rmse_aux_blocked5"]
                ].std(axis=1, ddof=0)

    if len(base_diag):
        if "rank" in base_diag.columns:
            gap_diag_top = base_diag.sort_values(["rank"]).head(top_k).copy()
        else:
            sort_cols = [c for c in ["rmse_primary_time", "selection_score", "rmse_weighted"] if c in base_diag.columns]
            gap_diag_top = base_diag.sort_values(sort_cols if sort_cols else base_diag.columns.tolist()[:1]).head(top_k).copy()
    else:
        gap_diag_top = pd.DataFrame()

    # distribution alignment
    if len(gap_diag_top) and len(v2_pred_dist_df):
        dist_rows = []
        for rid in gap_diag_top["run_id"].astype(str).tolist():
            m = compute_distribution_alignment_from_dist(v2_pred_dist_df, rid, split="primary_time")
            m["run_id"] = rid
            dist_rows.append(m)
        gap_diag_top = gap_diag_top.merge(pd.DataFrame(dist_rows), on="run_id", how="left")

    # DS diagnostics on the fly for shortlist
    if len(gap_diag_top) and len(v2_oof_df):
        diag_rows = []
        for rid in gap_diag_top["run_id"].astype(str).tolist():
            try:
                dd = ds.compute_oof_model_diagnostics(v2_oof_df, run_id=rid, split="primary_time", decile_mode="zero_aware")
                m = dd.get("metrics", pd.DataFrame())
                if len(m):
                    row = m.iloc[0].to_dict()
                    row["run_id"] = rid
                    diag_rows.append(row)
            except Exception as e:
                diag_rows.append({"run_id": rid, "diag_error": f"{type(e).__name__}: {e}"})
        diag_df = pd.DataFrame(diag_rows)
        if len(diag_df):
            keep = [c for c in ["run_id", "rmse_prime_top1pct", "q95_ratio_pos", "q99_ratio_pos", "auc_freq", "brier_freq"] if c in diag_df.columns]
            gap_diag_top = gap_diag_top.drop(columns=[c for c in ["q99_ratio_pos", "auc_freq", "brier_freq"] if c in gap_diag_top.columns], errors="ignore")
            gap_diag_top = gap_diag_top.merge(diag_df[keep], on="run_id", how="left")

    unseen_focus_flag = 0
    if len(ds_drift_cat_df):
        focus = ds_drift_cat_df[ds_drift_cat_df["column"].astype(str).isin(["code_postal", "modele_vehicule", "marque_vehicule"])]
        if len(focus):
            ratios = pd.to_numeric(focus.get("unseen_ratio_test", pd.Series(dtype=float)), errors="coerce")
            unseen_focus_flag = int((ratios.fillna(0.0) > 0.02).any())

    for c in ["rmse_gap_secondary", "rmse_gap_aux", "rmse_split_std", "q99_ratio_pos", "q99_test_over_oof"]:
        if c not in gap_diag_top.columns:
            gap_diag_top[c] = np.nan
    gap_diag_top["tail_undercoverage_flag"] = (pd.to_numeric(gap_diag_top["q99_ratio_pos"], errors="coerce") < 0.50).astype(int)
    gap_diag_top["distribution_mismatch_flag"] = (
        (pd.to_numeric(gap_diag_top["q99_test_over_oof"], errors="coerce") < 0.85)
        | (pd.to_numeric(gap_diag_top["q99_test_over_oof"], errors="coerce") > 1.15)
    ).fillna(False).astype(int)
    gap_diag_top["cv_instability_flag"] = (
        (pd.to_numeric(gap_diag_top["rmse_gap_secondary"], errors="coerce") > 1.0)
        | (pd.to_numeric(gap_diag_top["rmse_gap_aux"], errors="coerce") > 1.0)
        | (pd.to_numeric(gap_diag_top["rmse_split_std"], errors="coerce") > 1.2)
    ).fillna(False).astype(int)
    gap_diag_top["ood_risk_flag"] = ((unseen_focus_flag == 1) | (gap_diag_top["distribution_mismatch_flag"] == 1)).astype(int)
    gap_diag_top["split"] = "primary_time"
    if len(gap_diag_top):
        gap_diag_top["kaggle_gap_hypothesis"] = gap_diag_top.apply(classify_gap_hypothesis, axis=1)
    else:
        gap_diag_top["kaggle_gap_hypothesis"] = []

    summary = {
        "kaggle_public_rmse_user": float(kaggle_public_rmse),
        "note_metric_non_comparable": True,
        "n_runs_analyzed": int(len(gap_diag_top)),
        "global_unseen_focus_flag": int(unseen_focus_flag),
        "dominant_hypothesis_top_run": (str(gap_diag_top.iloc[0]["kaggle_gap_hypothesis"]) if len(gap_diag_top) else None),
        "dominant_hypothesis_counts": (gap_diag_top["kaggle_gap_hypothesis"].value_counts(dropna=False).to_dict() if len(gap_diag_top) else {}),
    }
    return gap_diag_top, summary


def save_gap_diagnosis_artifacts(
    report_df: pd.DataFrame,
    summary: Mapping[str, Any],
    *,
    out_dir: str | Path,
) -> None:
    out = v2.ensure_dir(out_dir)
    report_df.to_csv(out / "gap_diagnosis_report.csv", index=False)
    (out / "gap_diagnosis_summary.json").write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")


def run_quick_benchmark(
    *,
    data_dir: str | Path,
    candidates_df: pd.DataFrame,
    te_cols: Sequence[str],
    seed: int = 42,
    feature_set_override: Optional[str] = None,
    out_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "fold_df": pd.DataFrame(),
        "run_df": pd.DataFrame(),
        "pred_df": pd.DataFrame(),
        "dist_df": pd.DataFrame(),
        "errors_df": pd.DataFrame(),
        "train_raw": pd.DataFrame(),
        "test_raw": pd.DataFrame(),
        "feature_sets": {},
    }
    if len(candidates_df) == 0:
        return result

    train_raw, test_raw = v2.load_train_test(data_dir)
    feature_sets = v2.prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
    splits = v2.build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")
    cfg_lookup = build_cfg_lookup()

    all_folds: list[pd.DataFrame] = []
    all_runs: list[pd.DataFrame] = []
    all_preds: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []

    for _, cand in candidates_df.iterrows():
        rid = str(cand.get("run_id"))
        try:
            spec = build_spec_from_row(
                cand.to_dict(),
                cfg_lookup=cfg_lookup,
                te_cols=te_cols,
                feature_set_override=feature_set_override,
            )
            f, r, p = v2.run_benchmark(spec=spec, bundle=feature_sets, splits=splits, seed=int(float(cand.get("seed", seed))))
            all_folds.append(f)
            all_runs.append(r)
            all_preds.append(p)
        except Exception as e:
            errors.append({"run_id": rid, "error": f"{type(e).__name__}: {e}"})

    fold_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    run_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    dist_df = v2.build_prediction_distribution_table(pred_df) if len(pred_df) else pd.DataFrame()
    errors_df = pd.DataFrame(errors)

    result.update(
        {
            "fold_df": fold_df,
            "run_df": run_df,
            "pred_df": pred_df,
            "dist_df": dist_df,
            "errors_df": errors_df,
            "train_raw": train_raw,
            "test_raw": test_raw,
            "feature_sets": feature_sets,
        }
    )

    if out_dir is not None:
        out = v2.ensure_dir(out_dir)
        if len(fold_df):
            fold_df.to_parquet(out / "quick_fold_metrics.parquet", index=False)
        if len(pred_df):
            pred_df.to_parquet(out / "quick_oof_predictions.parquet", index=False)
        if len(dist_df):
            dist_df.to_csv(out / "pred_distribution_audit_v2_2_quick.csv", index=False)
        if len(errors_df):
            errors_df.to_csv(out / "quick_run_errors.csv", index=False)
    return result


def _make_status_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["tail_penalty"] = 20.0 * (pd.to_numeric(d["q99_ratio_pos"], errors="coerce") - 1.0).abs()
    d["rmse_gap_secondary_pos"] = np.maximum(pd.to_numeric(d["rmse_secondary_group"], errors="coerce") - pd.to_numeric(d["rmse_primary_time"], errors="coerce"), 0.0)
    d["rmse_gap_aux_pos"] = np.maximum(pd.to_numeric(d["rmse_aux_blocked5"], errors="coerce") - pd.to_numeric(d["rmse_primary_time"], errors="coerce"), 0.0)
    d["distribution_alignment_penalty"] = pd.to_numeric(d["distribution_alignment_penalty"], errors="coerce").fillna(0.0)
    d["shakeup_penalty"] = pd.to_numeric(d["shakeup_std_gap"], errors="coerce").fillna(0.0)
    d["selection_score_quick"] = (
        pd.to_numeric(d["rmse_primary_time"], errors="coerce")
        + 0.5 * d["rmse_gap_secondary_pos"]
        + 0.5 * d["rmse_gap_aux_pos"]
        + d["tail_penalty"]
        + d["distribution_alignment_penalty"]
        + d["shakeup_penalty"]
    )
    d["passes_local_guardrails"] = (
        (d["rmse_gap_secondary_pos"] <= 1.0)
        & (d["rmse_gap_aux_pos"] <= 1.0)
        & (pd.to_numeric(d.get("distribution_collapse_flag", 0.0), errors="coerce").fillna(0.0) <= 0.0)
    )
    return d


def score_quick_runs(
    run_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    *,
    seed: int = 42,
    n_sim_shakeup: int = 300,
    run_shakeup: bool = True,
) -> pd.DataFrame:
    if len(run_df) == 0:
        return pd.DataFrame()
    rr = run_df.copy()
    if "run_id" not in rr.columns:
        rr["run_id"] = v2.make_run_id(rr)
    if "level" in rr.columns:
        rr = rr[rr["level"].astype(str) == "run"]

    piv_rmse = rr.pivot_table(index="run_id", columns="split", values="rmse_prime", aggfunc="mean")
    piv_auc = rr.pivot_table(index="run_id", columns="split", values="auc_freq", aggfunc="mean") if "auc_freq" in rr.columns else pd.DataFrame()
    piv_brier = rr.pivot_table(index="run_id", columns="split", values="brier_freq", aggfunc="mean") if "brier_freq" in rr.columns else pd.DataFrame()
    piv_sev = rr.pivot_table(index="run_id", columns="split", values="rmse_sev_pos", aggfunc="mean") if "rmse_sev_pos" in rr.columns else pd.DataFrame()
    meta_cols = [c for c in ["feature_set", "engine", "family", "tweedie_power", "config_id", "seed", "severity_mode", "calibration", "tail_mapper"] if c in rr.columns]
    meta = rr.groupby("run_id")[meta_cols].first()

    reg = meta.copy()
    for split_name in ["primary_time", "secondary_group", "aux_blocked5"]:
        reg[f"rmse_{split_name}"] = piv_rmse[split_name] if split_name in piv_rmse.columns else np.nan
        if len(piv_auc):
            reg[f"auc_{split_name}"] = piv_auc[split_name] if split_name in piv_auc.columns else np.nan
        if len(piv_brier):
            reg[f"brier_{split_name}"] = piv_brier[split_name] if split_name in piv_brier.columns else np.nan
        if len(piv_sev):
            reg[f"rmse_sev_{split_name}"] = piv_sev[split_name] if split_name in piv_sev.columns else np.nan

    reg["rmse_split_std"] = reg[[c for c in ["rmse_primary_time", "rmse_secondary_group", "rmse_aux_blocked5"] if c in reg.columns]].std(axis=1, ddof=0)
    reg["rmse_gap_secondary"] = reg["rmse_secondary_group"] - reg["rmse_primary_time"]
    reg["rmse_gap_aux"] = reg["rmse_aux_blocked5"] - reg["rmse_primary_time"]

    diag_rows: list[dict[str, Any]] = []
    for rid in reg.index.astype(str).tolist():
        try:
            dd = ds.compute_oof_model_diagnostics(pred_df, run_id=rid, split="primary_time", decile_mode="zero_aware")
            m = dd.get("metrics", pd.DataFrame())
            if len(m):
                row = m.iloc[0].to_dict()
                row["run_id"] = rid
                diag_rows.append(row)
        except Exception as e:
            diag_rows.append({"run_id": rid, "diag_error": f"{type(e).__name__}: {e}"})
    diag_df = pd.DataFrame(diag_rows)
    if len(diag_df):
        reg = reg.reset_index().merge(diag_df, on="run_id", how="left").set_index("run_id")
    else:
        reg = reg.reset_index().set_index("run_id")

    dist_rows: list[dict[str, Any]] = []
    for rid in reg.index.astype(str).tolist():
        m = compute_distribution_alignment_from_dist(dist_df, rid, split="primary_time")
        m["run_id"] = rid
        dist_rows.append(m)
    if len(dist_rows):
        reg = reg.join(pd.DataFrame(dist_rows).set_index("run_id"), how="left")
    else:
        reg["distribution_alignment_penalty"] = np.nan
        reg["distribution_alignment_score"] = np.nan

    reg["distribution_collapse_flag"] = 0.0
    if len(dist_df):
        agg = (
            dist_df[dist_df["split"].astype(str) == "primary_time"]
            .groupby("run_id")["distribution_collapse_flag"]
            .max()
        )
        reg["distribution_collapse_flag"] = reg.index.to_series().map(agg).fillna(0.0).astype(float)

    reg["shakeup_std_gap"] = np.nan
    if run_shakeup:
        for rid in reg.index.astype(str).tolist():
            dprim = pred_df[
                (pred_df["is_test"] == 0)
                & (pred_df["split"].astype(str) == "primary_time")
                & (pred_df["run_id"].astype(str) == rid)
            ].copy()
            if len(dprim) == 0:
                continue
            dprim = dprim[dprim["pred_prime"].notna() & dprim["y_sev"].notna()].drop_duplicates(subset=["row_idx"])
            if len(dprim) == 0:
                continue
            sh = v2.simulate_public_private_shakeup_v2(
                dprim["y_sev"].to_numpy(dtype=float),
                dprim["pred_prime"].to_numpy(dtype=float),
                n_sim=n_sim_shakeup,
                seed=seed,
                stratified_tail=False,
            )
            reg.loc[rid, "shakeup_std_gap"] = float(sh["gap_public_minus_private"].std(ddof=0))

    reg = _make_status_columns(reg)
    reg["split"] = "multi"
    reg["rmse_prime"] = reg["rmse_primary_time"]
    if "auc_primary_time" in reg.columns:
        reg["auc_freq"] = reg["auc_primary_time"]
    if "brier_primary_time" in reg.columns:
        reg["brier_freq"] = reg["brier_primary_time"]
    if "rmse_sev_primary_time" in reg.columns:
        reg["rmse_sev_pos"] = reg["rmse_sev_primary_time"]
    return reg.reset_index()


def choose_robust_and_challenger(scored_df: pd.DataFrame) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if len(scored_df) == 0:
        return None, None
    d = scored_df.copy()
    d = d.sort_values(["selection_score_quick", "rmse_primary_time"]).reset_index(drop=True)
    robust = d.iloc[0].to_dict()
    cands = d.sort_values(["rmse_primary_time", "selection_score_quick"]).reset_index(drop=True)
    challenger = robust
    for _, r in cands.iterrows():
        if str(r["run_id"]) != str(robust["run_id"]):
            challenger = r.to_dict()
            break
    return robust, challenger


def _choose_oof_source(run_id: str, quick_pred_df: pd.DataFrame, v2_oof_df: pd.DataFrame) -> pd.DataFrame:
    if len(quick_pred_df) and (quick_pred_df["run_id"].astype(str) == str(run_id)).any():
        return quick_pred_df
    return v2_oof_df


def build_submission_with_refit(
    run_row: Mapping[str, Any],
    *,
    feature_sets: Mapping[str, v2.DatasetBundle],
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cfg_lookup: Mapping[str, Mapping[str, Mapping[str, Any]]],
    te_cols: Sequence[str],
    oof_source_df: pd.DataFrame,
    seed_default: int = 42,
) -> Dict[str, Any]:
    r = dict(run_row)
    run_id = str(r["run_id"])
    fs_name = str(r.get("feature_set", "base_v2"))
    seed = int(float(r.get("seed", seed_default)))
    family = str(r.get("family", "two_part_classic"))
    calibration = str(r.get("calibration", "none"))
    tail_mapper_name = str(r.get("tail_mapper", "none"))
    spec = build_spec_from_row(r, cfg_lookup=cfg_lookup, te_cols=te_cols)
    bundle = feature_sets[fs_name]
    out = v2.fit_full_predict_fulltrain(spec=spec, bundle=bundle, seed=seed, complexity={})
    test_freq = out["test_freq"].copy()
    test_sev = out["test_sev"].copy()

    oo = pd.DataFrame()
    if len(oof_source_df):
        oo = oof_source_df.copy()
        if "run_id" not in oo.columns:
            oo["run_id"] = v2.make_run_id(oo)
        oo = oo[
            (oo["is_test"] == 0)
            & (oo["split"].astype(str) == "primary_time")
            & (oo["run_id"].astype(str) == run_id)
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
            mapper = v2.fit_tail_mapper_safe(
                oo.loc[pos, "pred_sev"].to_numpy(),
                oo.loc[pos, "y_sev"].to_numpy(),
            )
            sev_before = test_sev.copy()
            test_sev = v2.apply_tail_mapper_safe(mapper, test_sev)
            std_ratio = float(np.std(test_sev) / max(np.std(sev_before), 1e-9))
            q99_oof = float(np.nanquantile(oo.loc[pos, "pred_sev"].to_numpy(), 0.99))
            q99_test = float(np.nanquantile(test_sev, 0.99))
            if (std_ratio < 0.70) or (q99_test < 0.60 * q99_oof):
                test_sev = sev_before

    pred = np.maximum(out["test_prime"], 0.0) if family == "direct_tweedie" else np.maximum(test_freq * test_sev, 0.0)
    sub = v2.build_submission(test_raw["index"], pred)
    audit = v2.compute_prediction_distribution_audit(
        sub["pred"].to_numpy(),
        run_id=run_id,
        split="test",
        sample="test",
    )
    return {"run_id": run_id, "sub": sub, "pred": pred, "audit": audit}


def _find_scored_row(scored_df: pd.DataFrame, run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if len(scored_df) == 0 or not run_id:
        return None
    d = scored_df[scored_df["run_id"].astype(str) == str(run_id)]
    if len(d) == 0:
        return None
    return d.iloc[0].to_dict()


def build_quick_submissions(
    ctx: Mapping[str, Any],
    *,
    quick_result: Mapping[str, Any],
    scored_df: pd.DataFrame,
    robust_row: Optional[Mapping[str, Any]],
    challenger_row: Optional[Mapping[str, Any]],
    te_cols: Sequence[str],
    seed: int = 42,
    out_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    out_dir = v2.ensure_dir(out_dir or ctx["artifact_quick"])
    train_raw = quick_result.get("train_raw", pd.DataFrame())
    test_raw = quick_result.get("test_raw", pd.DataFrame())
    feature_sets = quick_result.get("feature_sets", {})
    quick_pred_df = quick_result.get("pred_df", pd.DataFrame())
    v2_oof_df = ctx.get("v2_oof", pd.DataFrame())
    v2_selected_df = ctx.get("v2_selected", pd.DataFrame())
    v2_run_df = ctx.get("v2_run_registry", pd.DataFrame())

    baseline_v2_selected_row = None
    baseline_v2_selected_run_id = None
    if len(v2_selected_df) and "run_id" in v2_selected_df.columns:
        baseline_v2_selected_run_id = str(v2_selected_df.iloc[0]["run_id"])
        baseline_v2_selected_row = extract_row_from_run_table(v2_run_df, baseline_v2_selected_run_id)
        if baseline_v2_selected_row is not None:
            baseline_v2_selected_row["run_id"] = baseline_v2_selected_run_id

    if len(train_raw) == 0 or len(test_raw) == 0 or not feature_sets:
        train_raw, test_raw = v2.load_train_test(ctx["data_dir"])
        feature_sets = v2.prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
    cfg_lookup = build_cfg_lookup()

    robust_target = dict(robust_row) if robust_row else baseline_v2_selected_row
    challenger_target = dict(challenger_row) if challenger_row else baseline_v2_selected_row

    result: Dict[str, Any] = {
        "robust_submission": pd.DataFrame(),
        "challenger_submission": pd.DataFrame(),
        "robust_meta": {},
        "challenger_meta": {},
        "submission_dist_df": pd.DataFrame(),
        "baseline_v2_selected_row": baseline_v2_selected_row,
        "baseline_v2_selected_run_id": baseline_v2_selected_run_id,
    }

    audits: list[dict[str, Any]] = []

    for name, target in [("robust", robust_target), ("challenger", challenger_target)]:
        payload = None
        if target is not None:
            try:
                rid = str(target["run_id"])
                oof_source = _choose_oof_source(rid, quick_pred_df, v2_oof_df)
                payload = build_submission_with_refit(
                    target,
                    feature_sets=feature_sets,
                    train_raw=train_raw,
                    test_raw=test_raw,
                    cfg_lookup=cfg_lookup,
                    te_cols=te_cols,
                    oof_source_df=oof_source,
                    seed_default=seed,
                )
            except Exception:
                payload = None

        if payload is None:
            if name == "robust":
                fallback = ctx.get("v2_submission_robust", pd.DataFrame())
                src = "copy_artifacts_v2_submission_robust"
            else:
                fallback = ctx.get("v2_submission_single", pd.DataFrame())
                src = "copy_artifacts_v2_submission_single"
            if len(fallback):
                sub = fallback.copy()
                audit = v2.compute_prediction_distribution_audit(
                    sub["pred"].to_numpy(),
                    run_id=f"fallback_{name}",
                    split="test",
                    sample="test",
                )
                payload = {"run_id": baseline_v2_selected_run_id, "sub": sub, "pred": sub["pred"].to_numpy(), "audit": audit}
                meta = {"run_id": baseline_v2_selected_run_id, "source": src}
            else:
                payload = {"run_id": None, "sub": pd.DataFrame(), "pred": np.array([]), "audit": {}}
                meta = {"run_id": None, "source": "missing_fallback"}
        else:
            meta = {"run_id": payload["run_id"], "source": "fit_full_predict_fulltrain"}

        if name == "robust":
            result["robust_submission"] = payload["sub"]
            result["robust_meta"] = meta
            if len(payload["sub"]):
                payload["sub"].to_csv(out_dir / "submission_v2_2_quick_robust.csv", index=False)
        else:
            result["challenger_submission"] = payload["sub"]
            result["challenger_meta"] = meta
            if len(payload["sub"]):
                payload["sub"].to_csv(out_dir / "submission_v2_2_quick_challenger.csv", index=False)

        if payload and isinstance(payload.get("audit"), dict) and payload.get("audit"):
            audits.append({"submission_name": f"submission_v2_2_quick_{name}", **payload["audit"]})

    result["submission_dist_df"] = pd.DataFrame(audits)
    return result


def build_quick_checks(
    *,
    scored_df: pd.DataFrame,
    quick_pred_df: pd.DataFrame,
    robust_meta: Mapping[str, Any],
    challenger_meta: Mapping[str, Any],
    robust_submission: pd.DataFrame,
    challenger_submission: pd.DataFrame,
    baseline_row_v2: Optional[Mapping[str, Any]],
    seed: int = 42,
    n_sim_shakeup: int = 300,
    out_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    out_dir = v2.ensure_dir(out_dir or ARTIFACT_QUICK_DIR)
    checks_rows: list[dict[str, Any]] = []

    def _append_check(name: str, run_id: Optional[str]):
        row = _find_scored_row(scored_df, run_id)
        out: dict[str, Any] = {"submission": name, "run_id": run_id}
        if row is None:
            out["status"] = "no_registry_row"
            checks_rows.append(out)
            return
        out.update(
            {
                "rmse_primary_time": float(row.get("rmse_primary_time", np.nan)),
                "rmse_secondary_group": float(row.get("rmse_secondary_group", np.nan)),
                "rmse_aux_blocked5": float(row.get("rmse_aux_blocked5", np.nan)),
                "q99_ratio_pos": float(row.get("q99_ratio_pos", np.nan)),
                "rmse_prime_top1pct": float(row.get("rmse_prime_top1pct", np.nan)),
                "distribution_collapse_flag": float(row.get("distribution_collapse_flag", np.nan)),
            }
        )
        if baseline_row_v2 is not None:
            b_rmse = float(baseline_row_v2.get("rmse_prime", baseline_row_v2.get("rmse_primary_time", np.nan)))
            b_q99 = float(baseline_row_v2.get("q99_ratio_pos", np.nan))
            out["delta_rmse_primary_vs_baseline"] = out["rmse_primary_time"] - b_rmse if np.isfinite(b_rmse) else np.nan
            out["delta_q99_ratio_vs_baseline"] = out["q99_ratio_pos"] - b_q99 if np.isfinite(b_q99) else np.nan
            out["passes_rmse_tol"] = bool((not np.isfinite(b_rmse)) or (out["delta_rmse_primary_vs_baseline"] <= 0.5))
            out["passes_q99_tol"] = bool((not np.isfinite(b_q99)) or (out["delta_q99_ratio_vs_baseline"] >= -0.03))
        else:
            out["passes_rmse_tol"] = True
            out["passes_q99_tol"] = True

        gap_sec = out["rmse_secondary_group"] - out["rmse_primary_time"] if np.isfinite(out["rmse_secondary_group"]) else np.nan
        gap_aux = out["rmse_aux_blocked5"] - out["rmse_primary_time"] if np.isfinite(out["rmse_aux_blocked5"]) else np.nan
        out["passes_secondary_tol"] = bool((not np.isfinite(gap_sec)) or (gap_sec <= 1.0))
        out["passes_aux_tol"] = bool((not np.isfinite(gap_aux)) or (gap_aux <= 1.0))
        out["passes_collapse_flag"] = bool((not np.isfinite(out["distribution_collapse_flag"])) or (out["distribution_collapse_flag"] <= 0))
        out["local_overall_pass"] = bool(
            out["passes_rmse_tol"] and out["passes_q99_tol"] and out["passes_secondary_tol"] and out["passes_aux_tol"] and out["passes_collapse_flag"]
        )
        checks_rows.append(out)

    _append_check("robust", robust_meta.get("run_id"))
    _append_check("challenger", challenger_meta.get("run_id"))
    checks_df = pd.DataFrame(checks_rows)

    # submission distribution audits
    sub_dist_rows: list[dict[str, Any]] = []
    if len(robust_submission):
        sub_dist_rows.append(
            {
                "submission_name": "submission_v2_2_quick_robust",
                **v2.compute_prediction_distribution_audit(
                    robust_submission["pred"].to_numpy(),
                    run_id="submission_v2_2_quick_robust",
                    split="test",
                    sample="test",
                ),
            }
        )
    if len(challenger_submission):
        sub_dist_rows.append(
            {
                "submission_name": "submission_v2_2_quick_challenger",
                **v2.compute_prediction_distribution_audit(
                    challenger_submission["pred"].to_numpy(),
                    run_id="submission_v2_2_quick_challenger",
                    split="test",
                    sample="test",
                ),
            }
        )
    submission_dist_df = pd.DataFrame(sub_dist_rows)

    shakeup_robust = pd.DataFrame()
    shakeup_challenger = pd.DataFrame()

    def _run_shakeup(run_id: Optional[str], name: str) -> pd.DataFrame:
        if not run_id or len(quick_pred_df) == 0:
            return pd.DataFrame()
        d = quick_pred_df[
            (quick_pred_df["is_test"] == 0)
            & (quick_pred_df["split"].astype(str) == "primary_time")
            & (quick_pred_df["run_id"].astype(str) == str(run_id))
        ].copy()
        if len(d) == 0:
            return pd.DataFrame()
        d = d[d["pred_prime"].notna() & d["y_sev"].notna()].drop_duplicates(subset=["row_idx"])
        if len(d) == 0:
            return pd.DataFrame()
        sh = v2.simulate_public_private_shakeup_v2(
            d["y_sev"].to_numpy(dtype=float),
            d["pred_prime"].to_numpy(dtype=float),
            n_sim=n_sim_shakeup,
            seed=seed,
            stratified_tail=False,
        )
        sh.to_parquet(out_dir / f"shakeup_quick_{name}.parquet", index=False)
        return sh

    shakeup_robust = _run_shakeup(robust_meta.get("run_id"), "robust")
    shakeup_challenger = _run_shakeup(challenger_meta.get("run_id"), "challenger")

    return {
        "checks_df": checks_df,
        "submission_dist_df": submission_dist_df,
        "shakeup_robust": shakeup_robust,
        "shakeup_challenger": shakeup_challenger,
    }


def build_oof_compare_artifact(
    ctx: Mapping[str, Any],
    *,
    quick_pred_df: pd.DataFrame,
    robust_run_id: Optional[str],
    challenger_run_id: Optional[str],
    out_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    out_dir = v2.ensure_dir(out_dir or ctx["artifact_quick"])
    v1_run_df = ctx.get("v1_run_registry", pd.DataFrame())
    v1_oof_df = ctx.get("v1_oof", pd.DataFrame())
    v2_selected_df = ctx.get("v2_selected", pd.DataFrame())
    v2_oof_df = ctx.get("v2_oof", pd.DataFrame())

    def _extract_v1_best() -> pd.DataFrame:
        if len(v1_run_df) == 0 or len(v1_oof_df) == 0:
            return pd.DataFrame()
        rr = v1_run_df.copy()
        if "level" in rr.columns:
            rr = rr[rr["level"].astype(str) == "run"]
        if "split" in rr.columns:
            rr = rr[rr["split"].astype(str) == "primary_time"]
        if len(rr) == 0:
            return pd.DataFrame()
        best = rr.sort_values("rmse_prime").iloc[0]
        d = v1_oof_df[
            (v1_oof_df["is_test"] == 0)
            & (v1_oof_df["split"].astype(str) == "primary_time")
            & (v1_oof_df["engine"].astype(str) == str(best["engine"]))
            & (v1_oof_df["config_id"].astype(str) == str(best["config_id"]))
            & (v1_oof_df["seed"].astype(float).astype(int) == int(best["seed"]))
            & (v1_oof_df["severity_mode"].astype(str) == str(best["severity_mode"]))
            & (v1_oof_df["calibration"].astype(str) == str(best["calibration"]))
        ].copy()
        if len(d) == 0:
            return pd.DataFrame()
        d = d[["row_idx", "y_sev", "pred_prime"]].drop_duplicates(subset=["row_idx"], keep="last")
        return d.rename(columns={"y_sev": "y_true", "pred_prime": "pred_v1_best"})

    def _extract_v2_by_run(run_id: Optional[str], col_name: str, source_df: pd.DataFrame) -> pd.DataFrame:
        if not run_id or len(source_df) == 0:
            return pd.DataFrame()
        d = source_df[
            (source_df["is_test"] == 0)
            & (source_df["split"].astype(str) == "primary_time")
            & (source_df["run_id"].astype(str) == str(run_id))
        ].copy()
        if len(d) == 0:
            return pd.DataFrame()
        d = d[["row_idx", "y_sev", "pred_prime"]].drop_duplicates(subset=["row_idx"], keep="last")
        return d.rename(columns={"y_sev": "y_true", "pred_prime": col_name})

    frames: list[pd.DataFrame] = []
    d1 = _extract_v1_best()
    if len(d1):
        frames.append(d1)
    if len(v2_selected_df) and "run_id" in v2_selected_df.columns:
        base_run_id = str(v2_selected_df.iloc[0]["run_id"])
        d2 = _extract_v2_by_run(base_run_id, "pred_v2_selected", v2_oof_df)
        if len(d2):
            frames.append(d2)
    d3 = _extract_v2_by_run(robust_run_id, "pred_v22_quick_robust", quick_pred_df)
    if len(d3):
        frames.append(d3)
    d4 = _extract_v2_by_run(challenger_run_id, "pred_v22_quick_challenger", quick_pred_df)
    if len(d4):
        frames.append(d4)

    if not frames:
        return pd.DataFrame()
    out = frames[0].copy()
    for df in frames[1:]:
        out = out.merge(df, on=["row_idx", "y_true"], how="outer")
    out = out.sort_values("row_idx").reset_index(drop=True)
    out.to_parquet(out_dir / "oof_compare_v1_v2_v22.parquet", index=False)
    return out


def write_submission_decision_report(
    ctx: Mapping[str, Any],
    *,
    kaggle_public_rmse: float,
    gap_summary: Mapping[str, Any],
    checks_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    robust_meta: Mapping[str, Any],
    challenger_meta: Mapping[str, Any],
    baseline_v2_row: Optional[Mapping[str, Any]],
    out_dir: Optional[str | Path] = None,
) -> Path:
    out_dir = v2.ensure_dir(out_dir or ctx["artifact_quick"])
    lines: list[str] = []
    lines.append("# Submission decision V2.2 Quick")
    lines.append("")
    lines.append("## 1) Contexte")
    lines.append(f"- Kaggle public (utilisateur): ~{float(kaggle_public_rmse):.3f}")
    lines.append("- OOF local vs Kaggle public: metriques non directement comparables.")
    lines.append("")
    lines.append("## 2) Diagnostic du gap (resume)")
    lines.append(f"- Hypothese dominante (top run): {gap_summary.get('dominant_hypothesis_top_run')}")
    lines.append(f"- OOD focus flag (categories fines): {gap_summary.get('global_unseen_focus_flag')}")
    lines.append(f"- Nombre de runs analyses: {gap_summary.get('n_runs_analyzed')}")
    lines.append("")

    compare_rows: list[dict[str, Any]] = []
    v1_run = ctx.get("v1_run_registry", pd.DataFrame())
    if len(v1_run):
        rr = v1_run.copy()
        if "level" in rr.columns:
            rr = rr[rr["level"].astype(str) == "run"]
        if "split" in rr.columns:
            rr = rr[rr["split"].astype(str) == "primary_time"]
        if len(rr):
            best = rr.sort_values("rmse_prime").iloc[0]
            compare_rows.append(
                {
                    "variant": "V1_best_local_ref",
                    "run_id": "|".join(map(str, [best.get("engine"), best.get("config_id"), best.get("seed"), best.get("severity_mode"), best.get("calibration")])),
                    "rmse_primary_time": float(best.get("rmse_prime", np.nan)),
                    "q99_ratio_pos": float(best.get("q99_ratio_pos", np.nan)),
                }
            )
    if baseline_v2_row is not None:
        compare_rows.append(
            {
                "variant": "V2_selected_baseline",
                "run_id": str(baseline_v2_row.get("run_id")),
                "rmse_primary_time": float(baseline_v2_row.get("rmse_prime", baseline_v2_row.get("rmse_primary_time", np.nan))),
                "q99_ratio_pos": float(baseline_v2_row.get("q99_ratio_pos", np.nan)),
            }
        )
    for variant, meta in [("V2_2_quick_robust", robust_meta), ("V2_2_quick_challenger", challenger_meta)]:
        row = _find_scored_row(scored_df, meta.get("run_id"))
        if row:
            compare_rows.append(
                {
                    "variant": variant,
                    "run_id": str(row.get("run_id")),
                    "rmse_primary_time": float(row.get("rmse_primary_time", np.nan)),
                    "q99_ratio_pos": float(row.get("q99_ratio_pos", np.nan)),
                    "selection_score_quick": float(row.get("selection_score_quick", np.nan)),
                }
            )
    comp_df = pd.DataFrame(compare_rows)
    lines.append("## 3) Tableau comparatif (local)")
    if len(comp_df):
        lines.append("")
        lines.append(comp_df.to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- Indisponible")
        lines.append("")

    rec = "Ne pas soumettre"
    if len(checks_df):
        r = checks_df[checks_df["submission"].astype(str) == "robust"]
        c = checks_df[checks_df["submission"].astype(str) == "challenger"]
        robust_pass = bool(r.iloc[0]["local_overall_pass"]) if len(r) and "local_overall_pass" in r.columns else False
        challenger_pass = bool(c.iloc[0]["local_overall_pass"]) if len(c) and "local_overall_pass" in c.columns else False
        if robust_pass:
            rec = "Envoyer robust"
        elif challenger_pass:
            rec = "Envoyer challenger"
        elif robust_meta.get("run_id"):
            rec = "Envoyer robust (prudent)"

    lines.append("## 4) Recommandation d'envoi")
    lines.append(f"- **Decision**: {rec}")
    lines.append("")
    lines.append("## 5) Cause principale retenue (classification)")
    lines.append(f"- {gap_summary.get('dominant_hypothesis_top_run')}")
    lines.append("")
    lines.append("## 6) Notes")
    lines.append("- Cycle quick (1-2h): diagnostic + retraining cible, pas tuning exhaustif.")
    lines.append("- Les notebooks 07/08/09 restent la reference d'analyse; ce notebook fait le pont vers la decision de soumission.")
    lines.append("")

    path = out_dir / "submission_decision_v2_2_quick.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def train_run(config_path: str) -> dict:
    from src.insurance_pricing import train_run as _train_run

    return _train_run(config_path)

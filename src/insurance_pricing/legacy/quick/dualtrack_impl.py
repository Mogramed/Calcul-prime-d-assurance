from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from insurance_pricing import analytics as ds
from insurance_pricing import training as v2
from insurance_pricing._typing import BoolArray, FloatArray, SplitIndices

from . import gap_diagnosis_impl as w22
from .common import (
    rmse as _rmse,
)
from .common import (
    safe_float as _safe_float,
)
from .common import (
    safe_read_csv as _safe_read_csv,
)
from .common import (
    safe_read_json as _safe_read_json,
)
from .common import (
    safe_read_parquet as _safe_read_parquet,
)

ARTIFACT_V23_DIR = Path("artifacts") / "v2_3_dualtrack_quick"
EPS = 1e-9

DIRECT_TWEEDIE_CAT_COLS = [
    "type_contrat",
    "freq_paiement",
    "paiement",
    "utilisation",
    "code_postal",
    "conducteur2",
    "sex_conducteur1",
    "sex_conducteur2",
    "essence_vehicule",
    "marque_vehicule",
    "modele_vehicule",
    "type_vehicule",
]


def ensure_v23_dir(root: str | Path = ".") -> Path:
    return Path(v2.ensure_dir(Path(root) / ARTIFACT_V23_DIR))


def _ls_alpha(y_true: FloatArray, pred: FloatArray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    denom = float(np.dot(p, p))
    if denom < EPS:
        return 1.0
    alpha = float(np.dot(y, p) / denom)
    return alpha if np.isfinite(alpha) else 1.0


def detect_submission_schema(
    test_df: pd.DataFrame,
    *,
    root: str | Path = ".",
) -> tuple[str, str, pd.DataFrame | None]:
    root = Path(root)
    for p in [root / "sample_submission.csv", root / "data" / "sample_submission.csv"]:
        if p.exists():
            try:
                sample = pd.read_csv(p)
                id_col = str(sample.columns[0])
                target_col = str(sample.columns[1]) if sample.shape[1] > 1 else "pred"
                return id_col, target_col, sample
            except Exception:
                pass
    for cand in [v2.INDEX_COL, "id"]:
        if cand in test_df.columns:
            return cand, "pred", None
    return str(test_df.columns[0]), "pred", None


def load_existing_artifacts(root: str | Path = ".") -> dict[str, object]:
    root = Path(root)
    ctx = w22.load_existing_artifacts(root)
    out_dir = ensure_v23_dir(root)
    ctx["artifact_v23"] = out_dir

    a_v22 = root / "artifacts" / "v2_2_quick"
    ctx["v22_quick_retrain_registry"] = _safe_read_csv(a_v22 / "retrain_registry_quick.csv")
    ctx["v22_quick_gap_summary"] = _safe_read_json(a_v22 / "gap_diagnosis_summary.json")
    ctx["v22_quick_gap_report"] = _safe_read_csv(a_v22 / "gap_diagnosis_report.csv")
    ctx["v22_quick_submission_robust"] = _safe_read_csv(a_v22 / "submission_v2_2_quick_robust.csv")
    ctx["v22_quick_submission_challenger"] = _safe_read_csv(
        a_v22 / "submission_v2_2_quick_challenger.csv"
    )
    ctx["v22_quick_pred_dist"] = _safe_read_csv(a_v22 / "pred_distribution_audit_v2_2_quick.csv")
    ctx["v22_quick_oof"] = _safe_read_parquet(a_v22 / "quick_oof_predictions.parquet")
    ctx["v22_quick_decision_md"] = (
        (a_v22 / "submission_decision_v2_2_quick.md").read_text(encoding="utf-8")
        if (a_v22 / "submission_decision_v2_2_quick.md").exists()
        else ""
    )
    return ctx


def _extract_run_rows(rr: pd.DataFrame, *, split: str = "primary_time") -> pd.DataFrame:
    if rr is None or rr.empty:
        return pd.DataFrame()
    d = rr.copy()
    if "run_id" not in d.columns:
        with suppress(Exception):
            d["run_id"] = v2.make_run_id(d)
    if "level" in d.columns:
        d = d[d["level"].astype(str) == "run"]
    if "split" in d.columns and split is not None:
        d = d[d["split"].astype(str) == str(split)]
    return d


def _best_row_by_rmse(rr: pd.DataFrame, *, split: str = "primary_time") -> dict[str, Any] | None:
    d = _extract_run_rows(rr, split=split)
    if d.empty or "rmse_prime" not in d.columns:
        return None
    return dict(d.sort_values("rmse_prime").iloc[0].to_dict())


def _submission_stats(df: pd.DataFrame, name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "exists": int(df is not None and not df.empty)}
    if df is None or df.empty:
        return out
    pred_col = "pred" if "pred" in df.columns else (df.columns[1] if df.shape[1] > 1 else None)
    if pred_col is None:
        return out
    s = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    out.update(
        {
            "n": int(len(s)),
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


def build_bridge_summary(ctx: Mapping[str, object], *, kaggle_public_rmse: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    v1_best = _best_row_by_rmse(
        pd.DataFrame(ctx.get("v1_run_registry", pd.DataFrame())), split="primary_time"
    )
    if v1_best:
        rows.append(
            {
                "model_version": "v1_best_local",
                "track": "v1_two_part",
                "run_id": str(v1_best.get("run_id", "")),
                "rmse_primary_time": _safe_float(v1_best.get("rmse_prime")),
                "auc_freq": _safe_float(v1_best.get("auc_freq")),
                "brier_freq": _safe_float(v1_best.get("brier_freq")),
                "rmse_sev_pos": _safe_float(v1_best.get("rmse_sev_pos")),
                "q99_ratio_pos": _safe_float(v1_best.get("q99_ratio_pos")),
            }
        )

    v2_sel = pd.DataFrame(ctx.get("v2_selected", pd.DataFrame()))
    v2_rr = pd.DataFrame(ctx.get("v2_run_registry", pd.DataFrame()))
    if not v2_sel.empty and "run_id" in v2_sel.columns:
        rid = str(v2_sel.iloc[0]["run_id"])
        d = _extract_run_rows(v2_rr, split="primary_time")
        row = d[d["run_id"].astype(str) == rid]
        if not row.empty:
            r = row.iloc[0]
            rows.append(
                {
                    "model_version": "v2_selected_top",
                    "track": "v2_two_part",
                    "run_id": rid,
                    "rmse_primary_time": _safe_float(r.get("rmse_prime")),
                    "auc_freq": _safe_float(r.get("auc_freq")),
                    "brier_freq": _safe_float(r.get("brier_freq")),
                    "rmse_sev_pos": _safe_float(r.get("rmse_sev_pos")),
                    "q99_ratio_pos": _safe_float(r.get("q99_ratio_pos")),
                }
            )

    v22_reg = pd.DataFrame(ctx.get("v22_quick_retrain_registry", pd.DataFrame()))
    if not v22_reg.empty and "selection_status" in v22_reg.columns:
        for status, label in [
            ("selected_robust", "v2_2_quick_robust"),
            ("selected_challenger", "v2_2_quick_challenger"),
        ]:
            rr = v22_reg[v22_reg["selection_status"].astype(str) == status]
            if rr.empty:
                continue
            r = rr.sort_values(["selection_score_quick", "rmse_primary_time"]).iloc[0]
            rows.append(
                {
                    "model_version": label,
                    "track": "v2_2_quick",
                    "run_id": str(r.get("run_id")),
                    "rmse_primary_time": _safe_float(r.get("rmse_primary_time")),
                    "rmse_secondary_group": _safe_float(r.get("rmse_secondary_group")),
                    "rmse_aux_blocked5": _safe_float(r.get("rmse_aux_blocked5")),
                    "rmse_split_std": _safe_float(r.get("rmse_split_std")),
                    "q95_ratio_pos": _safe_float(r.get("q95_ratio_pos")),
                    "q99_ratio_pos": _safe_float(r.get("q99_ratio_pos")),
                    "rmse_prime_top1pct": _safe_float(r.get("rmse_prime_top1pct")),
                    "selection_score_quick": _safe_float(r.get("selection_score_quick")),
                }
            )

    ds_metrics = pd.DataFrame(ctx.get("ds_metrics", pd.DataFrame()))
    if not ds_metrics.empty:
        r = ds_metrics.iloc[0]
        rows.append(
            {
                "model_version": "ds_diagnostic_run",
                "track": "v2_two_part",
                "run_id": str(r.get("run_id")),
                "rmse_primary_time": _safe_float(r.get("rmse_prime")),
                "auc_freq": _safe_float(r.get("auc_freq")),
                "brier_freq": _safe_float(r.get("brier_freq")),
                "rmse_sev_pos": _safe_float(r.get("rmse_sev_pos")),
                "q95_ratio_pos": _safe_float(r.get("q95_ratio_pos")),
                "q99_ratio_pos": _safe_float(r.get("q99_ratio_pos")),
                "rmse_prime_top1pct": _safe_float(r.get("rmse_prime_top1pct")),
                "kaggle_public_rmse_user": float(kaggle_public_rmse),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["rmse_primary_time", "model_version"], na_position="last"
        ).reset_index(drop=True)
    return out


def build_pred_distribution_compare(ctx: Mapping[str, object]) -> pd.DataFrame:
    rows = [
        _submission_stats(pd.DataFrame(ctx.get("v1_submission", pd.DataFrame())), "submission_v1"),
        _submission_stats(
            pd.DataFrame(ctx.get("v2_submission_robust", pd.DataFrame())), "submission_v2_robust"
        ),
        _submission_stats(
            pd.DataFrame(ctx.get("v2_submission_single", pd.DataFrame())), "submission_v2_single"
        ),
        _submission_stats(
            pd.DataFrame(ctx.get("v22_quick_submission_robust", pd.DataFrame())),
            "submission_v2_2_quick_robust",
        ),
        _submission_stats(
            pd.DataFrame(ctx.get("v22_quick_submission_challenger", pd.DataFrame())),
            "submission_v2_2_quick_challenger",
        ),
    ]
    return pd.DataFrame(rows)


def build_direct_tweedie_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, FloatArray, list[int]]:
    y = train[v2.TARGET_SEV_COL].to_numpy(dtype=float)
    drop_cols = [v2.TARGET_SEV_COL, v2.TARGET_FREQ_COL, *v2.ID_COLS]
    if id_col in train.columns:
        drop_cols.append(id_col)
    if id_col in test.columns:
        drop_cols.append(id_col)

    X_train = train.drop(columns=[c for c in drop_cols if c in train.columns]).copy()
    X_test = test.drop(columns=[c for c in drop_cols if c in test.columns]).copy()

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    cat_cols = [c for c in DIRECT_TWEEDIE_CAT_COLS if c in X_train.columns and c in X_test.columns]
    for c in cat_cols:
        X_train[c] = X_train[c].astype(str).fillna("NA")
        X_test[c] = X_test[c].astype(str).fillna("NA")
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    return X_train, X_test, y, cat_idx


def _catboost_tweedie_fit_predict(
    X_tr: pd.DataFrame,
    y_tr: FloatArray,
    X_va: pd.DataFrame,
    y_va: FloatArray,
    X_te: pd.DataFrame | None,
    cat_idx: Sequence[int],
    *,
    variance_power: float,
    seed: int,
    iterations: int,
    learning_rate: float,
    depth: int,
    od_wait: int,
    verbose: int | bool = False,
) -> tuple[FloatArray, FloatArray]:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function=f"Tweedie:variance_power={float(variance_power)}",
        eval_metric="RMSE",
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        random_seed=int(seed),
        od_type="Iter",
        od_wait=int(od_wait),
        verbose=verbose,
        allow_writing_files=False,
    )
    model.fit(
        X_tr,
        np.clip(np.asarray(y_tr, dtype=float), 0.0, None),
        eval_set=(X_va, np.clip(np.asarray(y_va, dtype=float), 0.0, None)),
        cat_features=list(cat_idx),
    )
    pred_va = np.maximum(np.asarray(model.predict(X_va), dtype=float), 0.0)
    pred_te = np.array([], dtype=float)
    if X_te is not None and len(X_te):
        pred_te = np.maximum(np.asarray(model.predict(X_te), dtype=float), 0.0)
    return pred_va, pred_te


@dataclass
class DirectTweedieFoldPayload:
    oof: FloatArray
    test_pred: FloatArray
    valid_mask: BoolArray
    alpha_ls: float
    variance_power: float
    cv_scheme: str


def _build_direct_pred_df(
    *,
    run_id: str,
    split: str,
    y_true_train: FloatArray,
    oof: FloatArray,
    valid_mask: BoolArray,
    test_pred: FloatArray,
    n_test: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    idx_train = np.where(valid_mask)[0]
    if len(idx_train):
        p_train = np.maximum(np.asarray(oof[idx_train], dtype=float), 0.0)
        y_train = np.asarray(y_true_train[idx_train], dtype=float)
        freq_proxy_scale = max(float(np.quantile(p_train, 0.99)), 1e-6)
        pred_freq = np.clip(p_train / freq_proxy_scale, 0.0, 1.0)
        rows.append(
            pd.DataFrame(
                {
                    "row_idx": idx_train.astype(int),
                    "is_test": 0,
                    "split": split,
                    "track": "direct_tweedie",
                    "engine": "catboost",
                    "family": "direct_tweedie",
                    "config_id": "cb_direct_tweedie_quick",
                    "seed": 42,
                    "severity_mode": "direct",
                    "calibration": "none",
                    "tail_mapper": "none",
                    "fold_id": np.nan,
                    "pred_freq": pred_freq,
                    "pred_sev": p_train,
                    "pred_prime": p_train,
                    "y_freq": (y_train > 0).astype(int),
                    "y_sev": y_train,
                    "run_id": run_id,
                }
            )
        )
    if n_test > 0 and len(test_pred):
        p_test = np.maximum(np.asarray(test_pred, dtype=float), 0.0)
        freq_proxy_scale_te = max(float(np.quantile(p_test, 0.99)), 1e-6) if len(p_test) else 1.0
        pred_freq_te = np.clip(p_test / freq_proxy_scale_te, 0.0, 1.0)
        rows.append(
            pd.DataFrame(
                {
                    "row_idx": np.arange(n_test, dtype=int),
                    "is_test": 1,
                    "split": split,
                    "track": "direct_tweedie",
                    "engine": "catboost",
                    "family": "direct_tweedie",
                    "config_id": "cb_direct_tweedie_quick",
                    "seed": 42,
                    "severity_mode": "direct",
                    "calibration": "none",
                    "tail_mapper": "none",
                    "fold_id": np.nan,
                    "pred_freq": pred_freq_te,
                    "pred_sev": p_test,
                    "pred_prime": p_test,
                    "y_freq": np.nan,
                    "y_sev": np.nan,
                    "run_id": run_id,
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _diag_from_pred_df(pred_df: pd.DataFrame, run_id: str, split: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "run_id": run_id,
        "split": split,
        "rmse_prime": np.nan,
        "mae_prime": np.nan,
        "rmse_prime_top1pct": np.nan,
        "q95_ratio_pos": np.nan,
        "q99_ratio_pos": np.nan,
        "n": np.nan,
        "n_nonzero": np.nan,
        "share_nonzero": np.nan,
    }
    if pred_df.empty:
        return out
    try:
        diag = ds.compute_oof_model_diagnostics(
            pred_df, run_id=run_id, split=split, decile_mode="zero_aware"
        )
        metrics = diag.get("metrics", pd.DataFrame())
        if not metrics.empty:
            row = metrics.iloc[0].to_dict()
            for k in list(out.keys()):
                if k in row:
                    out[k] = row.get(k)
            out["mae_prime_nonzero"] = row.get("mae_prime_nonzero")
            out["r2_prime_nonzero"] = row.get("r2_prime_nonzero")
        out["diag_tables"] = diag
    except Exception as e:
        out["diag_error"] = f"{type(e).__name__}: {e}"
    return out


def run_direct_tweedie_random_kfold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_col: str,
    variance_powers: Sequence[float] = (1.2, 1.4),
    n_splits_random: int = 5,
    seed: int = 42,
    iterations: int = 12000,
    learning_rate: float = 0.03,
    depth: int = 8,
    od_wait: int = 300,
    verbose: int | bool = False,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    X_train, X_test, y, cat_idx = build_direct_tweedie_features(train, test, id_col=id_col)
    kf = KFold(n_splits=int(n_splits_random), shuffle=True, random_state=int(seed))
    results: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    for p in variance_powers:
        oof = np.zeros(len(X_train), dtype=float)
        val_mask = np.zeros(len(X_train), dtype=bool)
        test_pred = np.zeros(len(X_test), dtype=float)
        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
            pred_va, pred_te = _catboost_tweedie_fit_predict(
                X_train.iloc[tr_idx],
                y[tr_idx],
                X_train.iloc[va_idx],
                y[va_idx],
                X_test,
                cat_idx,
                variance_power=float(p),
                seed=int(seed + fold_id),
                iterations=int(iterations),
                learning_rate=float(learning_rate),
                depth=int(depth),
                od_wait=int(od_wait),
                verbose=verbose,
            )
            oof[va_idx] = pred_va
            val_mask[va_idx] = True
            test_pred += pred_te / int(n_splits_random)

        alpha = _ls_alpha(y[val_mask], oof[val_mask])
        oof_cal = np.maximum(oof * alpha, 0.0)
        test_cal = np.maximum(test_pred * alpha, 0.0)
        run_id = f"direct_tweedie|catboost|random_kfold|p{p}|seed{seed}|alpha_ls"
        pred_df = _build_direct_pred_df(
            run_id=run_id,
            split="random_kfold",
            y_true_train=y,
            oof=oof_cal,
            valid_mask=val_mask,
            test_pred=test_cal,
            n_test=len(X_test),
        )
        pred_parts.append(pred_df)
        dist_df = (
            v2.build_prediction_distribution_table(pred_df) if not pred_df.empty else pd.DataFrame()
        )
        diag = _diag_from_pred_df(pred_df, run_id=run_id, split="random_kfold")
        dist_align: dict[str, Any] = {}
        if not dist_df.empty:
            row_oof = dist_df[
                (dist_df["run_id"].astype(str) == run_id)
                & (dist_df["split"] == "random_kfold")
                & (dist_df["sample"] == "oof")
            ]
            row_test = dist_df[
                (dist_df["run_id"].astype(str) == run_id)
                & (dist_df["split"] == "random_kfold")
                & (dist_df["sample"] == "test")
            ]
            if not row_oof.empty and not row_test.empty:
                o = row_oof.iloc[0]
                t = row_test.iloc[0]
                q90_ratio = _safe_float(t.get("pred_q90")) / max(
                    _safe_float(o.get("pred_q90")), 1e-9
                )
                q99_ratio = _safe_float(t.get("pred_q99")) / max(
                    _safe_float(o.get("pred_q99")), 1e-9
                )
                std_ratio = _safe_float(t.get("pred_std")) / max(
                    _safe_float(o.get("pred_std")), 1e-9
                )
                penalty = (
                    10.0 * abs(q99_ratio - 1.0)
                    + 5.0 * abs(q90_ratio - 1.0)
                    + 2.0 * abs(std_ratio - 1.0)
                )
                dist_align = {
                    "pred_q90_oof": _safe_float(o.get("pred_q90")),
                    "pred_q99_oof": _safe_float(o.get("pred_q99")),
                    "pred_q90_test": _safe_float(t.get("pred_q90")),
                    "pred_q99_test": _safe_float(t.get("pred_q99")),
                    "q90_test_over_oof": q90_ratio,
                    "q99_test_over_oof": q99_ratio,
                    "std_test_over_oof": std_ratio,
                    "distribution_alignment_penalty": penalty,
                    "distribution_alignment_score": -penalty,
                    "distribution_collapse_flag": int(t.get("distribution_collapse_flag", 0)),
                }

        result_row = {
            "track": "direct_tweedie_randomkfold",
            "candidate_id": f"direct_randomkfold_p{p}",
            "run_id": run_id,
            "cv_scheme": "random_kfold",
            "variance_power": float(p),
            "alpha_ls": float(alpha),
            "scale_multiplier": 1.0,
            "blend_weight": np.nan,
            "baseline_blend_source": None,
            "rmse_local": _rmse(y[val_mask], oof_cal[val_mask]),
            "rmse_primary_time": np.nan,
            "rmse_secondary_group": np.nan,
            "rmse_aux_blocked5": np.nan,
            "rmse_split_std": np.nan,
            "q95_ratio_pos": _safe_float(diag.get("q95_ratio_pos")),
            "q99_ratio_pos": _safe_float(diag.get("q99_ratio_pos")),
            "rmse_prime_top1pct": _safe_float(diag.get("rmse_prime_top1pct")),
            "distribution_alignment_score": _safe_float(
                dist_align.get("distribution_alignment_score")
            ),
            "distribution_alignment_penalty": _safe_float(
                dist_align.get("distribution_alignment_penalty")
            ),
            "dominant_gap_hypothesis": None,
            "selection_status": "candidate",
            "n_valid_oof": int(val_mask.sum()),
            "pred_test_array": test_cal,
        }
        result_row.update(dist_align)
        results.append(result_row)

    registry_df = (
        pd.DataFrame(results)
        .sort_values(["rmse_local", "distribution_alignment_penalty"], na_position="last")
        .reset_index(drop=True)
        if results
        else pd.DataFrame()
    )
    payloads_by_candidate: dict[str, dict[str, Any]] = {}
    best_row = registry_df.iloc[0].to_dict() if not registry_df.empty else None
    best_payload = None
    if pred_parts and not registry_df.empty:
        pred_all = pd.concat(pred_parts, ignore_index=True)
        for _, r in registry_df.iterrows():
            rid = str(r["run_id"])
            d = pred_all[pred_all["run_id"].astype(str) == rid].copy()
            if d.empty:
                continue
            oof_rows = d[d["is_test"] == 0].copy()
            test_rows = d[d["is_test"] == 1].copy()
            payloads_by_candidate[str(r["candidate_id"])] = {
                "candidate_id": str(r["candidate_id"]),
                "run_id": rid,
                "variance_power": float(r["variance_power"]),
                "alpha_ls": float(r["alpha_ls"]),
                "cv_scheme": "random_kfold",
                "oof_df": oof_rows,
                "test_df": test_rows,
                "pred_test": test_rows.sort_values("row_idx")["pred_prime"].to_numpy(dtype=float)
                if not test_rows.empty
                else np.array([], dtype=float),
            }
        if best_row is not None:
            best_payload = payloads_by_candidate.get(str(best_row["candidate_id"]))

    if out_dir is not None and not registry_df.empty:
        out = v2.ensure_dir(out_dir)
        registry_df.drop(columns=["pred_test_array"], errors="ignore").to_csv(
            out / "direct_tweedie_cv_registry.csv", index=False
        )

    return {
        "registry_df": registry_df,
        "best_row": best_row,
        "best_payload": best_payload,
        "payloads_by_candidate": payloads_by_candidate,
        "X_train": X_train,
        "X_test": X_test,
        "y": y,
        "cat_idx": cat_idx,
    }


def _run_direct_tweedie_on_folds(
    X_train: pd.DataFrame,
    y: FloatArray,
    X_test: pd.DataFrame,
    cat_idx: Sequence[int],
    folds: Mapping[int, SplitIndices],
    *,
    variance_power: float,
    seed: int,
    iterations: int,
    learning_rate: float,
    depth: int,
    od_wait: int,
    split_name: str,
    verbose: int | bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    oof = np.zeros(len(X_train), dtype=float)
    valid_mask = np.zeros(len(X_train), dtype=bool)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_rows: list[dict[str, Any]] = []
    n_folds = max(len(folds), 1)
    for fold_id, (tr_idx, va_idx) in folds.items():
        pred_va, pred_te = _catboost_tweedie_fit_predict(
            X_train.iloc[np.asarray(tr_idx, dtype=int)],
            y[np.asarray(tr_idx, dtype=int)],
            X_train.iloc[np.asarray(va_idx, dtype=int)],
            y[np.asarray(va_idx, dtype=int)],
            X_test,
            cat_idx,
            variance_power=float(variance_power),
            seed=int(seed + int(fold_id)),
            iterations=int(iterations),
            learning_rate=float(learning_rate),
            depth=int(depth),
            od_wait=int(od_wait),
            verbose=verbose,
        )
        va_idx_arr = np.asarray(va_idx, dtype=int)
        oof[va_idx_arr] = pred_va
        valid_mask[va_idx_arr] = True
        test_pred += pred_te / float(n_folds)
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "split": split_name,
                "n_valid": int(len(va_idx_arr)),
                "fold_rmse": _rmse(y[va_idx_arr], pred_va),
            }
        )

    alpha = _ls_alpha(y[valid_mask], oof[valid_mask])
    oof_cal = np.maximum(oof * alpha, 0.0)
    test_cal = np.maximum(test_pred * alpha, 0.0)
    return (
        {
            "oof": oof_cal,
            "test_pred": test_cal,
            "valid_mask": valid_mask,
            "alpha_ls": alpha,
            "fold_metrics": pd.DataFrame(fold_rows),
        },
        {"oof_raw": oof, "test_raw": test_pred},
    )


def run_direct_tweedie_v2_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    id_col: str,
    variance_powers: Sequence[float],
    seed: int = 42,
    iterations: int = 12000,
    learning_rate: float = 0.03,
    depth: int = 8,
    od_wait: int = 300,
    verbose: int | bool = False,
    return_payloads: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    X_train, X_test, y, cat_idx = build_direct_tweedie_features(train, test, id_col=id_col)
    splits = v2.build_split_registry(
        train, n_blocks_time=5, n_splits_group=5, group_col="id_client"
    )
    split_names = ["primary_time", "secondary_group", "aux_blocked5"]

    summary_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    payloads: dict[str, Any] = {
        "per_run_split": {},
        "combined_pred_df": pd.DataFrame(),
        "train_y": y,
    }

    for p in variance_powers:
        run_base_id = f"direct_tweedie|catboost|v2splits|p{p}|seed{seed}|alpha_ls"
        split_metric_rows: list[dict[str, Any]] = []
        for split_name in split_names:
            fold_payload, raw_payload = _run_direct_tweedie_on_folds(
                X_train,
                y,
                X_test,
                cat_idx,
                splits[split_name],
                variance_power=float(p),
                seed=int(seed),
                iterations=int(iterations),
                learning_rate=float(learning_rate),
                depth=int(depth),
                od_wait=int(od_wait),
                split_name=split_name,
                verbose=verbose,
            )
            run_id = f"{run_base_id}|{split_name}"
            pred_df = _build_direct_pred_df(
                run_id=run_id,
                split=split_name,
                y_true_train=y,
                oof=fold_payload["oof"],
                valid_mask=fold_payload["valid_mask"],
                test_pred=fold_payload["test_pred"],
                n_test=len(X_test),
            )
            pred_parts.append(pred_df)
            diag = _diag_from_pred_df(pred_df, run_id=run_id, split=split_name)
            dist_df = (
                v2.build_prediction_distribution_table(pred_df)
                if not pred_df.empty
                else pd.DataFrame()
            )
            dist_align: dict[str, Any] = {}
            if not dist_df.empty:
                o = dist_df[(dist_df["sample"] == "oof") & (dist_df["split"] == split_name)]
                t = dist_df[(dist_df["sample"] == "test") & (dist_df["split"] == split_name)]
                if not o.empty and not t.empty:
                    oo = o.iloc[0]
                    tt = t.iloc[0]
                    q90_ratio = _safe_float(tt.get("pred_q90")) / max(
                        _safe_float(oo.get("pred_q90")), 1e-9
                    )
                    q99_ratio = _safe_float(tt.get("pred_q99")) / max(
                        _safe_float(oo.get("pred_q99")), 1e-9
                    )
                    std_ratio = _safe_float(tt.get("pred_std")) / max(
                        _safe_float(oo.get("pred_std")), 1e-9
                    )
                    penalty = (
                        10.0 * abs(q99_ratio - 1.0)
                        + 5.0 * abs(q90_ratio - 1.0)
                        + 2.0 * abs(std_ratio - 1.0)
                    )
                    dist_align = {
                        "pred_q90_oof": _safe_float(oo.get("pred_q90")),
                        "pred_q99_oof": _safe_float(oo.get("pred_q99")),
                        "pred_q90_test": _safe_float(tt.get("pred_q90")),
                        "pred_q99_test": _safe_float(tt.get("pred_q99")),
                        "q90_test_over_oof": q90_ratio,
                        "q99_test_over_oof": q99_ratio,
                        "std_test_over_oof": std_ratio,
                        "distribution_alignment_penalty": penalty,
                        "distribution_alignment_score": -penalty,
                        "distribution_collapse_flag": int(tt.get("distribution_collapse_flag", 0)),
                    }
            row = {
                "track": "direct_tweedie_v2splits",
                "candidate_id": f"direct_v2splits_p{p}",
                "run_id": run_id,
                "cv_scheme": split_name,
                "variance_power": float(p),
                "alpha_ls": _safe_float(fold_payload.get("alpha_ls")),
                "scale_multiplier": 1.0,
                "blend_weight": np.nan,
                "baseline_blend_source": None,
                "rmse_local": _safe_float(diag.get("rmse_prime")),
                "rmse_primary_time": np.nan,
                "rmse_secondary_group": np.nan,
                "rmse_aux_blocked5": np.nan,
                "rmse_split_std": np.nan,
                "q95_ratio_pos": _safe_float(diag.get("q95_ratio_pos")),
                "q99_ratio_pos": _safe_float(diag.get("q99_ratio_pos")),
                "rmse_prime_top1pct": _safe_float(diag.get("rmse_prime_top1pct")),
                "distribution_alignment_score": _safe_float(
                    dist_align.get("distribution_alignment_score")
                ),
                "distribution_alignment_penalty": _safe_float(
                    dist_align.get("distribution_alignment_penalty")
                ),
                "dominant_gap_hypothesis": None,
                "selection_status": "candidate",
                "n_valid_oof": int(np.sum(fold_payload["valid_mask"])),
                "pred_test_array": fold_payload["test_pred"],
            }
            row.update(dist_align)
            split_metric_rows.append(row)
            payloads["per_run_split"][(float(p), split_name)] = {
                "pred_df": pred_df,
                "diag": diag,
                "fold_payload": fold_payload,
                "raw_payload": raw_payload,
                "dist_df": dist_df,
            }

        split_df = pd.DataFrame(split_metric_rows)
        if not split_df.empty:
            by_name = split_df.set_index("cv_scheme")
            agg = {
                "track": "direct_tweedie_v2splits",
                "candidate_id": f"direct_v2splits_p{p}",
                "run_id": f"{run_base_id}|multi",
                "cv_scheme": "multi",
                "variance_power": float(p),
                "alpha_ls": float(
                    np.nanmedian(pd.to_numeric(split_df["alpha_ls"], errors="coerce"))
                ),
                "scale_multiplier": 1.0,
                "blend_weight": np.nan,
                "baseline_blend_source": None,
                "rmse_local": _safe_float(by_name["rmse_local"].get("primary_time")),
                "rmse_primary_time": _safe_float(by_name["rmse_local"].get("primary_time")),
                "rmse_secondary_group": _safe_float(by_name["rmse_local"].get("secondary_group")),
                "rmse_aux_blocked5": _safe_float(by_name["rmse_local"].get("aux_blocked5")),
                "q95_ratio_pos": _safe_float(by_name["q95_ratio_pos"].get("primary_time")),
                "q99_ratio_pos": _safe_float(by_name["q99_ratio_pos"].get("primary_time")),
                "rmse_prime_top1pct": _safe_float(
                    by_name["rmse_prime_top1pct"].get("primary_time")
                ),
                "distribution_alignment_penalty": _safe_float(
                    by_name["distribution_alignment_penalty"].get("primary_time")
                ),
                "distribution_alignment_score": _safe_float(
                    by_name["distribution_alignment_score"].get("primary_time")
                ),
                "distribution_collapse_flag": _safe_float(
                    by_name["distribution_collapse_flag"].get("primary_time")
                ),
            }
            try:
                agg["pred_test_array"] = payloads["per_run_split"][(float(p), "primary_time")][
                    "fold_payload"
                ]["test_pred"]
            except Exception:
                agg["pred_test_array"] = None
            rmses = np.asarray(
                [agg["rmse_primary_time"], agg["rmse_secondary_group"], agg["rmse_aux_blocked5"]],
                dtype=float,
            )
            agg["rmse_split_std"] = (
                float(np.nanstd(rmses, ddof=0)) if np.sum(np.isfinite(rmses)) >= 2 else np.nan
            )
            rmse_primary = _safe_float(agg.get("rmse_primary_time"))
            rmse_secondary = _safe_float(agg.get("rmse_secondary_group"))
            rmse_aux = _safe_float(agg.get("rmse_aux_blocked5"))
            agg["rmse_gap_secondary"] = (
                rmse_secondary - rmse_primary
                if np.isfinite(rmse_secondary) and np.isfinite(rmse_primary)
                else np.nan
            )
            agg["rmse_gap_aux"] = (
                rmse_aux - rmse_primary
                if np.isfinite(rmse_aux) and np.isfinite(rmse_primary)
                else np.nan
            )
            summary_rows.extend(split_metric_rows)
            summary_rows.append(agg)

    summary_df = pd.DataFrame(summary_rows)
    if pred_parts:
        payloads["combined_pred_df"] = pd.concat(pred_parts, ignore_index=True)
    return (summary_df, payloads) if return_payloads else summary_df


def compute_scale_sweep(
    *,
    random_result: Mapping[str, Any],
    scale_sweep: Sequence[float] = (1.0, 1.1, 1.2, 1.3),
    v2_split_payloads: Mapping[object, Any] | None = None,
) -> pd.DataFrame:
    best_payload = random_result.get("best_payload")
    best_row = random_result.get("best_row")
    if not best_payload or not best_row:
        return pd.DataFrame()

    oof_df = best_payload.get("oof_df", pd.DataFrame()).copy()
    test_df = best_payload.get("test_df", pd.DataFrame()).copy()
    if oof_df.empty or test_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    y_oof = oof_df["y_sev"].to_numpy(dtype=float)
    p_oof_base = oof_df["pred_prime"].to_numpy(dtype=float)
    p_test_base = test_df["pred_prime"].to_numpy(dtype=float)
    base_run_id = str(best_payload.get("run_id"))
    p_value = _safe_float(best_payload.get("variance_power"))
    alpha_ls = _safe_float(best_payload.get("alpha_ls"))

    primary_payload = None
    if v2_split_payloads:
        primary_payload = v2_split_payloads.get((p_value, "primary_time"))

    for m in scale_sweep:
        m = float(m)
        p_oof = np.maximum(p_oof_base * m, 0.0)
        p_test = np.maximum(p_test_base * m, 0.0)
        rmse_local = _rmse(y_oof, p_oof)
        tmp_pred = _build_direct_pred_df(
            run_id=f"{base_run_id}|scale{m}",
            split="random_kfold",
            y_true_train=y_oof,
            oof=p_oof,
            valid_mask=np.ones(len(y_oof), dtype=bool),
            test_pred=p_test,
            n_test=len(p_test),
        )
        diag = _diag_from_pred_df(tmp_pred, run_id=f"{base_run_id}|scale{m}", split="random_kfold")
        dist_df = v2.build_prediction_distribution_table(tmp_pred)
        dist_row: dict[str, Any] = {}
        if not dist_df.empty:
            o = dist_df[(dist_df["sample"] == "oof") & (dist_df["split"] == "random_kfold")].iloc[0]
            t = dist_df[(dist_df["sample"] == "test") & (dist_df["split"] == "random_kfold")].iloc[
                0
            ]
            q90_ratio = _safe_float(t.get("pred_q90")) / max(_safe_float(o.get("pred_q90")), 1e-9)
            q99_ratio = _safe_float(t.get("pred_q99")) / max(_safe_float(o.get("pred_q99")), 1e-9)
            std_ratio = _safe_float(t.get("pred_std")) / max(_safe_float(o.get("pred_std")), 1e-9)
            penalty = (
                10.0 * abs(q99_ratio - 1.0)
                + 5.0 * abs(q90_ratio - 1.0)
                + 2.0 * abs(std_ratio - 1.0)
            )
            dist_row = {
                "pred_q90_oof": _safe_float(o.get("pred_q90")),
                "pred_q99_oof": _safe_float(o.get("pred_q99")),
                "pred_q90_test": _safe_float(t.get("pred_q90")),
                "pred_q99_test": _safe_float(t.get("pred_q99")),
                "q90_test_over_oof": q90_ratio,
                "q99_test_over_oof": q99_ratio,
                "std_test_over_oof": std_ratio,
                "distribution_alignment_penalty": penalty,
                "distribution_alignment_score": -penalty,
                "distribution_collapse_flag": int(t.get("distribution_collapse_flag", 0)),
            }

        row = {
            "track": "direct_tweedie_scale",
            "candidate_id": f"direct_scale_p{p_value}_x{m}",
            "parent_candidate_id": str(best_row.get("candidate_id")),
            "cv_scheme": "random_kfold",
            "variance_power": float(p_value),
            "alpha_ls": float(alpha_ls),
            "scale_multiplier": m,
            "blend_weight": np.nan,
            "baseline_blend_source": None,
            "rmse_local": float(rmse_local),
            "rmse_primary_time": np.nan,
            "rmse_secondary_group": np.nan,
            "rmse_aux_blocked5": np.nan,
            "rmse_split_std": np.nan,
            "q95_ratio_pos": _safe_float(diag.get("q95_ratio_pos")),
            "q99_ratio_pos": _safe_float(diag.get("q99_ratio_pos")),
            "rmse_prime_top1pct": _safe_float(diag.get("rmse_prime_top1pct")),
            "selection_status": "candidate",
            "pred_test_array": p_test,
        }
        row.update(dist_row)

        if primary_payload:
            pr = primary_payload["fold_payload"]
            valid_mask = np.asarray(pr["valid_mask"], dtype=bool)
            y_primary = (v2_split_payloads or {}).get("train_y") or random_result.get("y")
            if y_primary is not None:
                y_primary_arr = np.asarray(y_primary, dtype=float)
                p_primary = np.maximum(np.asarray(pr["oof"], dtype=float) * m, 0.0)
                row["rmse_primary_time"] = _rmse(y_primary_arr[valid_mask], p_primary[valid_mask])
        rows.append(row)
    return pd.DataFrame(rows)


def _find_baseline_submission_for_blend(
    ctx: Mapping[str, object],
) -> tuple[FloatArray | None, str | None, pd.DataFrame | None]:
    for key, label in [
        ("v1_submission", "submission_v1.csv"),
        ("v2_submission_robust", "artifacts/v2/submission_v2_robust.csv"),
        ("v22_quick_submission_robust", "artifacts/v2_2_quick/submission_v2_2_quick_robust.csv"),
    ]:
        df = pd.DataFrame(ctx.get(key, pd.DataFrame()))
        if df.empty:
            continue
        pred_col = "pred" if "pred" in df.columns else (df.columns[1] if df.shape[1] > 1 else None)
        if pred_col is None:
            continue
        return (
            np.maximum(
                pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0
            ),
            label,
            df,
        )
    return None, None, None


def _extract_v2_selected_primary_oof(
    ctx: Mapping[str, object],
) -> tuple[FloatArray | None, FloatArray | None, str | None]:
    v2_sel = pd.DataFrame(ctx.get("v2_selected", pd.DataFrame()))
    v2_oof = pd.DataFrame(ctx.get("v2_oof", pd.DataFrame()))
    if v2_sel.empty or v2_oof.empty or "run_id" not in v2_sel.columns:
        return None, None, None
    rid = str(v2_sel.iloc[0]["run_id"])
    d = v2_oof[
        (v2_oof["is_test"] == 0)
        & (v2_oof["split"].astype(str) == "primary_time")
        & (v2_oof["run_id"].astype(str) == rid)
    ].copy()
    if d.empty:
        return None, None, rid
    d = (
        d[["row_idx", "y_sev", "pred_prime"]]
        .drop_duplicates(subset=["row_idx"])
        .sort_values("row_idx")
    )
    return d["pred_prime"].to_numpy(dtype=float), d["y_sev"].to_numpy(dtype=float), rid


def compute_blend_sweep(
    *,
    ctx: Mapping[str, object],
    scale_df: pd.DataFrame,
    random_result: Mapping[str, Any],
    v2_splits_payloads: Mapping[object, Any] | None = None,
    blend_weights: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
) -> pd.DataFrame:
    if scale_df.empty:
        return pd.DataFrame()
    base_preds, base_source, _ = _find_baseline_submission_for_blend(ctx)
    if base_preds is None or base_source is None:
        return pd.DataFrame()

    scale_best = scale_df.sort_values(
        ["rmse_local", "distribution_alignment_penalty"], na_position="last"
    ).iloc[0]
    tweedie_test = np.asarray(scale_best["pred_test_array"], dtype=float)
    n = min(len(tweedie_test), len(base_preds))
    if n <= 0:
        return pd.DataFrame()
    tweedie_test = tweedie_test[:n]
    base_preds = base_preds[:n]

    baseline_oof_pred, baseline_oof_y, _ = _extract_v2_selected_primary_oof(ctx)
    direct_primary_oof = None
    p_val = _safe_float(scale_best.get("variance_power"))
    m_val = _safe_float(scale_best.get("scale_multiplier"), 1.0)
    if v2_splits_payloads:
        payload = v2_splits_payloads.get((p_val, "primary_time"))
        y_train = v2_splits_payloads.get("train_y")
        if payload and y_train is not None:
            pr = payload["fold_payload"]
            valid_mask = np.asarray(pr["valid_mask"], dtype=bool)
            oof_arr = np.maximum(np.asarray(pr["oof"], dtype=float) * m_val, 0.0)
            y_arr = np.asarray(y_train, dtype=float)
            direct_primary_oof = (oof_arr[valid_mask], y_arr[valid_mask])

    comparability_partial = not (
        baseline_oof_pred is not None
        and baseline_oof_y is not None
        and direct_primary_oof is not None
        and len(baseline_oof_pred) == len(direct_primary_oof[0])
        and len(baseline_oof_y) == len(direct_primary_oof[0])
    )

    rows: list[dict[str, Any]] = []
    for w in blend_weights:
        w = float(w)
        blended_test = np.maximum(w * tweedie_test + (1.0 - w) * base_preds, 0.0)
        row = {
            "track": "blend",
            "candidate_id": f"blend_direct_scale_w{w}",
            "cv_scheme": "test_only" if comparability_partial else "primary_time_proxy",
            "variance_power": p_val,
            "alpha_ls": _safe_float(scale_best.get("alpha_ls")),
            "scale_multiplier": m_val,
            "blend_weight": w,
            "baseline_blend_source": base_source,
            "rmse_local": np.nan,
            "rmse_primary_time": np.nan,
            "rmse_secondary_group": np.nan,
            "rmse_aux_blocked5": np.nan,
            "rmse_split_std": np.nan,
            "q95_ratio_pos": np.nan,
            "q99_ratio_pos": np.nan,
            "rmse_prime_top1pct": np.nan,
            "comparability_partial": int(comparability_partial),
            "pred_test_array": blended_test,
            "selection_status": "candidate",
        }
        if (
            not comparability_partial
            and baseline_oof_pred is not None
            and baseline_oof_y is not None
            and direct_primary_oof is not None
        ):
            direct_pred, y_true = direct_primary_oof
            blended_oof = np.maximum(w * direct_pred + (1.0 - w) * baseline_oof_pred, 0.0)
            y_true = np.asarray(y_true, dtype=float)
            row["rmse_local"] = _rmse(y_true, blended_oof)
            row["rmse_primary_time"] = row["rmse_local"]
            pos = y_true > 0
            if np.any(pos):
                row["q95_ratio_pos"] = float(
                    np.quantile(blended_oof[pos], 0.95) / max(np.quantile(y_true[pos], 0.95), 1e-9)
                )
                row["q99_ratio_pos"] = float(
                    np.quantile(blended_oof[pos], 0.99) / max(np.quantile(y_true[pos], 0.99), 1e-9)
                )
                thr = float(np.quantile(y_true, 0.99))
                mask_top = y_true >= thr
                row["rmse_prime_top1pct"] = (
                    _rmse(y_true[mask_top], blended_oof[mask_top]) if np.any(mask_top) else np.nan
                )

        audit = v2.compute_prediction_distribution_audit(
            blended_test, run_id=str(row["candidate_id"]), split="test", sample="test"
        )
        row.update(
            {
                "pred_q90_test": _safe_float(audit.get("pred_q90")),
                "pred_q99_test": _safe_float(audit.get("pred_q99")),
                "pred_std_test": _safe_float(audit.get("pred_std")),
                "distribution_collapse_flag": _safe_float(audit.get("distribution_collapse_flag")),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _classify_gap_hypothesis_from_flags(row: Mapping[str, Any]) -> str:
    tail_under = bool(int(_safe_float(row.get("tail_undercoverage_flag"), 0.0)))
    ood = bool(int(_safe_float(row.get("ood_risk_flag"), 0.0)))
    cv = bool(int(_safe_float(row.get("cv_instability_flag"), 0.0)))
    if cv and not (tail_under or ood):
        return "Overfitting CV probable"
    if tail_under and ood:
        return "Mixte (OOD + queue)"
    if tail_under:
        return "Gap domine par sous-modelisation de la queue"
    if ood:
        return "Gap domine par OOD / shift"
    return "Overfitting non prouve"


def classify_gap_cause_dualtrack(
    *,
    ctx: Mapping[str, object],
    v23_candidates_df: pd.DataFrame,
    kaggle_public_rmse_user: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    v22_gap = pd.DataFrame(ctx.get("v22_quick_gap_report", pd.DataFrame()))
    if not v22_gap.empty:
        for _, r in v22_gap.head(5).iterrows():
            rows.append(
                {
                    "candidate_id": str(r.get("run_id")),
                    "track": "v2_two_part",
                    "cv_instability_flag": int(_safe_float(r.get("cv_instability_flag"), 0.0)),
                    "ood_risk_flag": int(_safe_float(r.get("ood_risk_flag"), 0.0)),
                    "tail_undercoverage_flag": int(
                        _safe_float(r.get("tail_undercoverage_flag"), 0.0)
                    ),
                    "tail_overcorrection_flag": 0,
                    "public_lb_heuristic_flag": 0,
                    "kaggle_gap_hypothesis": str(r.get("kaggle_gap_hypothesis", "")),
                    "evidence_summary": f"V2 shortlist; q99={_safe_float(r.get('q99_ratio_pos')):.3f}; split_std={_safe_float(r.get('rmse_split_std')):.3f}",
                }
            )

    d = v23_candidates_df.copy() if v23_candidates_df is not None else pd.DataFrame()
    if not d.empty:
        for _, r in d.iterrows():
            q99 = _safe_float(r.get("q99_ratio_pos"))
            gap_sec = _safe_float(r.get("rmse_secondary_group")) - _safe_float(
                r.get("rmse_primary_time")
            )
            gap_aux = _safe_float(r.get("rmse_aux_blocked5")) - _safe_float(
                r.get("rmse_primary_time")
            )
            split_std = _safe_float(r.get("rmse_split_std"))
            dist_ratio = _safe_float(r.get("q99_test_over_oof"))
            public_heur = int(
                str(r.get("track", "")).startswith("direct_tweedie_scale")
                or str(r.get("track", "")) == "blend"
                or str(r.get("cv_scheme", "")) == "random_kfold"
            )
            row = {
                "candidate_id": str(r.get("candidate_id")),
                "track": str(r.get("track", "")),
                "cv_instability_flag": int(
                    (np.isfinite(gap_sec) and gap_sec > 1.0)
                    or (np.isfinite(gap_aux) and gap_aux > 1.0)
                    or (np.isfinite(split_std) and split_std > 1.2)
                ),
                "ood_risk_flag": int(
                    np.isfinite(dist_ratio) and (dist_ratio < 0.85 or dist_ratio > 1.15)
                ),
                "tail_undercoverage_flag": int(np.isfinite(q99) and q99 < 0.50),
                "tail_overcorrection_flag": int(np.isfinite(q99) and q99 > 1.20),
                "public_lb_heuristic_flag": public_heur,
                "evidence_summary": "; ".join(
                    [
                        f"{k}={_safe_float(r.get(k)):.3f}"
                        for k in [
                            "rmse_primary_time",
                            "rmse_secondary_group",
                            "rmse_aux_blocked5",
                            "q99_ratio_pos",
                        ]
                        if np.isfinite(_safe_float(r.get(k)))
                    ]
                ),
            }
            row["kaggle_gap_hypothesis"] = _classify_gap_hypothesis_from_flags(row)
            rows.append(row)

    diag_df = (
        pd.DataFrame(rows).drop_duplicates(subset=["candidate_id", "track"], keep="last")
        if rows
        else pd.DataFrame()
    )
    summary = {
        "kaggle_public_rmse_user": float(kaggle_public_rmse_user),
        "note_metric_non_comparable": True,
        "n_candidates_analyzed": int(len(diag_df)),
        "dominant_hypothesis_counts": diag_df["kaggle_gap_hypothesis"].value_counts().to_dict()
        if not diag_df.empty
        else {},
        "dominant_hypothesis_top_run": (
            str(diag_df.iloc[0]["kaggle_gap_hypothesis"]) if not diag_df.empty else None
        ),
        "initial_working_hypothesis": "Mixte (OOD + queue)",
        "overfitting_cv_proven": False,
    }
    return diag_df, summary


def _compute_selection_score_quick_dualtrack(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    for c in [
        "rmse_primary_time",
        "rmse_secondary_group",
        "rmse_aux_blocked5",
        "rmse_local",
        "q99_ratio_pos",
        "distribution_alignment_penalty",
        "shakeup_std_gap",
        "distribution_collapse_flag",
    ]:
        if c not in out.columns:
            out[c] = np.nan
    out["rmse_gap_secondary_pos"] = np.maximum(
        pd.to_numeric(out["rmse_secondary_group"], errors="coerce")
        - pd.to_numeric(out["rmse_primary_time"], errors="coerce"),
        0.0,
    )
    out["rmse_gap_aux_pos"] = np.maximum(
        pd.to_numeric(out["rmse_aux_blocked5"], errors="coerce")
        - pd.to_numeric(out["rmse_primary_time"], errors="coerce"),
        0.0,
    )
    q99 = pd.to_numeric(out["q99_ratio_pos"], errors="coerce")
    out["tail_penalty"] = 20.0 * (q99 - 1.0).abs()
    out["distribution_alignment_penalty"] = pd.to_numeric(
        out["distribution_alignment_penalty"], errors="coerce"
    ).fillna(0.0)
    out["shakeup_penalty"] = pd.to_numeric(out["shakeup_std_gap"], errors="coerce").fillna(0.0)
    base_rmse = (
        pd.to_numeric(out["rmse_primary_time"], errors="coerce")
        .fillna(pd.to_numeric(out["rmse_local"], errors="coerce"))
        .fillna(1e9)
    )
    out["selection_score_dualtrack"] = (
        base_rmse
        + 0.5 * out["rmse_gap_secondary_pos"]
        + 0.5 * out["rmse_gap_aux_pos"]
        + out["tail_penalty"]
        + out["distribution_alignment_penalty"]
        + out["shakeup_penalty"]
    )
    out["has_robust_cv"] = (
        out[["rmse_primary_time", "rmse_secondary_group", "rmse_aux_blocked5"]].notna().sum(axis=1)
        >= 2
    )
    out["cv_instability_flag"] = (
        (
            (out["rmse_gap_secondary_pos"] > 1.0)
            | (out["rmse_gap_aux_pos"] > 1.0)
            | (pd.to_numeric(out.get("rmse_split_std", np.nan), errors="coerce") > 1.2)
        )
        .fillna(False)
        .astype(int)
    )
    out["tail_undercoverage_flag"] = (q99 < 0.50).fillna(False).astype(int)
    out["tail_overcorrection_flag"] = (q99 > 1.20).fillna(False).astype(int)
    out["distribution_collapse_flag"] = pd.to_numeric(
        out["distribution_collapse_flag"], errors="coerce"
    ).fillna(0.0)
    out["public_lb_heuristic_flag"] = (
        out["track"].astype(str).isin(["direct_tweedie_scale", "blend"]).astype(int)
    )
    out["passes_robust_guardrails"] = (
        out["has_robust_cv"]
        & (out["cv_instability_flag"] == 0)
        & (out["tail_overcorrection_flag"] == 0)
        & (out["distribution_collapse_flag"] <= 0)
    )
    return out


def select_dualtrack_submissions(
    candidates_df: pd.DataFrame,
    *,
    baseline_submission_df: pd.DataFrame | None = None,
    baseline_name: str = "baseline_existing",
    id_col: str = "index",
) -> pd.DataFrame:
    if candidates_df is None or candidates_df.empty:
        if baseline_submission_df is None or baseline_submission_df.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "role": "robust",
                    "candidate_id": "fallback_baseline",
                    "track": "fallback",
                    "selection_status": "selected_robust",
                    "risk_tag": "fallback",
                    "submission_source": baseline_name,
                    "pred_test_array": np.maximum(
                        pd.to_numeric(baseline_submission_df["pred"], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(dtype=float),
                        0.0,
                    ),
                    "id_col": id_col,
                }
            ]
        )

    d = _compute_selection_score_quick_dualtrack(candidates_df)
    robust_pool = d[d["passes_robust_guardrails"]].copy()
    if robust_pool.empty:
        robust_pool = d[(d["has_robust_cv"]) & (d["cv_instability_flag"] == 0)].copy()
    robust_row = (
        robust_pool.sort_values(
            ["selection_score_dualtrack", "rmse_primary_time"], na_position="last"
        )
        .iloc[0]
        .to_dict()
        if not robust_pool.empty
        else None
    )

    challenger_pool = d.copy()
    challenger_pool["_lb_pref"] = (
        challenger_pool["track"]
        .astype(str)
        .map({"blend": 0, "direct_tweedie_scale": 1, "direct_tweedie_randomkfold": 2})
        .fillna(3)
    )
    challenger_row = (
        challenger_pool.sort_values(
            ["_lb_pref", "selection_score_dualtrack", "rmse_local"], na_position="last"
        )
        .iloc[0]
        .to_dict()
        if not challenger_pool.empty
        else None
    )

    out_rows: list[dict[str, Any]] = []
    if robust_row is not None:
        robust_row["role"] = "robust"
        robust_row["selection_status"] = "selected_robust"
        robust_row["risk_tag"] = "robust"
        robust_row["id_col"] = id_col
        out_rows.append(robust_row)
    elif baseline_submission_df is not None and not baseline_submission_df.empty:
        out_rows.append(
            {
                "role": "robust",
                "candidate_id": "fallback_baseline",
                "track": "fallback",
                "selection_status": "selected_robust",
                "risk_tag": "fallback",
                "submission_source": baseline_name,
                "pred_test_array": np.maximum(
                    pd.to_numeric(baseline_submission_df["pred"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float),
                    0.0,
                ),
                "id_col": id_col,
            }
        )

    if challenger_row is not None:
        challenger_row["role"] = "lb_challenger"
        challenger_row["selection_status"] = "selected_challenger"
        challenger_row["risk_tag"] = (
            "public_private_risk"
            if int(challenger_row.get("public_lb_heuristic_flag", 0)) == 1
            else "challenger"
        )
        challenger_row["id_col"] = id_col
        if not out_rows or str(out_rows[0].get("candidate_id")) != str(
            challenger_row["candidate_id"]
        ):
            out_rows.append(challenger_row)
    return pd.DataFrame(out_rows)


def save_submission(
    test_df: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    preds: FloatArray,
    out_path: str | Path,
    sample_submission_df: pd.DataFrame | None = None,
) -> Path:
    pred = np.maximum(np.asarray(preds, dtype=float), 0.0)
    sub = pd.DataFrame({id_col: test_df[id_col].to_numpy(), target_col: pred})
    if (
        sample_submission_df is not None
        and not sample_submission_df.empty
        and id_col in sample_submission_df.columns
    ):
        sub = sample_submission_df[[id_col]].merge(sub, on=id_col, how="left")
        sub[target_col] = (
            pd.to_numeric(sub[target_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
    p = Path(out_path)
    v2.ensure_dir(p.parent)
    sub.to_csv(p, index=False)
    return p


def _extract_submission_baseline(ctx: Mapping[str, object]) -> tuple[pd.DataFrame, str]:
    for key, label in [
        ("v2_submission_robust", "artifacts/v2/submission_v2_robust.csv"),
        ("v1_submission", "artifacts/submission_v1.csv"),
        ("v22_quick_submission_robust", "artifacts/v2_2_quick/submission_v2_2_quick_robust.csv"),
    ]:
        df = pd.DataFrame(ctx.get(key, pd.DataFrame()))
        if not df.empty and "pred" in df.columns:
            return df.copy(), label
    return pd.DataFrame(), "none"


def _submission_distribution_rows(selection_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if selection_df is None or selection_df.empty:
        return pd.DataFrame()
    for _, r in selection_df.iterrows():
        pred = r.get("pred_test_array")
        if pred is None:
            continue
        try:
            arr = np.asarray(pred, dtype=float)
            if arr.size == 0:
                continue
            audit = v2.compute_prediction_distribution_audit(
                arr, run_id=str(r.get("candidate_id")), split="test", sample="test"
            )
            rows.append(
                {"role": str(r.get("role")), "candidate_id": str(r.get("candidate_id")), **audit}
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def build_oof_compare_bridge(
    *,
    ctx: Mapping[str, object],
    direct_random_result: Mapping[str, Any] | None,
    direct_v2_payloads: Mapping[object, Any] | None,
    out_dir: str | Path,
) -> pd.DataFrame:
    out_frames: list[pd.DataFrame] = []
    v1_oof = pd.DataFrame(ctx.get("v1_oof", pd.DataFrame()))
    v1_rr = pd.DataFrame(ctx.get("v1_run_registry", pd.DataFrame()))
    v2_oof = pd.DataFrame(ctx.get("v2_oof", pd.DataFrame()))
    v2_sel = pd.DataFrame(ctx.get("v2_selected", pd.DataFrame()))

    if not v1_oof.empty and not v1_rr.empty:
        best = _best_row_by_rmse(v1_rr, split="primary_time")
        if best:
            d = v1_oof[
                (v1_oof["is_test"] == 0)
                & (v1_oof["split"].astype(str) == "primary_time")
                & (v1_oof["engine"].astype(str) == str(best.get("engine")))
                & (v1_oof["config_id"].astype(str) == str(best.get("config_id")))
                & (v1_oof["seed"].astype(float).astype(int) == int(_safe_float(best.get("seed"))))
                & (v1_oof["severity_mode"].astype(str) == str(best.get("severity_mode")))
                & (v1_oof["calibration"].astype(str) == str(best.get("calibration")))
            ][["row_idx", "y_sev", "pred_prime"]].drop_duplicates(subset=["row_idx"])
            if not d.empty:
                out_frames.append(
                    d.rename(columns={"y_sev": "y_true", "pred_prime": "pred_v1_best"})
                )

    if not v2_oof.empty and not v2_sel.empty and "run_id" in v2_sel.columns:
        rid = str(v2_sel.iloc[0]["run_id"])
        d = v2_oof[
            (v2_oof["is_test"] == 0)
            & (v2_oof["split"].astype(str) == "primary_time")
            & (v2_oof["run_id"].astype(str) == rid)
        ][["row_idx", "y_sev", "pred_prime"]].drop_duplicates(subset=["row_idx"])
        if not d.empty:
            out_frames.append(
                d.rename(columns={"y_sev": "y_true", "pred_prime": "pred_v2_selected"})
            )

    if direct_random_result and direct_random_result.get("best_payload"):
        d = direct_random_result["best_payload"]["oof_df"].copy()
        if not d.empty:
            d = d[["row_idx", "y_sev", "pred_prime"]].drop_duplicates(subset=["row_idx"])
            out_frames.append(
                d.rename(columns={"y_sev": "y_true", "pred_prime": "pred_direct_random_best"})
            )

    if (
        direct_v2_payloads
        and direct_random_result
        and direct_random_result.get("best_row") is not None
    ):
        pval = float(direct_random_result["best_row"]["variance_power"])
        payload = direct_v2_payloads.get((pval, "primary_time"))
        if payload:
            d = payload["pred_df"]
            if isinstance(d, pd.DataFrame) and not d.empty:
                d = d[d["is_test"] == 0][["row_idx", "y_sev", "pred_prime"]].drop_duplicates(
                    subset=["row_idx"]
                )
                if not d.empty:
                    out_frames.append(
                        d.rename(columns={"y_sev": "y_true", "pred_prime": "pred_direct_primary"})
                    )

    if not out_frames:
        return pd.DataFrame()
    out = out_frames[0]
    for df in out_frames[1:]:
        out = out.merge(df, on=["row_idx", "y_true"], how="outer")
    out = out.sort_values("row_idx").reset_index(drop=True)
    p = v2.ensure_dir(out_dir) / "oof_compare_bridge.parquet"
    out.to_parquet(p, index=False)
    return out


def _maybe_run_shakeup_from_candidate(
    *,
    candidate_row: Mapping[str, Any],
    direct_random_result: Mapping[str, Any] | None,
    direct_v2_payloads: Mapping[object, Any] | None,
    n_sim: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    track = str(candidate_row.get("track", ""))
    cid = str(candidate_row.get("candidate_id", ""))
    d = pd.DataFrame()
    if track == "direct_tweedie_randomkfold":
        payload = (direct_random_result or {}).get("payloads_by_candidate", {}).get(cid)
        if payload:
            d = pd.DataFrame(payload.get("oof_df", pd.DataFrame()))
    elif track == "direct_tweedie_v2splits":
        p = _safe_float(candidate_row.get("variance_power"))
        payload = (direct_v2_payloads or {}).get((p, "primary_time"))
        if payload:
            d = pd.DataFrame(payload.get("pred_df", pd.DataFrame()))
            if not d.empty:
                d = d[d["is_test"] == 0]
    if d.empty or "y_sev" not in d.columns or "pred_prime" not in d.columns:
        return pd.DataFrame()
    d = d[d["y_sev"].notna() & d["pred_prime"].notna()].drop_duplicates(subset=["row_idx"])
    if d.empty:
        return pd.DataFrame()
    return v2.simulate_public_private_shakeup_v2(
        d["y_sev"].to_numpy(dtype=float),
        d["pred_prime"].to_numpy(dtype=float),
        n_sim=int(n_sim),
        seed=int(seed),
        stratified_tail=False,
    )


def materialize_dualtrack_outputs(
    *,
    ctx: Mapping[str, object],
    test_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    sample_submission_df: pd.DataFrame | None,
    selected_df: pd.DataFrame,
    direct_random_result: Mapping[str, Any] | None = None,
    direct_v2_payloads: Mapping[object, Any] | None = None,
    run_shakeup_quick: bool = True,
    n_sim_shakeup: int = 300,
    seed: int = 42,
    out_dir: str | Path = ARTIFACT_V23_DIR,
) -> dict[str, Any]:
    out = v2.ensure_dir(out_dir)
    result: dict[str, Any] = {
        "selected_df": selected_df.copy() if selected_df is not None else pd.DataFrame(),
        "submission_paths": {},
        "pred_distribution_compare": pd.DataFrame(),
        "shakeup_bridge_top": pd.DataFrame(),
    }
    if selected_df is None or selected_df.empty:
        return result

    for _, row in selected_df.iterrows():
        role = str(row.get("role"))
        preds = row.get("pred_test_array")
        if preds is None:
            continue
        preds_arr = np.maximum(np.asarray(preds, dtype=float), 0.0)
        filename = (
            "submission_v2_3_robust.csv"
            if role == "robust"
            else "submission_v2_3_lb_challenger.csv"
        )
        if len(preds_arr) != len(test_df):
            # fallback strictness: if misaligned, skip writing and keep trace in selected_df
            continue
        path = save_submission(
            test_df,
            id_col=id_col,
            target_col=target_col,
            preds=preds_arr,
            out_path=out / filename,
            sample_submission_df=sample_submission_df,
        )
        result["submission_paths"][role] = path

    pred_dist_df = _submission_distribution_rows(selected_df)
    if not pred_dist_df.empty:
        pred_dist_df.to_csv(out / "pred_distribution_compare.csv", index=False)
    result["pred_distribution_compare"] = pred_dist_df

    shake_parts = []
    if run_shakeup_quick:
        for _, row in selected_df.iterrows():
            sh = _maybe_run_shakeup_from_candidate(
                candidate_row=row,
                direct_random_result=direct_random_result,
                direct_v2_payloads=direct_v2_payloads,
                n_sim=n_sim_shakeup,
                seed=seed,
            )
            if not sh.empty:
                sh = sh.copy()
                sh["candidate_id"] = str(row.get("candidate_id"))
                sh["role"] = str(row.get("role"))
                shake_parts.append(sh)
        if shake_parts:
            shake_df = pd.concat(shake_parts, ignore_index=True)
            shake_df.to_parquet(out / "shakeup_bridge_top.parquet", index=False)
            result["shakeup_bridge_top"] = shake_df
    return result


def write_submission_decision_report(
    *,
    ctx: Mapping[str, object],
    kaggle_public_rmse_user: float,
    bridge_summary_df: pd.DataFrame,
    diagnosis_summary: Mapping[str, Any],
    diagnosis_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    out_dir: str | Path = ARTIFACT_V23_DIR,
) -> Path:
    out = v2.ensure_dir(out_dir)
    lines: list[str] = []
    lines.append("# Submission decision V2.3 Dual-Track Quick")
    lines.append("")
    lines.append("## 1) Contexte")
    lines.append(f"- Kaggle public (utilisateur): ~{float(kaggle_public_rmse_user):.3f}")
    lines.append("- OOF local et Kaggle public ne sont pas directement comparables.")
    lines.append(
        "- Ce cycle compare un track robuste V2 et un track direct Tweedie CatBoost (random KFold + scale/blend)."
    )
    lines.append("")
    lines.append("## 2) Diagnostic du gap (resume)")
    lines.append(
        f"- Hypothese dominante (dual-track): {diagnosis_summary.get('dominant_hypothesis_top_run')}"
    )
    lines.append(f"- Comptage hypotheses: {diagnosis_summary.get('dominant_hypothesis_counts')}")
    lines.append("- Overfitting CV: non conclu sans preuve inter-splits.")
    lines.append("- Public LB peut reagir positivement a un scaling global sans garantie privee.")
    lines.append("")
    lines.append("## 3) Bridge V1 / V2 / V2.2 (local)")
    if bridge_summary_df is not None and not bridge_summary_df.empty:
        cols = [
            c
            for c in [
                "model_version",
                "track",
                "run_id",
                "rmse_primary_time",
                "rmse_secondary_group",
                "rmse_aux_blocked5",
                "q95_ratio_pos",
                "q99_ratio_pos",
                "rmse_prime_top1pct",
            ]
            if c in bridge_summary_df.columns
        ]
        lines.append("")
        lines.append(bridge_summary_df[cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- Indisponible")
        lines.append("")
    lines.append("## 4) Classification explicite overfit / OOD / queue")
    if diagnosis_df is not None and not diagnosis_df.empty:
        lines.append("")
        lines.append(diagnosis_df.head(12).to_markdown(index=False))
        lines.append("")
    else:
        lines.append("- Indisponible")
        lines.append("")

    recommendation = "Ne pas soumettre"
    if selected_df is not None and not selected_df.empty:
        if (selected_df["role"].astype(str) == "robust").any():
            recommendation = "Envoyer robust"
        elif (selected_df["role"].astype(str) == "lb_challenger").any():
            recommendation = "Envoyer challenger"

    lines.append("## 5) Recommandation d'envoi")
    lines.append(f"- **Decision**: {recommendation}")
    lines.append("")
    if selected_df is not None and not selected_df.empty:
        lines.append("## 6) Candidats selectionnes")
        show_cols = [
            c
            for c in [
                "role",
                "candidate_id",
                "track",
                "risk_tag",
                "rmse_primary_time",
                "rmse_secondary_group",
                "rmse_aux_blocked5",
                "q99_ratio_pos",
                "selection_score_dualtrack",
            ]
            if c in selected_df.columns
        ]
        lines.append("")
        lines.append(selected_df[show_cols].to_markdown(index=False))
        lines.append("")
    lines.append("## 7) Interpretation du test `p=1.2 x1.3`")
    lines.append(
        "- Le gain public peut venir d'un meilleur alignement de distribution/queue et/ou d'un tuning public-LB opportuniste."
    )
    lines.append(
        "- Le notebook compare random KFold et splits robustes (`primary_time`, `secondary_group`, `aux_blocked5`) pour qualifier ce risque."
    )
    lines.append("")
    lines.append("## 8) Notes")
    lines.append("- Cycle quick (1-2h): comparatif cible, pas tuning exhaustif.")
    lines.append(
        "- Aucune modification des fichiers existants; outputs ecrits sous `artifacts/v2_3_dualtrack_quick/`."
    )
    lines.append("")

    path = out / "submission_decision_v2_3_dualtrack.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def train_run(config_path: str) -> dict[str, Any]:
    from insurance_pricing import train_run as _train_run

    return dict(_train_run(config_path))

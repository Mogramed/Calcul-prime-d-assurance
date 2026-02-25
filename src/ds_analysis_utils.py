from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.v2_pipeline import (
    ID_COLS,
    INDEX_COL,
    TARGET_FREQ_COL,
    TARGET_SEV_COL,
    build_prediction_distribution_table,
    compute_prediction_distribution_audit,
    ensure_dir,
    load_train_test,
)


DEFAULT_DS_DIR = Path("artifacts") / "ds"


def _safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mad(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _psi_from_series(train_s: pd.Series, test_s: pd.Series, bins: int = 10) -> float:
    a = pd.to_numeric(train_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.to_numeric(test_s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(a, qs))
    if len(cuts) < 3:
        return float("nan")
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    a_bin = pd.cut(a, bins=cuts, include_lowest=True)
    b_bin = pd.cut(b, bins=cuts, include_lowest=True)
    a_dist = a_bin.value_counts(normalize=True).sort_index()
    b_dist = b_bin.value_counts(normalize=True).sort_index()
    idx = a_dist.index.union(b_dist.index)
    a_p = a_dist.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    b_p = b_dist.reindex(idx, fill_value=0.0).to_numpy(dtype=float)
    eps = 1e-6
    a_p = np.clip(a_p, eps, None)
    b_p = np.clip(b_p, eps, None)
    return float(np.sum((a_p - b_p) * np.log(a_p / b_p)))


def _role_guess(col: str, dtype: str, train_present: bool, test_present: bool) -> str:
    c = col.lower()
    if c == INDEX_COL.lower():
        return "id_index"
    if c in {k.lower() for k in ID_COLS}:
        return "id_group"
    if c == TARGET_FREQ_COL.lower():
        return "target_freq"
    if c == TARGET_SEV_COL.lower():
        return "target_sev"
    if "date" in c or "debut" in c or "fin_" in c:
        return "time_or_period"
    if dtype.startswith("object"):
        return "categorical"
    if dtype.startswith(("int", "float")):
        return "numeric"
    return "unknown"


def load_project_datasets(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_train_test(data_dir)


def build_data_dictionary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    cols = sorted(set(train.columns).union(set(test.columns)))
    rows = []
    for c in cols:
        tr_present = c in train.columns
        te_present = c in test.columns
        tr_s = train[c] if tr_present else pd.Series(dtype="object")
        te_s = test[c] if te_present else pd.Series(dtype="object")
        dtype_train = str(tr_s.dtype) if tr_present else None
        dtype_test = str(te_s.dtype) if te_present else None
        rows.append(
            {
                "column": c,
                "present_train": int(tr_present),
                "present_test": int(te_present),
                "dtype_train": dtype_train,
                "dtype_test": dtype_test,
                "nunique_train": int(tr_s.nunique(dropna=False)) if tr_present else np.nan,
                "nunique_test": int(te_s.nunique(dropna=False)) if te_present else np.nan,
                "missing_rate_train": float(tr_s.isna().mean()) if tr_present else np.nan,
                "missing_rate_test": float(te_s.isna().mean()) if te_present else np.nan,
                "sample_values_train": " | ".join(map(str, tr_s.dropna().astype(str).head(3).tolist()))
                if tr_present
                else "",
                "role_guess": _role_guess(c, dtype_train or dtype_test or "unknown", tr_present, te_present),
            }
        )
    return pd.DataFrame(rows).sort_values(["role_guess", "column"]).reset_index(drop=True)


def classify_columns(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    df = build_data_dictionary(train, test).copy()
    df["is_target"] = df["role_guess"].isin(["target_freq", "target_sev"]).astype(int)
    df["is_id_like"] = df["role_guess"].str.startswith("id_").astype(int)
    df["is_categorical"] = (df["role_guess"] == "categorical").astype(int)
    df["is_numeric"] = (df["role_guess"] == "numeric").astype(int)
    df["high_cardinality_train"] = (
        pd.to_numeric(df["nunique_train"], errors="coerce").fillna(0) >= 100
    ).astype(int)
    return df


def detect_leakage_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lower_targets = {TARGET_FREQ_COL.lower(), TARGET_SEV_COL.lower()}
    for c in df.columns:
        cl = c.lower()
        risk = []
        action = []
        if c == INDEX_COL or c in ID_COLS:
            risk.append("identifier")
            action.append("exclude_as_feature")
        if cl in lower_targets:
            risk.append("target")
            action.append("target_only")
        if any(token in cl for token in ["sinistre", "claim", "cout", "montant"]) and cl not in lower_targets:
            risk.append("target_proximity")
            action.append("manual_review")
        if any(token in cl for token in ["after", "post", "resolution", "indemn"]):
            risk.append("post_event_suspect")
            action.append("manual_review")
        if risk:
            rows.append(
                {
                    "column": c,
                    "risk_types": ",".join(risk),
                    "risk_level": "high" if ("target" in risk or "identifier" in risk) else "medium",
                    "recommended_action": ",".join(dict.fromkeys(action)),
                }
            )
    return pd.DataFrame(rows).sort_values(["risk_level", "column"]).reset_index(drop=True)


def compute_missingness_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    cols = sorted(set(train.columns).union(set(test.columns)))
    for c in cols:
        row = {
            "scope": "global",
            "group_col": None,
            "group_value": None,
            "column": c,
            "missing_rate_train": float(train[c].isna().mean()) if c in train.columns else np.nan,
            "missing_rate_test": float(test[c].isna().mean()) if c in test.columns else np.nan,
            "missing_gap_test_minus_train": (
                (float(test[c].isna().mean()) if c in test.columns else np.nan)
                - (float(train[c].isna().mean()) if c in train.columns else np.nan)
            ),
        }
        rows.append(row)

    if group_cols:
        for g in group_cols:
            if g not in train.columns:
                continue
            top_levels = train[g].astype(str).value_counts(dropna=False).head(8).index.tolist()
            for level in top_levels:
                mask = train[g].astype(str) == str(level)
                if mask.sum() == 0:
                    continue
                sub = train.loc[mask]
                for c in train.columns:
                    rows.append(
                        {
                            "scope": "by_group",
                            "group_col": g,
                            "group_value": str(level),
                            "column": c,
                            "missing_rate_train": float(sub[c].isna().mean()),
                            "missing_rate_test": np.nan,
                            "missing_gap_test_minus_train": np.nan,
                        }
                    )
    out = pd.DataFrame(rows)
    return out.sort_values(["scope", "column", "group_col", "group_value"]).reset_index(drop=True)


def compute_rule_violations(train: pd.DataFrame) -> pd.DataFrame:
    checks: list[tuple[str, pd.Series, str]] = []
    if {"age_conducteur1", "anciennete_permis1"}.issubset(train.columns):
        checks.append(
            (
                "permis_gt_age_conducteur1",
                (pd.to_numeric(train["anciennete_permis1"], errors="coerce") >
                 pd.to_numeric(train["age_conducteur1"], errors="coerce")),
                "anciennete_permis1 > age_conducteur1",
            )
        )
    if {"age_conducteur2", "anciennete_permis2"}.issubset(train.columns):
        checks.append(
            (
                "permis_gt_age_conducteur2",
                (pd.to_numeric(train["anciennete_permis2"], errors="coerce") >
                 pd.to_numeric(train["age_conducteur2"], errors="coerce")),
                "anciennete_permis2 > age_conducteur2",
            )
        )
    if "poids_vehicule" in train.columns:
        checks.append(("poids_zero_or_neg", pd.to_numeric(train["poids_vehicule"], errors="coerce") <= 0, "poids_vehicule <= 0"))
    if "cylindre_vehicule" in train.columns:
        checks.append(("cylindre_zero_or_neg", pd.to_numeric(train["cylindre_vehicule"], errors="coerce") <= 0, "cylindre_vehicule <= 0"))
    if "age_conducteur1" in train.columns:
        a1 = pd.to_numeric(train["age_conducteur1"], errors="coerce")
        checks.append(("age_conducteur1_impossible", (a1 < 16) | (a1 > 100), "age_conducteur1 out of [16,100]"))
    if TARGET_SEV_COL in train.columns:
        sev = pd.to_numeric(train[TARGET_SEV_COL], errors="coerce")
        checks.append(("sinistre_negatif", sev < 0, "montant_sinistre < 0"))
        checks.append(("petit_sinistre_lt_50", (sev > 0) & (sev < 50), "0 < montant_sinistre < 50"))

    rows = []
    for name, mask, rule in checks:
        mask = pd.Series(mask).fillna(False)
        rows.append(
            {
                "check_name": name,
                "rule": rule,
                "n_violations": int(mask.sum()),
                "ratio_violations": float(mask.mean()),
                "example_indices": ",".join(map(str, train.loc[mask].index[:5].tolist())),
            }
        )
    return pd.DataFrame(rows).sort_values("n_violations", ascending=False).reset_index(drop=True)


def compute_outlier_report(df: pd.DataFrame, cols: list[str], method: str = "iqr_mad") -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = _safe_series(df[c])
        x = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 5:
            continue
        q1, q3 = np.quantile(x, [0.25, 0.75])
        iqr = float(q3 - q1)
        med = float(np.median(x))
        mad = _mad(x)
        lo_iqr = q1 - 1.5 * iqr
        hi_iqr = q3 + 1.5 * iqr
        mad_scale = max(1.4826 * mad, 1e-9) if np.isfinite(mad) else np.nan
        mz = 0.6745 * (x - med) / mad_scale if np.isfinite(mad_scale) else pd.Series(np.nan, index=x.index)
        rows.append(
            {
                "column": c,
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "median": med,
                "std": float(np.std(x)),
                "iqr": iqr,
                "mad": float(mad) if np.isfinite(mad) else np.nan,
                "q01": float(np.quantile(x, 0.01)),
                "q05": float(np.quantile(x, 0.05)),
                "q95": float(np.quantile(x, 0.95)),
                "q99": float(np.quantile(x, 0.99)),
                "max": float(np.max(x)),
                "min": float(np.min(x)),
                "n_out_iqr": int(((x < lo_iqr) | (x > hi_iqr)).sum()),
                "ratio_out_iqr": float(((x < lo_iqr) | (x > hi_iqr)).mean()),
                "n_out_mad_z35": int((np.abs(mz) > 3.5).sum()) if np.isfinite(mad_scale) else np.nan,
                "ratio_out_mad_z35": float((np.abs(mz) > 3.5).mean()) if np.isfinite(mad_scale) else np.nan,
                "method": method,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio_out_iqr", ascending=False).reset_index(drop=True)


def compute_drift_numeric_ks_psi(
    train: pd.DataFrame,
    test: pd.DataFrame,
    num_cols: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for c in num_cols:
        if c not in train.columns or c not in test.columns:
            continue
        tr = _safe_series(train[c]).replace([np.inf, -np.inf], np.nan).dropna()
        te = _safe_series(test[c]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(tr) < 5 or len(te) < 5:
            continue
        ks = stats.ks_2samp(tr.to_numpy(dtype=float), te.to_numpy(dtype=float), method="auto")
        rows.append(
            {
                "column": c,
                "n_train_non_null": int(len(tr)),
                "n_test_non_null": int(len(te)),
                "mean_train": float(np.mean(tr)),
                "mean_test": float(np.mean(te)),
                "median_train": float(np.median(tr)),
                "median_test": float(np.median(te)),
                "std_train": float(np.std(tr)),
                "std_test": float(np.std(te)),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "psi": _psi_from_series(tr, te, bins=bins),
            }
        )
    return pd.DataFrame(rows).sort_values(["psi", "ks_stat"], ascending=[False, False]).reset_index(drop=True)


def compute_drift_categorical_chi2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
    top_k: int = 50,
) -> pd.DataFrame:
    rows = []
    for c in cat_cols:
        if c not in train.columns or c not in test.columns:
            continue
        tr = train[c].astype(str).fillna("NA")
        te = test[c].astype(str).fillna("NA")
        tr_vc = tr.value_counts(dropna=False)
        te_vc = te.value_counts(dropna=False)
        seen_tr = set(tr_vc.index.astype(str))
        seen_te = set(te_vc.index.astype(str))
        unseen = seen_te - seen_tr
        top_levels = tr_vc.head(top_k).index.astype(str).tolist()
        levels = sorted(set(top_levels) | set(te_vc.head(top_k).index.astype(str).tolist()))
        if len(levels) < 2:
            chi2_stat, pvalue, dof = np.nan, np.nan, np.nan
        else:
            cont = np.vstack(
                [
                    tr_vc.reindex(levels, fill_value=0).to_numpy(dtype=float),
                    te_vc.reindex(levels, fill_value=0).to_numpy(dtype=float),
                ]
            )
            try:
                chi2_stat, pvalue, dof, _ = stats.chi2_contingency(cont)
            except Exception:
                chi2_stat, pvalue, dof = np.nan, np.nan, np.nan
        rows.append(
            {
                "column": c,
                "train_unique": int(tr.nunique(dropna=False)),
                "test_unique": int(te.nunique(dropna=False)),
                "unseen_test_levels": int(len(unseen)),
                "unseen_ratio_on_test_levels": float(len(unseen) / max(len(seen_te), 1)),
                "unseen_ratio_on_test_rows": float(te.isin(list(unseen)).mean()) if unseen else 0.0,
                "chi2_stat_topk": float(chi2_stat) if np.isfinite(chi2_stat) else np.nan,
                "chi2_pvalue_topk": float(pvalue) if np.isfinite(pvalue) else np.nan,
                "chi2_dof_topk": float(dof) if np.isfinite(dof) else np.nan,
                "top_train_levels": " | ".join(map(str, top_levels[:10])),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["unseen_ratio_on_test_rows", "unseen_test_levels"], ascending=[False, False]
    ).reset_index(drop=True)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    tab = pd.crosstab(x.astype(str), y.astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan")
    try:
        chi2, _, _, _ = stats.chi2_contingency(tab)
    except Exception:
        return float("nan")
    n = tab.values.sum()
    if n <= 0:
        return float("nan")
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return float("nan")
    return float(np.sqrt(phi2corr / denom))


def compute_cramers_v_table(df: pd.DataFrame, cat_cols: list[str], max_cols: int = 12) -> pd.DataFrame:
    cols = [c for c in cat_cols if c in df.columns][:max_cols]
    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            val = _cramers_v(df[c1], df[c2])
            rows.append({"col_a": c1, "col_b": c2, "cramers_v": val})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def compute_segment_target_tables(train: pd.DataFrame, segment_cols: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if TARGET_SEV_COL not in train.columns:
        return out
    sev = pd.to_numeric(train[TARGET_SEV_COL], errors="coerce").fillna(0.0)
    freq = (sev > 0).astype(int)
    tmp = train.copy()
    tmp["_y_sev"] = sev
    tmp["_y_freq"] = freq
    for c in segment_cols:
        if c not in tmp.columns:
            continue
        grp = (
            tmp.groupby(c, dropna=False)
            .agg(
                n=("_y_sev", "size"),
                claim_rate=("_y_freq", "mean"),
                severity_mean_pos=("_y_sev", lambda s: float(np.mean(s[s > 0])) if (s > 0).any() else np.nan),
                severity_median_pos=("_y_sev", lambda s: float(np.median(s[s > 0])) if (s > 0).any() else np.nan),
                severity_q95_pos=("_y_sev", lambda s: float(np.quantile(s[s > 0], 0.95)) if (s > 0).any() else np.nan),
                severity_q99_pos=("_y_sev", lambda s: float(np.quantile(s[s > 0], 0.99)) if (s > 0).any() else np.nan),
                pure_premium_obs=("_y_sev", "mean"),
            )
            .reset_index()
            .rename(columns={c: "segment_value"})
        )
        grp.insert(0, "segment_col", c)
        out[c] = grp.sort_values("pure_premium_obs", ascending=False).reset_index(drop=True)
    return out


def build_feature_blocks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        cl = c.lower()
        if c in [INDEX_COL, *ID_COLS, TARGET_FREQ_COL, TARGET_SEV_COL]:
            continue
        if "conducteur" in cl or "permis" in cl:
            block = "driver"
        elif "vehicule" in cl or "marque" in cl or "modele" in cl or "essence" in cl:
            block = "vehicle"
        elif "contrat" in cl or "paiement" in cl or "bonus" in cl or "utilisation" in cl:
            block = "contract_usage"
        elif "postal" in cl or cl.startswith("cp"):
            block = "location"
        elif cl.startswith("is_") or "flag" in cl:
            block = "indicator"
        elif "ratio" in cl or "_x_" in cl or "par_" in cl:
            block = "interaction_ratio"
        else:
            block = "other"
        rows.append({"feature": c, "feature_block": block})
    return pd.DataFrame(rows).sort_values(["feature_block", "feature"]).reset_index(drop=True)


def compute_preprocessing_recommendations(
    meta_df: pd.DataFrame,
    cardinality_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> pd.DataFrame:
    meta = meta_df.copy()
    if "column" not in meta.columns and "feature" in meta.columns:
        meta = meta.rename(columns={"feature": "column"})
    card = cardinality_df.copy()
    if "column" not in card.columns and "feature" in card.columns:
        card = card.rename(columns={"feature": "column"})
    miss = missing_df.copy()
    miss = miss[miss["scope"] == "global"][["column", "missing_rate_train", "missing_rate_test"]].drop_duplicates()
    merged = meta.merge(miss, on="column", how="left").merge(card, on="column", how="left")

    rows = []
    for _, r in merged.iterrows():
        col = r["column"]
        role = str(r.get("role_guess", "unknown"))
        dtype = str(r.get("dtype_train", r.get("dtype", "")))
        nunique = float(r.get("nunique_train", np.nan))
        miss_rate = float(r.get("missing_rate_train", np.nan)) if pd.notna(r.get("missing_rate_train", np.nan)) else 0.0
        if role in {"target_freq", "target_sev"}:
            action = "target_only"
            encoding = "none"
            imputation = "none"
            scaling = "none"
            transform = "none"
        elif role.startswith("id_"):
            action = "exclude_feature"
            encoding = "none"
            imputation = "none"
            scaling = "none"
            transform = "none"
        elif dtype.startswith("object"):
            action = "use_feature"
            if nunique <= 10:
                encoding = "one_hot_or_catboost"
            elif nunique <= 100:
                encoding = "catboost_or_target_encoding_cv"
            else:
                encoding = "target_encoding_cv_or_catboost_rare_grouping"
            imputation = "Unknown + is_missing" if miss_rate > 0 else "none_or_Unknown"
            scaling = "none"
            transform = "rare_grouping_if_needed"
        else:
            action = "use_feature"
            encoding = "numeric"
            imputation = "median + is_missing" if miss_rate > 0 else "none_or_median"
            scaling = "robust_scaler_if_distance_or_linear_model"
            if any(tok in col.lower() for tok in ["prix", "montant", "valeur"]):
                transform = "consider_log1p"
            else:
                transform = "none_or_binning_if_explainability"
        rows.append(
            {
                "column": col,
                "action": action,
                "encoding": encoding,
                "imputation": imputation,
                "scaling": scaling,
                "transform": transform,
                "comment": "TE uniquement en CV cross-fit; IDs exclus; features high-card à regrouper",
            }
        )
    return pd.DataFrame(rows).sort_values(["action", "column"]).reset_index(drop=True)


def sample_for_exploration(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    if stratify_col is None or stratify_col not in df.columns:
        return df.sample(n=n, random_state=seed).copy()
    out_parts = []
    rng = np.random.default_rng(seed)
    tmp = df.copy()
    tmp["_strata"] = tmp[stratify_col].astype(str).fillna("NA")
    vc = tmp["_strata"].value_counts(dropna=False)
    props = vc / vc.sum()
    for k, p in props.items():
        g = tmp[tmp["_strata"] == k]
        take = max(1, int(round(n * p)))
        take = min(take, len(g))
        out_parts.append(g.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    out = pd.concat(out_parts, ignore_index=False).drop(columns=["_strata"], errors="ignore")
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.copy()


def _prepare_mixed_matrix(
    df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    num = [c for c in num_cols if c in df.columns]
    cat = [c for c in cat_cols if c in df.columns]
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num,
            )
        )
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            )
        )
    if not transformers:
        return np.empty((len(df), 0)), []
    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    X = ct.fit_transform(df)
    feat_names = []
    try:
        feat_names = list(ct.get_feature_names_out())
    except Exception:
        feat_names = [f"x{i}" for i in range(X.shape[1])]
    return np.asarray(X, dtype=float), feat_names


def compute_gower_like_distance_sample(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> np.ndarray:
    num = [c for c in num_cols if c in df.columns]
    cat = [c for c in cat_cols if c in df.columns]
    n = len(df)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    parts = []
    if num:
        Xn = df[num].apply(pd.to_numeric, errors="coerce")
        med = Xn.median()
        Xn = Xn.fillna(med)
        ranges = (Xn.max() - Xn.min()).replace(0, 1.0)
        Xn = (Xn - Xn.min()) / ranges
        xn = Xn.to_numpy(dtype=float)
        num_dist = np.zeros((n, n), dtype=float)
        for j in range(xn.shape[1]):
            col = xn[:, [j]]
            num_dist += np.abs(col - col.T)
        num_dist /= max(xn.shape[1], 1)
        parts.append(num_dist)

    if cat:
        Xc = df[cat].astype(str).fillna("NA").to_numpy(dtype=object)
        cat_dist = np.zeros((n, n), dtype=float)
        for j in range(Xc.shape[1]):
            col = Xc[:, [j]]
            cat_dist += (col != col.T).astype(float)
        cat_dist /= max(Xc.shape[1], 1)
        parts.append(cat_dist)

    if not parts:
        return np.zeros((n, n), dtype=float)
    D = np.mean(parts, axis=0)
    np.fill_diagonal(D, 0.0)
    return D


def fit_mixed_embedding_proxy(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_components: int = 2,
) -> pd.DataFrame:
    X, feat_names = _prepare_mixed_matrix(df, num_cols=num_cols, cat_cols=cat_cols)
    if X.shape[1] == 0:
        return pd.DataFrame({"comp_1": np.zeros(len(df)), "comp_2": np.zeros(len(df))}, index=df.index)
    reducer = TruncatedSVD(n_components=n_components, random_state=42) if X.shape[1] > 50 else PCA(n_components=n_components, random_state=42)
    comps = reducer.fit_transform(X)
    cols = [f"comp_{i+1}" for i in range(comps.shape[1])]
    out = pd.DataFrame(comps, index=df.index, columns=cols)
    for i in range(n_components - comps.shape[1]):
        out[f"comp_{comps.shape[1] + i + 1}"] = 0.0
    return out[[f"comp_{i+1}" for i in range(n_components)]]


def compute_error_by_deciles(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return pd.DataFrame()
    rank = pd.qcut(pd.Series(y), q=min(n_bins, len(np.unique(y))), duplicates="drop")
    d = pd.DataFrame({"y_true": y, "y_pred": p, "bin": rank.astype(str)})
    d["err"] = d["y_pred"] - d["y_true"]
    d["abs_err"] = d["err"].abs()
    out = (
        d.groupby("bin")
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
    return out


def compute_calibration_table(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
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
) -> dict[str, pd.DataFrame]:
    d = oof_df.copy()
    if "run_id" not in d.columns:
        # fallback legacy
        cols = ["feature_set", "engine", "family", "config_id", "seed", "severity_mode", "calibration", "tail_mapper"]
        cols = [c for c in cols if c in d.columns]
        if cols:
            rid = d[cols[0]].astype(str)
            for c in cols[1:]:
                rid = rid + "|" + d[c].astype(str)
            d["run_id"] = rid
    d = d[(d.get("is_test", 0) == 0) & (d["split"] == split) & (d["run_id"].astype(str) == str(run_id))].copy()
    if d.empty:
        return {
            "metrics": pd.DataFrame(),
            "error_by_decile_true": pd.DataFrame(),
            "error_by_decile_pred": pd.DataFrame(),
            "calibration_freq": pd.DataFrame(),
            "residuals": pd.DataFrame(),
            "distribution": pd.DataFrame(),
        }
    d = d[d["y_sev"].notna() & d["pred_prime"].notna()].copy()
    y = d["y_sev"].to_numpy(dtype=float)
    pred_prime = d["pred_prime"].to_numpy(dtype=float)
    pred_freq = d["pred_freq"].to_numpy(dtype=float) if "pred_freq" in d.columns else np.full(len(d), np.nan)
    pred_sev = d["pred_sev"].to_numpy(dtype=float) if "pred_sev" in d.columns else np.full(len(d), np.nan)
    y_freq = d["y_freq"].to_numpy(dtype=float) if "y_freq" in d.columns else (y > 0).astype(float)
    pos = y > 0

    metrics = {
        "run_id": str(run_id),
        "split": split,
        "n": int(len(d)),
        "rmse_prime": _rmse(y, pred_prime),
        "mae_prime": float(mean_absolute_error(y, pred_prime)),
        "r2_prime": float(r2_score(y, pred_prime)),
        "auc_freq": float(roc_auc_score(y_freq, pred_freq)) if len(np.unique(y_freq)) > 1 else np.nan,
        "gini_freq": float(2 * roc_auc_score(y_freq, pred_freq) - 1) if len(np.unique(y_freq)) > 1 else np.nan,
        "brier_freq": float(brier_score_loss(y_freq.astype(int), np.clip(pred_freq, 0, 1))) if np.isfinite(pred_freq).any() else np.nan,
        "logloss_freq": float(log_loss(y_freq.astype(int), np.clip(pred_freq, 1e-6, 1 - 1e-6))) if np.isfinite(pred_freq).any() and len(np.unique(y_freq)) > 1 else np.nan,
        "pr_auc_freq": float(average_precision_score(y_freq.astype(int), pred_freq)) if len(np.unique(y_freq)) > 1 else np.nan,
        "rmse_sev_pos": _rmse(y[pos], pred_sev[pos]) if pos.any() and np.isfinite(pred_sev[pos]).any() else np.nan,
        "mae_sev_pos": float(mean_absolute_error(y[pos], pred_sev[pos])) if pos.any() and np.isfinite(pred_sev[pos]).any() else np.nan,
        "q99_ratio_pos": (float(np.quantile(pred_sev[pos], 0.99)) / max(float(np.quantile(y[pos], 0.99)), 1e-9)) if pos.any() and np.isfinite(pred_sev[pos]).any() else np.nan,
    }
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

    err_true = compute_error_by_deciles(y_true=y, y_pred=pred_prime, n_bins=10)
    err_true.insert(0, "decile_basis", "y_true")

    # predicted deciles
    try:
        bins_pred = pd.qcut(pd.Series(pred_prime), q=min(10, len(np.unique(pred_prime))), duplicates="drop")
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
    dist = build_prediction_distribution_table(d.assign(is_test=0))
    if not dist.empty:
        dist = dist.assign(run_id=str(run_id), split=split)
    return {
        "metrics": metrics_df,
        "error_by_decile_true": err_true,
        "error_by_decile_pred": err_pred,
        "calibration_freq": cal,
        "residuals": residuals,
        "distribution": dist,
    }


def build_cardinality_report(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = sorted(set(train.columns).union(set(test.columns)))
    for c in cols:
        tr_present = c in train.columns
        te_present = c in test.columns
        if not tr_present and not te_present:
            continue
        tr = train[c].astype(str) if tr_present else pd.Series(dtype=str)
        te = test[c].astype(str) if te_present else pd.Series(dtype=str)
        rows.append(
            {
                "column": c,
                "dtype_train": str(train[c].dtype) if tr_present else None,
                "nunique_train": int(train[c].nunique(dropna=False)) if tr_present else np.nan,
                "nunique_test": int(test[c].nunique(dropna=False)) if te_present else np.nan,
                "top1_ratio_train": float(tr.value_counts(normalize=True, dropna=False).iloc[0]) if tr_present and len(tr) else np.nan,
                "rare_ratio_train_lt10": float((tr.value_counts(dropna=False) < 10).sum() / max(tr.nunique(dropna=False), 1)) if tr_present else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("nunique_train", ascending=False).reset_index(drop=True)


def build_feature_engineering_catalog(df: pd.DataFrame) -> pd.DataFrame:
    blocks = build_feature_blocks(df)
    rows = []
    for _, r in blocks.iterrows():
        feat = str(r["feature"])
        block = str(r["feature_block"])
        fl = feat.lower()
        if "cp2" in fl or "cp3" in fl or "postal" in fl:
            rationale = "hierarchie geographique pour robustesse OOD"
        elif "marque" in fl or "modele" in fl:
            rationale = "granularite vehicule / risque specifique"
        elif "ratio" in fl or "par_" in fl:
            rationale = "normalisation de taille/puissance/valeur"
        elif "_x_" in fl:
            rationale = "interaction metier non lineaire"
        elif fl.startswith("is_"):
            rationale = "signal binaire / gestion missing / OOD"
        else:
            rationale = "feature brute utile au scoring"
        rows.append({"feature": feat, "feature_block": block, "rationale": rationale})
    return pd.DataFrame(rows).sort_values(["feature_block", "feature"]).reset_index(drop=True)


def fit_kmeans_exploration(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_clusters: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    X, _ = _prepare_mixed_matrix(df, num_cols=num_cols, cat_cols=cat_cols)
    if X.shape[0] == 0:
        return pd.DataFrame(index=df.index)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    out = pd.DataFrame(index=df.index)
    out["cluster"] = labels
    return out


def compute_linkage_from_distance(distance_matrix: np.ndarray, method: str = "average") -> np.ndarray:
    if distance_matrix.shape[0] < 2:
        return np.zeros((0, 4))
    condensed = squareform(distance_matrix, checks=False)
    return linkage(condensed, method=method)


def export_analysis_tables(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = ensure_dir(out_dir)
    for name, df in tables.items():
        if df is None:
            continue
        if isinstance(df, pd.DataFrame):
            df.to_csv(out / f"{name}.csv", index=False)
        else:
            raise TypeError(f"Unsupported table type for {name}: {type(df)}")


__all__ = [
    "DEFAULT_DS_DIR",
    "_rmse",
    "_prepare_mixed_matrix",
    "load_project_datasets",
    "build_data_dictionary",
    "classify_columns",
    "detect_leakage_risk_columns",
    "compute_missingness_report",
    "compute_rule_violations",
    "compute_outlier_report",
    "compute_drift_numeric_ks_psi",
    "compute_drift_categorical_chi2",
    "compute_cramers_v_table",
    "compute_segment_target_tables",
    "build_feature_blocks",
    "compute_preprocessing_recommendations",
    "sample_for_exploration",
    "compute_gower_like_distance_sample",
    "fit_mixed_embedding_proxy",
    "compute_oof_model_diagnostics",
    "compute_error_by_deciles",
    "compute_calibration_table",
    "export_analysis_tables",
    "build_cardinality_report",
    "build_feature_engineering_catalog",
    "fit_kmeans_exploration",
    "compute_linkage_from_distance",
    "build_prediction_distribution_table",
    "compute_prediction_distribution_audit",
]

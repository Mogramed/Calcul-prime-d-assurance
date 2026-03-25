from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from insurance_pricing._typing import FloatArray, as_float_array
from insurance_pricing.data.schema import (
    ID_COLS,
    INDEX_COL,
    TARGET_FREQ_COL,
    TARGET_SEV_COL,
)


def _safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mad(x: pd.Series | FloatArray) -> float:
    arr = as_float_array(pd.to_numeric(pd.Series(x), errors="coerce"))
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


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


def detect_leakage_risk_columns(
    df: pd.DataFrame,
    *,
    strict_word_boundary: bool = True,
) -> pd.DataFrame:
    rows = []
    lower_targets = {TARGET_FREQ_COL.lower(), TARGET_SEV_COL.lower()}
    neutral_post_tokens = {"postal", "postcode", "code_postal"}

    def _matches_token(name: str, token: str) -> bool:
        if strict_word_boundary:
            # Split snake/camel-ish names into lexical tokens.
            parts = [p for p in re.split(r"[^a-z0-9]+", name.lower()) if p]
            return token.lower() in parts
        return token.lower() in name.lower()

    for c in df.columns:
        cl = c.lower()
        risk = []
        action = []
        rules_matched = []
        if c == INDEX_COL or c in ID_COLS:
            risk.append("identifier")
            action.append("exclude_as_feature")
            rules_matched.append("identifier_exact")
        if cl in lower_targets:
            risk.append("target")
            action.append("target_only")
            rules_matched.append("target_exact")
        if (
            any(tok in cl for tok in ["sinistre", "claim", "cout", "montant"])
            and cl not in lower_targets
        ):
            risk.append("target_proximity")
            action.append("manual_review")
            rules_matched.append("target_proximity_token")
        post_event_tokens = ["after", "post", "resolution", "indemn"]
        if cl not in neutral_post_tokens and any(
            _matches_token(cl, token) for token in post_event_tokens
        ):
            risk.append("post_event_suspect")
            action.append("manual_review")
            rules_matched.append("post_event_token")
        if risk:
            rows.append(
                {
                    "column": c,
                    "risk_types": ",".join(risk),
                    "risk_level": "high"
                    if ("target" in risk or "identifier" in risk)
                    else "medium",
                    "recommended_action": ",".join(dict.fromkeys(action)),
                    "rule_matched": ",".join(dict.fromkeys(rules_matched)),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["column", "risk_types", "risk_level", "recommended_action", "rule_matched"]
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
        present_train = c in train.columns
        present_test = c in test.columns
        miss_train = float(train[c].isna().mean()) if present_train else np.nan
        miss_test = float(test[c].isna().mean()) if present_test else np.nan
        row = {
            "scope": "global",
            "group_col": None,
            "group_value": None,
            "column": c,
            "present_train": int(present_train),
            "present_test": int(present_test),
            "applicable_train": int(present_train),
            "applicable_test": int(present_test),
            "missing_rate_train": miss_train,
            "missing_rate_test": miss_test,
            "missing_rate_train_effective": miss_train if present_train else np.nan,
            "missing_rate_test_effective": miss_test if present_test else np.nan,
            "missing_rate_train_filled": float(np.nan_to_num(miss_train, nan=0.0)),
            "missing_rate_test_filled": float(np.nan_to_num(miss_test, nan=0.0)),
            "missing_gap_test_minus_train": (
                (miss_test if present_test else np.nan) - (miss_train if present_train else np.nan)
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
                    present_test = c in test.columns
                    rows.append(
                        {
                            "scope": "by_group",
                            "group_col": g,
                            "group_value": str(level),
                            "column": c,
                            "present_train": 1,
                            "present_test": int(present_test),
                            "applicable_train": 1,
                            "applicable_test": 0,
                            "missing_rate_train": float(sub[c].isna().mean()),
                            "missing_rate_test": np.nan,
                            "missing_rate_train_effective": float(sub[c].isna().mean()),
                            "missing_rate_test_effective": np.nan,
                            "missing_rate_train_filled": float(
                                np.nan_to_num(float(sub[c].isna().mean()), nan=0.0)
                            ),
                            "missing_rate_test_filled": 0.0,
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
                (
                    pd.to_numeric(train["anciennete_permis1"], errors="coerce")
                    > pd.to_numeric(train["age_conducteur1"], errors="coerce")
                ),
                "anciennete_permis1 > age_conducteur1",
            )
        )
    if {"age_conducteur2", "anciennete_permis2"}.issubset(train.columns):
        checks.append(
            (
                "permis_gt_age_conducteur2",
                (
                    pd.to_numeric(train["anciennete_permis2"], errors="coerce")
                    > pd.to_numeric(train["age_conducteur2"], errors="coerce")
                ),
                "anciennete_permis2 > age_conducteur2",
            )
        )
    if "poids_vehicule" in train.columns:
        checks.append(
            (
                "poids_zero_or_neg",
                pd.to_numeric(train["poids_vehicule"], errors="coerce") <= 0,
                "poids_vehicule <= 0",
            )
        )
    if "cylindre_vehicule" in train.columns:
        checks.append(
            (
                "cylindre_zero_or_neg",
                pd.to_numeric(train["cylindre_vehicule"], errors="coerce") <= 0,
                "cylindre_vehicule <= 0",
            )
        )
    if "age_conducteur1" in train.columns:
        a1 = pd.to_numeric(train["age_conducteur1"], errors="coerce")
        checks.append(
            (
                "age_conducteur1_impossible",
                (a1 < 16) | (a1 > 100),
                "age_conducteur1 out of [16,100]",
            )
        )
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


def compute_outlier_report(
    df: pd.DataFrame, cols: list[str], method: str = "iqr_mad"
) -> pd.DataFrame:
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
        mz = (
            0.6745 * (x - med) / mad_scale
            if np.isfinite(mad_scale)
            else pd.Series(np.nan, index=x.index)
        )
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
                "n_out_mad_z35": int((np.abs(mz) > 3.5).sum())
                if np.isfinite(mad_scale)
                else np.nan,
                "ratio_out_mad_z35": float((np.abs(mz) > 3.5).mean())
                if np.isfinite(mad_scale)
                else np.nan,
                "method": method,
            }
        )
    return pd.DataFrame(rows).sort_values("ratio_out_iqr", ascending=False).reset_index(drop=True)

from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_pricing.data.schema import ID_COLS, INDEX_COL, TARGET_FREQ_COL, TARGET_SEV_COL
from insurance_pricing.features.engineering import add_engineered_features_v2

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
    *,
    postal_as_categorical: bool = True,
) -> pd.DataFrame:
    meta = meta_df.copy()
    if "column" not in meta.columns and "feature" in meta.columns:
        meta = meta.rename(columns={"feature": "column"})
    card = cardinality_df.copy()
    if "column" not in card.columns and "feature" in card.columns:
        card = card.rename(columns={"feature": "column"})
    miss = missing_df.copy()
    if "scope" in miss.columns:
        miss = miss[miss["scope"] == "global"]
    miss_cols = [
        c
        for c in [
            "column",
            "present_train",
            "present_test",
            "missing_rate_train",
            "missing_rate_test",
            "missing_rate_train_effective",
            "missing_rate_test_effective",
        ]
        if c in miss.columns
    ]
    miss = miss[miss_cols].drop_duplicates(subset=["column"])
    merged = (
        meta.merge(miss, on="column", how="left", suffixes=("", "_miss"))
        .merge(card, on="column", how="left", suffixes=("_meta", "_card"))
    )

    rows = []
    for _, r in merged.iterrows():
        col = r["column"]
        cl = str(col).lower()
        role = str(r.get("role_guess", r.get("role_guess_meta", "unknown")))
        dtype = str(r.get("dtype_train_meta", r.get("dtype_train", r.get("dtype", ""))))
        nunique_raw = r.get(
            "nunique_train_meta",
            r.get("nunique_train_card", r.get("nunique_train", np.nan)),
        )
        nunique = float(nunique_raw) if pd.notna(nunique_raw) else np.nan
        miss_rate_raw = r.get("missing_rate_train_effective", r.get("missing_rate_train", np.nan))
        miss_rate = float(miss_rate_raw) if pd.notna(miss_rate_raw) else 0.0
        present_train = bool(int(r.get("present_train", 1))) if pd.notna(r.get("present_train", 1)) else True

        role_is_target = role in {"target_freq", "target_sev"}
        role_is_id = role.startswith("id_")
        role_is_categorical = role == "categorical"
        if postal_as_categorical and ("postal" in cl or cl in {"code_postal", "cp2", "cp3"}):
            role_is_categorical = True

        if role_is_target:
            action = "target_only"
            encoding = "none"
            imputation = "none"
            scaling = "none"
            transform = "none"
            special_handling = "target_column"
        elif role_is_id:
            action = "exclude_feature"
            encoding = "none"
            imputation = "none"
            scaling = "none"
            transform = "none"
            special_handling = "identifier"
        elif not present_train:
            action = "ignore_train_absent"
            encoding = "none"
            imputation = "none"
            scaling = "none"
            transform = "none"
            special_handling = "not_present_in_train"
        elif role_is_categorical or dtype.startswith("object") or dtype == "category":
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
            if "conducteur2" in cl:
                special_handling = "structural_missing_if_no_second_driver + indicator"
            elif "modele" in cl or "marque" in cl:
                special_handling = "rare_grouping + OOD fallback hierarchy"
            elif "postal" in cl or cl in {"code_postal", "cp2", "cp3"}:
                special_handling = "categorical_hierarchy_cp2_cp3_fallback"
            else:
                special_handling = ""
        else:
            action = "use_feature"
            encoding = "numeric"
            imputation = "median + is_missing" if miss_rate > 0 else "none_or_median"
            scaling = "robust_scaler_if_distance_or_linear_model"
            if any(tok in cl for tok in ["prix", "montant", "valeur"]):
                transform = "consider_log1p"
            else:
                transform = "none_or_binning_if_explainability"
            if cl in {"poids_vehicule", "cylindre_vehicule"}:
                special_handling = "zero_technique_to_na + indicator"
            elif cl in {"age_conducteur2", "anciennete_permis2"}:
                special_handling = "conditional_missing_if_no_second_driver"
            else:
                special_handling = ""
        rows.append(
            {
                "column": col,
                "role_guess": role,
                "dtype_train": dtype,
                "nunique_train": nunique,
                "missing_rate_train": miss_rate,
                "action": action,
                "encoding": encoding,
                "imputation": imputation,
                "scaling": scaling,
                "transform": transform,
                "special_handling": special_handling,
                "comment": "TE uniquement en CV cross-fit; IDs exclus; features high-card a regrouper",
            }
        )
    return pd.DataFrame(rows).sort_values(["action", "column"]).reset_index(drop=True)

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


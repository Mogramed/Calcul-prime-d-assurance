from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.insurance_pricing.data.io import build_targets
from src.insurance_pricing.data.schema import (
    DatasetBundle,
    ID_COLS,
    INDEX_COL,
    TARGET_FREQ_COL,
    TARGET_SEV_COL,
)

def _binary_indicator_name(col: str, token: str) -> str:
    token = token.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    return f"is_{col}_{token}"

def _add_binary_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype != "object":
            continue
        uniq = sorted({str(v) for v in out[c].dropna().unique() if str(v).strip()})
        if len(uniq) != 2:
            continue
        low = {v.lower() for v in uniq}
        if low == {"yes", "no"}:
            pos = "yes"
        elif low == {"true", "false"}:
            pos = "true"
        elif low == {"m", "f"}:
            pos = "m"
        else:
            pos = uniq[0]
        out[_binary_indicator_name(c, pos)] = (
            out[c].astype(str).str.lower().eq(str(pos).lower()).astype(int)
        )
    return out

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "conducteur2" in out.columns:
        second_driver = ~out["conducteur2"].astype(str).str.lower().eq("no")
        out["has_second_driver"] = second_driver.astype(int)
        for c in ["age_conducteur2", "anciennete_permis2"]:
            if c in out.columns:
                out.loc[~second_driver, c] = np.nan

    for c in ["poids_vehicule", "cylindre_vehicule"]:
        if c in out.columns:
            out.loc[out[c] == 0, c] = np.nan

    if "code_postal" in out.columns:
        cp = out["code_postal"].astype(str).str.zfill(5)
        out["code_postal"] = cp
        out["cp2"] = cp.str[:2]
        out["cp3"] = cp.str[:3]

    if {"prix_vehicule", "poids_vehicule"}.issubset(out.columns):
        out["prix_par_kg"] = out["prix_vehicule"] / out["poids_vehicule"].replace(0, np.nan)

    if {"din_vehicule", "cylindre_vehicule"}.issubset(out.columns):
        out["din_par_cylindre"] = (
            out["din_vehicule"] / out["cylindre_vehicule"].replace(0, np.nan)
        )

    out = _add_binary_indicators(out)

    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).fillna("NA")

    return out

def prepare_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    drop_identifiers: bool = True,
) -> DatasetBundle:
    train_fe = add_engineered_features(train)
    test_fe = add_engineered_features(test)
    y_freq, y_sev = build_targets(train_fe)

    exclude = {INDEX_COL, TARGET_FREQ_COL, TARGET_SEV_COL}
    cols = [c for c in train_fe.columns if c in test_fe.columns and c not in exclude]
    if drop_identifiers:
        cols = [c for c in cols if c not in ID_COLS]

    X_train = train_fe[cols].copy()
    X_test = test_fe[cols].copy()
    cat_cols = [c for c in cols if X_train[c].dtype == "object"]
    num_cols = [c for c in cols if c not in cat_cols]

    return DatasetBundle(
        train_raw=train,
        test_raw=test,
        X_train=X_train,
        X_test=X_test,
        y_freq=y_freq,
        y_sev=y_sev,
        feature_cols=cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
    )

def _normalize_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].fillna("NA").astype(str)
    return out

def _build_rare_maps(
    train_df: pd.DataFrame, *, cat_cols: Sequence[str], min_count: int = 30
) -> Dict[str, set]:
    maps: Dict[str, set] = {}
    for c in cat_cols:
        if c not in train_df.columns:
            continue
        vc = train_df[c].astype(str).value_counts(dropna=False)
        maps[c] = set(vc[vc < min_count].index.astype(str))
    return maps

def _apply_rare_maps(df: pd.DataFrame, rare_maps: Mapping[str, set]) -> pd.DataFrame:
    out = df.copy()
    for c, rare in rare_maps.items():
        if c not in out.columns:
            continue
        s = out[c].astype(str)
        out[c] = np.where(s.isin(rare), "__RARE__", s)
    return out

def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    out = np.divide(aa, bb, out=np.full_like(aa, np.nan), where=np.isfinite(bb) & (bb != 0))
    return out

def _add_engineered_features_core_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = add_engineered_features(df)

    if {"marque_vehicule", "modele_vehicule"}.issubset(out.columns):
        out["marque_modele"] = (
            out["marque_vehicule"].astype(str) + "__" + out["modele_vehicule"].astype(str)
        )

    if {"din_vehicule", "poids_vehicule"}.issubset(out.columns):
        out["power_weight_ratio"] = _safe_div(out["din_vehicule"], out["poids_vehicule"])

    if "anciennete_vehicule" in out.columns:
        bins = [-np.inf, 2, 5, 9, 14, np.inf]
        labels = ["v_new", "v_recent", "v_mid", "v_old", "v_very_old"]
        out["veh_age_band"] = pd.cut(out["anciennete_vehicule"], bins=bins, labels=labels).astype(str)

    if "power_weight_ratio" in out.columns:
        bins = [-np.inf, 0.04, 0.06, 0.08, 0.12, np.inf]
        labels = ["pw_low", "pw_mid", "pw_high", "pw_perf", "pw_extreme"]
        out["power_weight_band"] = pd.cut(out["power_weight_ratio"], bins=bins, labels=labels).astype(str)

    if {"bonus", "utilisation"}.issubset(out.columns):
        out["bonus_x_usage"] = out["bonus"].astype(float) * (
            out["utilisation"].astype(str).str.len().astype(float)
        )

    if {"age_conducteur1", "anciennete_permis1"}.issubset(out.columns):
        out["age_x_permis"] = out["age_conducteur1"].astype(float) * out[
            "anciennete_permis1"
        ].astype(float)

    if {"prix_vehicule", "type_vehicule"}.issubset(out.columns):
        out["prix_x_type_vehicule"] = (
            out["type_vehicule"].astype(str)
            + "__"
            + pd.cut(
                out["prix_vehicule"].astype(float),
                bins=[-np.inf, 10000, 20000, 35000, np.inf],
                labels=["p_low", "p_mid", "p_hi", "p_lux"],
            ).astype(str)
        )

    out = _normalize_object_cols(out)
    return out

def add_engineered_features_v2(
    train: pd.DataFrame, test: pd.DataFrame, *, rare_min_count: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = _add_engineered_features_core_v2(train)
    te = _add_engineered_features_core_v2(test)

    cat_cols = [c for c in tr.columns if tr[c].dtype == "object" and c in te.columns]
    rare_maps = _build_rare_maps(tr, cat_cols=cat_cols, min_count=rare_min_count)
    tr = _apply_rare_maps(tr, rare_maps)
    te = _apply_rare_maps(te, rare_maps)

    if "code_postal" in tr.columns and "code_postal" in te.columns:
        seen_cp = set(tr["code_postal"].astype(str).unique())
        tr["is_unseen_cp5"] = 0
        te["is_unseen_cp5"] = (~te["code_postal"].astype(str).isin(seen_cp)).astype(int)
    else:
        tr["is_unseen_cp5"] = 0
        te["is_unseen_cp5"] = 0

    if "modele_vehicule" in tr.columns and "modele_vehicule" in te.columns:
        seen_modele = set(tr["modele_vehicule"].astype(str).unique())
        tr["is_unseen_modele"] = 0
        te["is_unseen_modele"] = (~te["modele_vehicule"].astype(str).isin(seen_modele)).astype(int)
    else:
        tr["is_unseen_modele"] = 0
        te["is_unseen_modele"] = 0

    if "marque_modele" in tr.columns:
        vc = tr["marque_modele"].astype(str).value_counts(dropna=False)
        rare_mm = set(vc[vc < 20].index.astype(str))
        seen_mm = set(vc.index.astype(str))
        tr["is_rare_marque_modele"] = tr["marque_modele"].astype(str).isin(rare_mm).astype(int)
        te_mm = te["marque_modele"].astype(str) if "marque_modele" in te.columns else pd.Series("", index=te.index)
        te["is_rare_marque_modele"] = (
            te_mm.isin(rare_mm) | (~te_mm.isin(seen_mm))
        ).astype(int)
    else:
        tr["is_rare_marque_modele"] = 0
        te["is_rare_marque_modele"] = 0
    return tr, te

def _make_bundle(train_fe: pd.DataFrame, test_fe: pd.DataFrame, *, keep_cols: Sequence[str], name: str) -> DatasetBundle:
    y_freq, y_sev = build_targets(train_fe)
    cols = [c for c in keep_cols if c in train_fe.columns and c in test_fe.columns]
    X_train = train_fe[cols].copy()
    X_test = test_fe[cols].copy()
    cat_cols = [c for c in cols if X_train[c].dtype == "object"]
    num_cols = [c for c in cols if c not in cat_cols]
    return DatasetBundle(
        train_raw=train_fe,
        test_raw=test_fe,
        X_train=X_train,
        X_test=X_test,
        y_freq=y_freq,
        y_sev=y_sev,
        feature_cols=cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
    )

def prepare_feature_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    rare_min_count: int = 30,
    drop_identifiers: bool = True,
) -> Dict[str, DatasetBundle]:
    train_fe, test_fe = add_engineered_features_v2(train, test, rare_min_count=rare_min_count)
    exclude = {INDEX_COL, TARGET_FREQ_COL, TARGET_SEV_COL}
    all_cols = [c for c in train_fe.columns if c in test_fe.columns and c not in exclude]
    if drop_identifiers:
        all_cols = [c for c in all_cols if c not in ID_COLS]

    robust_drop = {"code_postal", "modele_vehicule", "marque_modele"}
    compact_drop = robust_drop.union({"marque_vehicule"})

    return {
        "base_v2": _make_bundle(train_fe, test_fe, keep_cols=all_cols, name="base_v2"),
        "robust_v2": _make_bundle(
            train_fe, test_fe, keep_cols=[c for c in all_cols if c not in robust_drop], name="robust_v2"
        ),
        "compact_v2": _make_bundle(
            train_fe, test_fe, keep_cols=[c for c in all_cols if c not in compact_drop], name="compact_v2"
        ),
    }


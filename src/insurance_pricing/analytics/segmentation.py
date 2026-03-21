from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from insurance_pricing.data.schema import TARGET_SEV_COL

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


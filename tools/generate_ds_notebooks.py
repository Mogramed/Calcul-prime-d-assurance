from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip() + "\n",
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip() + "\n",
    }


def _nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_notebook(nb_obj: dict, path: Path) -> None:
    path.write_text(json.dumps(nb_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _common_setup_cell() -> dict:
    return code(
        """
        import sys
        import json
        import warnings
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid")

        ROOT = Path.cwd()
        if not (ROOT / "src").exists():
            ROOT = ROOT.parent
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        from src import v2_pipeline as v2
        from src.ds_analysis_utils import *

        DATA_DIR = ROOT / "data"
        ARTIFACT_DS = ensure_dir(ROOT / DEFAULT_DS_DIR)
        ARTIFACT_V2 = ROOT / v2.DEFAULT_V2_DIR

        SEED = 42
        QUICK_ANALYSIS = True
        FULL_ANALYSIS = False
        SAMPLE_N = 3000
        np.random.seed(SEED)
        """
    )


def _load_data_cell() -> dict:
    return code(
        """
        train_raw, test_raw = load_project_datasets(DATA_DIR)
        print("train:", train_raw.shape, "test:", test_raw.shape)
        display(train_raw.head(2))
        display(test_raw.head(2))
        """
    )


def _comment_framework_md() -> dict:
    return md(
        """
        **Convention de commentaire dans ce notebook (à respecter dans l'analyse)**

        - `Constat` : ce que montrent les chiffres/graphes
        - `Interprétation` : ce que cela signifie (avec prudence)
        - `Décision` : impact pratique sur preprocessing / CV / modèle
        - Terminer chaque bloc par une phrase `Impact modèle attendu`
        """
    )


def _safe_display_v2_artifacts_cell() -> dict:
    return code(
        """
        print("ARTIFACT_V2 exists:", ARTIFACT_V2.exists())
        if ARTIFACT_V2.exists():
            for p in sorted(ARTIFACT_V2.iterdir()):
                if p.is_file():
                    print(f"{p.name:<40} {p.stat().st_size/1024:.1f} KB")
        """
    )


def build_nb_07() -> dict:
    cells: list[dict] = [
        md(
            """
            # 07 - DS Cadrage métier, Qualité des données et Design de validation

            Objectif: montrer la démarche **avant** la modélisation.

            Ce notebook documente:
            - le problème métier (tarification, fréquence × sévérité),
            - la compréhension des colonnes,
            - la qualité des données (NA, incohérences, outliers),
            - la dérive train/test (drift/OOD),
            - la justification des splits de validation anti-fuite.
            """
        ),
        md(
            """
            ## Cadrage métier (avant de coder)

            **Constat**
            - La prime doit être compétitive et rentable.
            - Une grosse erreur de tarification est très coûteuse.

            **Interprétation**
            - Une décomposition actuarielle `fréquence × sévérité` est naturelle et explicable.
            - La métrique RMSE est cohérente car elle pénalise fortement les grosses erreurs.

            **Décision**
            - On documente la démarche two-part et on relie chaque choix de preprocessing/CV à ce cadre.
            """
        ),
        _comment_framework_md(),
        _common_setup_cell(),
        _load_data_cell(),
        code(
            """
            # Data contract + dictionnaire de données
            data_dict = build_data_dictionary(train_raw, test_raw)
            col_types = classify_columns(train_raw, test_raw)
            leakage_risk = detect_leakage_risk_columns(train_raw)

            display(data_dict.head(20))
            display(col_types[["column","role_guess","dtype_train","nunique_train","missing_rate_train","high_cardinality_train"]].head(20))
            display(leakage_risk)
            """
        ),
        md(
            """
            **À commenter (cellule précédente)**

            - `Constat`: quelles colonnes sont des identifiants (`index`, `id_client`, `id_vehicule`, `id_contrat`) ?
            - `Interprétation`: pourquoi ces colonnes créent un risque de fuite ou d'apprentissage non généralisable ?
            - `Décision`: colonnes à exclure comme features brutes.
            - **Impact modèle attendu**: baisse du risque d'overfitting et meilleure robustesse privée.
            """
        ),
        code(
            """
            # Cibles fréquence / sévérité
            y_sev = train_raw[v2.TARGET_SEV_COL].astype(float)
            y_freq = (y_sev > 0).astype(int)
            y_pos = y_sev[y_sev > 0]

            target_summary = pd.DataFrame([
                {"metric": "n_train", "value": len(train_raw)},
                {"metric": "claim_rate", "value": float(y_freq.mean())},
                {"metric": "n_claims_pos", "value": int(y_freq.sum())},
                {"metric": "sev_mean_pos", "value": float(y_pos.mean()) if len(y_pos) else np.nan},
                {"metric": "sev_median_pos", "value": float(y_pos.median()) if len(y_pos) else np.nan},
                {"metric": "sev_q95_pos", "value": float(y_pos.quantile(0.95)) if len(y_pos) else np.nan},
                {"metric": "sev_q99_pos", "value": float(y_pos.quantile(0.99)) if len(y_pos) else np.nan},
                {"metric": "sev_max_pos", "value": float(y_pos.max()) if len(y_pos) else np.nan},
            ])
            display(target_summary)

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            sns.histplot(y_sev, bins=60, ax=axes[0])
            axes[0].set_title("Montant sinistre (tous)")
            sns.histplot(np.log1p(y_pos), bins=60, ax=axes[1])
            axes[1].set_title("log1p(montant) sur positifs")
            sns.boxplot(x=y_pos, ax=axes[2], orient="h")
            axes[2].set_title("Boxplot positifs")
            plt.tight_layout()
            """
        ),
        md(
            """
            **À commenter**
            - `Constat`: rareté des sinistres + forte asymétrie / queue lourde de la sévérité.
            - `Interprétation`: pourquoi `log1p` et/ou Tweedie sont des candidats naturels.
            - `Décision`: conserver l'approche two-part et soigner la couverture de queue.
            """
        ),
        code(
            """
            # Missingness global + par sous-population
            missing_report = compute_missingness_report(
                train_raw, test_raw, group_cols=["utilisation", "type_vehicule", "conducteur2"]
            )
            missing_global = missing_report[missing_report["scope"] == "global"].copy()
            display(missing_global.sort_values("missing_rate_train", ascending=False).head(20))

            pivot_missing = (
                missing_global[["column","missing_rate_train","missing_rate_test"]]
                .set_index("column")
                .sort_values("missing_rate_train", ascending=False)
                .head(20)
            )
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_missing, annot=True, fmt=".2f", cmap="Blues")
            plt.title("Top missingness train/test")
            plt.tight_layout()

            missing_group = missing_report[missing_report["scope"] == "by_group"].copy()
            display(missing_group.sort_values("missing_rate_train", ascending=False).head(20))
            """
        ),
        code(
            """
            # Règles métier / incohérences / petits sinistres / zéros techniques
            rule_violations = compute_rule_violations(train_raw)
            display(rule_violations)

            zero_tech = pd.DataFrame([
                {
                    "column": c,
                    "zero_rate_train": float((pd.to_numeric(train_raw[c], errors='coerce') == 0).mean()),
                    "zero_rate_test": float((pd.to_numeric(test_raw[c], errors='coerce') == 0).mean()) if c in test_raw.columns else np.nan,
                }
                for c in ["poids_vehicule", "cylindre_vehicule"]
                if c in train_raw.columns
            ])
            display(zero_tech)
            """
        ),
        md(
            """
            ## Discussion attendue: sinistres négatifs, petits sinistres, extrêmes

            **Constat**
            - Le dataset peut contenir des valeurs atypiques (négatives, très petites, extrêmes).

            **Interprétation**
            - En assurance, ces cas peuvent refléter des conventions de gestion, des remboursements, ou du bruit.

            **Décision (à défendre en soutenance)**
            - Tagger systématiquement ces cas.
            - Comparer plusieurs stratégies (laisser / winsoriser pour l'apprentissage / traiter la queue séparément).
            - Ne jamais masquer une décision de nettoyage sans trace analytique.
            """
        ),
        code(
            """
            # Doublons exacts et quasi-doublons (hors cible)
            n_dup_exact = int(train_raw.duplicated().sum())
            feature_cols_wo_target = [c for c in train_raw.columns if c not in [v2.TARGET_FREQ_COL, v2.TARGET_SEV_COL]]
            n_dup_wo_target = int(train_raw.duplicated(subset=feature_cols_wo_target).sum())
            dup_report = pd.DataFrame([
                {"check": "duplicate_rows_exact", "count": n_dup_exact},
                {"check": "duplicate_rows_wo_target", "count": n_dup_wo_target},
            ])
            display(dup_report)
            """
        ),
        code(
            """
            # Drift train/test numérique + catégoriel + cardinalité
            num_cols = [c for c in train_raw.columns if str(train_raw[c].dtype).startswith(("int","float")) and c in test_raw.columns]
            cat_cols = [c for c in train_raw.columns if train_raw[c].dtype == "object" and c in test_raw.columns]

            drift_num = compute_drift_numeric_ks_psi(train_raw, test_raw, num_cols=num_cols, bins=10)
            drift_cat = compute_drift_categorical_chi2(train_raw, test_raw, cat_cols=cat_cols, top_k=50)
            cardinality = build_cardinality_report(train_raw, test_raw)
            ood = v2.compute_ood_diagnostics(train_raw.astype(object), test_raw.astype(object))

            display(drift_num.head(20))
            display(drift_cat.head(20))
            display(cardinality.head(20))
            display(ood.sort_values("unseen_test_levels", ascending=False).head(20))
            """
        ),
        md(
            """
            **À commenter (drift/OOD)**

            - `Constat`: OOD fort sur granularité fine (ex. `code_postal`, `modele_vehicule`) vs plus faible sur agrégats (`cp2/cp3`).
            - `Interprétation`: certaines colonnes demandent hiérarchies/fallbacks plutôt qu'un traitement brut.
            - `Décision`: prioriser robustesse OOD (rare grouping, TE cross-fit, hiérarchies géographiques).
            """
        ),
        code(
            """
            # Justification du split / validation anti-fuite
            splits = v2.build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")
            split_rows = []
            for split_name, folds in splits.items():
                v2.validate_folds_disjoint(
                    folds,
                    check_full_coverage=(split_name in {"secondary_group", "aux_blocked5"}),
                    n_rows=len(train_raw),
                )
                if split_name == "secondary_group":
                    v2.validate_group_disjoint(folds, train_raw["id_client"])
                for fold_id, (tr_idx, va_idx) in folds.items():
                    split_rows.append(
                        {
                            "split": split_name,
                            "fold_id": int(fold_id),
                            "n_train": int(len(tr_idx)),
                            "n_valid": int(len(va_idx)),
                        }
                    )
            split_summary = pd.DataFrame(split_rows)
            display(split_summary)
            """
        ),
        md(
            """
            ## OOF / GroupKFold / anti-fuite (rappel pédagogique)

            - **GroupKFold**: toutes les lignes d'un même `id_client` restent dans le même fold.
            - **OOF**: chaque ligne est prédite par un modèle qui ne l'a jamais vue.
            - **Pourquoi c'est crucial ici**: sinon, on mesure un score optimiste et on augmente le risque de shake-up public/privé.
            """
        ),
        code(
            """
            # Exports d'analyse (Notebook 1)
            tables = {
                "data_dictionary": data_dict,
                "column_typing_report": col_types,
                "leakage_risk_report": leakage_risk,
                "missingness_report": missing_report,
                "rule_violations_report": rule_violations,
                "drift_numeric_ks_psi": drift_num,
                "drift_categorical_chi2": drift_cat,
                "cardinality_report": cardinality,
            }
            export_analysis_tables(tables, ARTIFACT_DS)
            print("Exports DS notebook 1 ->", ARTIFACT_DS)
            """
        ),
        md(
            """
            ## Synthèse Notebook 1 (à compléter)

            **Risques identifiés**
            - [ ] fuite via identifiants
            - [ ] OOD catégoriel sur granularité fine
            - [ ] extrêmes de sévérité / queue lourde
            - [ ] incohérences et zéros techniques

            **Mitigations retenues**
            - [ ] split multi-schémas (`primary_time`, `secondary_group`, `aux_blocked5`)
            - [ ] hiérarchies/fallbacks OOD
            - [ ] traitements robustes de sévérité (log/Tweedie/tail mapping)
            - [ ] traçabilité des règles de nettoyage
            """
        ),
    ]
    return _nb(cells)


def build_nb_08() -> dict:
    cells: list[dict] = [
        md(
            """
            # 08 - DS EDA, Segmentation, Préprocessing et Feature Engineering

            Objectif: justifier les transformations et les features à partir d'observations statistiques et métier.
            """
        ),
        _comment_framework_md(),
        _common_setup_cell(),
        _load_data_cell(),
        code(
            """
            feature_sets = v2.prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
            bundle = feature_sets["base_v2"]
            print("feature sets:", {k: (len(v.feature_cols), len(v.cat_cols), len(v.num_cols)) for k, v in feature_sets.items()})
            """
        ),
        code(
            """
            # Typage enrichi + cardinalité
            data_dict = build_data_dictionary(train_raw, test_raw)
            col_types = classify_columns(train_raw, test_raw)
            cardinality = build_cardinality_report(train_raw, test_raw)
            display(col_types)
            display(cardinality.head(25))
            """
        ),
        code(
            """
            # Statistiques univariées numériques
            num_cols = [c for c in train_raw.columns if str(train_raw[c].dtype).startswith(("int","float"))]
            num_cols_eda = [c for c in num_cols if c not in [v2.TARGET_FREQ_COL]]
            rows = []
            for c in num_cols_eda:
                s = pd.to_numeric(train_raw[c], errors="coerce").replace([np.inf,-np.inf], np.nan).dropna()
                if len(s) == 0:
                    continue
                rows.append({
                    "column": c,
                    "n": int(len(s)),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "std": float(s.std(ddof=0)),
                    "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
                    "mad": float(np.median(np.abs(s - s.median()))),
                    "skew": float(stats.skew(s)) if len(s) > 2 else np.nan,
                    "kurtosis": float(stats.kurtosis(s)) if len(s) > 3 else np.nan,
                    "p01": float(s.quantile(0.01)),
                    "p05": float(s.quantile(0.05)),
                    "p50": float(s.quantile(0.50)),
                    "p95": float(s.quantile(0.95)),
                    "p99": float(s.quantile(0.99)),
                })
            uni_num = pd.DataFrame(rows).sort_values("skew", ascending=False)
            display(uni_num.head(20))
            """
        ),
        code(
            """
            # Visualisations univariées (échantillon de variables numériques)
            plot_num_cols = [c for c in ["bonus","age_conducteur1","anciennete_vehicule","prix_vehicule","poids_vehicule","din_vehicule",v2.TARGET_SEV_COL] if c in train_raw.columns]
            ncols = 3
            nrows = int(np.ceil(len(plot_num_cols) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
            axes = np.array(axes).reshape(-1)
            for ax, c in zip(axes, plot_num_cols):
                s = pd.to_numeric(train_raw[c], errors='coerce')
                sns.histplot(s, bins=50, ax=ax)
                ax.set_title(f"Histogramme - {c}")
            for ax in axes[len(plot_num_cols):]:
                ax.axis("off")
            plt.tight_layout()
            """
        ),
        md(
            """
            **À commenter (univarié)**
            - `Constat`: asymétrie, valeurs extrêmes, distributions tronquées.
            - `Interprétation`: impact sur choix de métriques et transformations.
            - `Décision`: `log1p`, winsorisation d'apprentissage, modèles robustes.
            """
        ),
        code(
            """
            # Statistiques catégorielles + rareté
            cat_cols = [c for c in train_raw.columns if train_raw[c].dtype == "object"]
            cat_rows = []
            for c in cat_cols:
                vc = train_raw[c].astype(str).value_counts(dropna=False)
                cat_rows.append({
                    "column": c,
                    "nunique": int(vc.size),
                    "top1": str(vc.index[0]) if len(vc) else "",
                    "top1_ratio": float(vc.iloc[0] / max(len(train_raw), 1)) if len(vc) else np.nan,
                    "rare_levels_lt10": int((vc < 10).sum()),
                    "rare_levels_ratio_lt10": float((vc < 10).sum() / max(len(vc), 1)) if len(vc) else np.nan,
                })
            cat_stats = pd.DataFrame(cat_rows).sort_values("nunique", ascending=False)
            display(cat_stats)
            """
        ),
        code(
            """
            # Barplots catégorielles (fréquences)
            plot_cat_cols = [c for c in ["type_contrat","utilisation","freq_paiement","paiement","conducteur2","essence_vehicule","type_vehicule"] if c in train_raw.columns]
            fig, axes = plt.subplots(len(plot_cat_cols), 1, figsize=(12, 3 * len(plot_cat_cols)))
            if len(plot_cat_cols) == 1:
                axes = [axes]
            for ax, c in zip(axes, plot_cat_cols):
                vc = train_raw[c].astype(str).value_counts(dropna=False).head(15)
                sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax)
                ax.set_title(f"Top modalités - {c}")
            plt.tight_layout()
            """
        ),
        code(
            """
            # Analyse bivariée num-num (Pearson / Spearman) sur sous-ensemble
            num_subset = [c for c in ["bonus","age_conducteur1","anciennete_permis1","anciennete_vehicule","din_vehicule","prix_vehicule","poids_vehicule"] if c in train_raw.columns]
            corr_pearson = train_raw[num_subset].corr(method="pearson")
            corr_spearman = train_raw[num_subset].corr(method="spearman")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
            axes[0].set_title("Corrélation Pearson")
            sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
            axes[1].set_title("Corrélation Spearman")
            plt.tight_layout()
            """
        ),
        code(
            """
            # Analyse cat-cat (Cramer's V)
            cat_subset = [c for c in ["type_contrat","utilisation","freq_paiement","paiement","essence_vehicule","type_vehicule","marque_vehicule"] if c in train_raw.columns]
            cramers_v = compute_cramers_v_table(train_raw, cat_cols=cat_subset, max_cols=12)
            display(cramers_v.head(20))
            """
        ),
        code(
            """
            # Analyse cible par segments (fréquence / sévérité / pure premium)
            train_fe = v2.add_engineered_features(train_raw.copy())
            segment_cols = [c for c in ["utilisation","type_contrat","cp2","cp3","marque_vehicule","type_vehicule"] if c in train_fe.columns]
            seg_tables = compute_segment_target_tables(train_fe, segment_cols=segment_cols)
            for name, df_seg in seg_tables.items():
                print(f"Segment: {name}")
                display(df_seg.head(10))
            """
        ),
        md(
            """
            ## Lecture métier des segments (à rédiger)

            Pour chaque segment clé (`utilisation`, `type_contrat`, `cp2`, `marque_vehicule`, etc.):
            - `Constat`: segments avec forte fréquence / forte sévérité / pure premium élevé
            - `Interprétation`: hypothèses métier plausibles
            - `Décision`: variables/interactions à privilégier ou hiérarchies de fallback
            """
        ),
        code(
            """
            # Réduction de dimension proxy FAMD/MCA (sans dépendance externe)
            sample_df = sample_for_exploration(train_fe, n=SAMPLE_N, seed=SEED, stratify_col="type_contrat" if "type_contrat" in train_fe.columns else None)
            sample_num = [c for c in sample_df.columns if str(sample_df[c].dtype).startswith(("int","float")) and c not in [v2.TARGET_FREQ_COL, v2.TARGET_SEV_COL]]
            sample_cat = [c for c in sample_df.columns if sample_df[c].dtype == "object"]
            emb = fit_mixed_embedding_proxy(sample_df, num_cols=sample_num, cat_cols=sample_cat, n_components=2)
            emb_plot = emb.copy()
            emb_plot["claim"] = (sample_df[v2.TARGET_SEV_COL].astype(float) > 0).astype(int).values
            emb_plot["utilisation"] = sample_df["utilisation"].astype(str).values if "utilisation" in sample_df.columns else "NA"
            emb_plot["type_contrat"] = sample_df["type_contrat"].astype(str).values if "type_contrat" in sample_df.columns else "NA"

            plt.figure(figsize=(7,5))
            sns.scatterplot(data=emb_plot, x="comp_1", y="comp_2", hue="claim", alpha=0.6, s=25)
            plt.title("Embedding proxy (SVD/PCA) coloré par sinistre")
            plt.tight_layout()
            """
        ),
        code(
            """
            # Clustering exploratoire (KMeans) sur représentation mixte encodée
            from src.ds_analysis_utils import fit_kmeans_exploration

            sample_num = [c for c in sample_df.columns if str(sample_df[c].dtype).startswith(("int","float")) and c not in [v2.TARGET_FREQ_COL, v2.TARGET_SEV_COL]]
            sample_cat = [c for c in sample_df.columns if sample_df[c].dtype == "object"]
            kmeans_df = fit_kmeans_exploration(sample_df, num_cols=sample_num, cat_cols=sample_cat, n_clusters=4, random_state=SEED)
            sample_cluster = sample_df.join(kmeans_df)
            cluster_summary = sample_cluster.groupby("cluster").agg(
                n=(v2.TARGET_SEV_COL, "size"),
                claim_rate=(v2.TARGET_SEV_COL, lambda s: float((pd.to_numeric(s, errors='coerce') > 0).mean())),
                pure_premium_obs=(v2.TARGET_SEV_COL, lambda s: float(pd.to_numeric(s, errors='coerce').fillna(0).mean())),
            ).reset_index()
            display(cluster_summary)
            """
        ),
        code(
            """
            # Gower-like (custom) + CAH sur petit échantillon (coût O(n²))
            n_gower = 300 if QUICK_ANALYSIS else 800
            gower_sample = sample_for_exploration(sample_df, n=n_gower, seed=SEED, stratify_col=None)
            g_num = [c for c in gower_sample.columns if str(gower_sample[c].dtype).startswith(("int","float")) and c not in [v2.TARGET_FREQ_COL, v2.TARGET_SEV_COL]]
            g_cat = [c for c in gower_sample.columns if gower_sample[c].dtype == "object"]
            D = compute_gower_like_distance_sample(gower_sample, num_cols=g_num, cat_cols=g_cat)
            print("distance matrix shape:", D.shape, "mean dist:", float(np.mean(D)))
            link = compute_linkage_from_distance(D, method="average")
            print("linkage shape:", link.shape)
            """
        ),
        md(
            """
            **À commenter (segmentation / mining)**
            - `Constat`: structures/amas potentiels dans les données mixtes.
            - `Interprétation`: exploration descriptive, pas segmentation de production.
            - `Décision`: réutiliser les patterns pour enrichir les features/segments métier.
            """
        ),
        code(
            """
            # Mapping preprocessing + recommandations
            missing_report = compute_missingness_report(train_raw, test_raw, group_cols=None)
            prep_reco = compute_preprocessing_recommendations(data_dict, cardinality, missing_report)
            display(prep_reco.head(50))
            """
        ),
        code(
            """
            # Catalogue de feature engineering (features existantes + blocs)
            train_fe_v2, test_fe_v2 = v2.add_engineered_features_v2(train_raw, test_raw, rare_min_count=30)
            fe_catalog = build_feature_engineering_catalog(train_fe_v2)
            display(fe_catalog.head(100))
            """
        ),
        code(
            """
            # Ablations via artefacts V2 (si disponibles)
            fs_cmp_path = ARTIFACT_V2 / "feature_set_comparison_v2.csv"
            if fs_cmp_path.exists():
                fs_cmp = pd.read_csv(fs_cmp_path)
                display(fs_cmp.sort_values(["rmse_prime","q99_ratio_pos"], ascending=[True, False]).head(30))
            else:
                print("Artifact absent:", fs_cmp_path)
            """
        ),
        code(
            """
            # Exports d'analyse (Notebook 2)
            segment_exports = {f"segment_target_{k}": v for k, v in seg_tables.items()}
            tables = {
                "preprocessing_recommendations": prep_reco,
                "feature_engineering_catalog": fe_catalog,
                "cardinality_report": cardinality,
                **segment_exports,
            }
            export_analysis_tables(tables, ARTIFACT_DS)
            print("Exports DS notebook 2 ->", ARTIFACT_DS)
            """
        ),
        md(
            """
            ## Synthèse Notebook 2 (à compléter)

            - Features / transformations retenues:
              - [ ] ...
            - Features / transformations rejetées (et pourquoi):
              - [ ] ...
            - Prochaines validations côté modèle:
              - [ ] queue de sévérité
              - [ ] calibration fréquence
              - [ ] biais par segment
            """
        ),
    ]
    return _nb(cells)


def build_nb_09() -> dict:
    cells: list[dict] = [
        md(
            """
            # 09 - DS Diagnostics Modèle, Explicabilité, Robustesse et Storytelling

            Objectif: comprendre **pourquoi** le modèle marche (ou échoue), pas seulement mesurer un score.
            """
        ),
        _comment_framework_md(),
        _common_setup_cell(),
        _safe_display_v2_artifacts_cell(),
        code(
            """
            # Chargement des artefacts V2 (avec fallback)
            paths = {
                "run_registry": ARTIFACT_V2 / "run_registry_v2.csv",
                "oof_predictions": ARTIFACT_V2 / "oof_predictions_v2.parquet",
                "selected_models": ARTIFACT_V2 / "selected_models_v2.csv",
                "selection_report": ARTIFACT_V2 / "selection_report_v2.csv",
                "ensemble_weights": ARTIFACT_V2 / "ensemble_weights_v2.json",
                "submission_audit": ARTIFACT_V2 / "submission_audit_v2.json",
                "pred_distribution_audit": ARTIFACT_V2 / "pred_distribution_audit_v2.csv",
            }

            run_df = pd.read_csv(paths["run_registry"]) if paths["run_registry"].exists() else pd.DataFrame()
            oof_df = pd.read_parquet(paths["oof_predictions"]) if paths["oof_predictions"].exists() else pd.DataFrame()
            selected_df = pd.read_csv(paths["selected_models"]) if paths["selected_models"].exists() else pd.DataFrame()
            selection_report_df = pd.read_csv(paths["selection_report"]) if paths["selection_report"].exists() else pd.DataFrame()
            ens_meta = json.loads(paths["ensemble_weights"].read_text(encoding="utf-8")) if paths["ensemble_weights"].exists() else {}
            submission_audit = json.loads(paths["submission_audit"].read_text(encoding="utf-8")) if paths["submission_audit"].exists() else {}
            pred_dist_df = pd.read_csv(paths["pred_distribution_audit"]) if paths["pred_distribution_audit"].exists() else pd.DataFrame()

            print("run_df:", run_df.shape, "oof_df:", oof_df.shape, "selected_df:", selected_df.shape)
            display(run_df.head(3))
            display(selected_df.head(3))
            """
        ),
        md(
            """
            ## Rappel du protocole (à présenter)

            - **Split primaire**: `primary_time` (référence de classement local)
            - **Split secondaire**: `secondary_group` sur `id_client` (anti-fuite)
            - **Split auxiliaire**: `aux_blocked5` (stress test)
            - **OOF**: prédictions sur validation uniquement, modèle n'ayant pas vu les lignes
            - **Score Kaggle**: RMSE sur la prime prédite (coût réel vs prime)

            Définitions utiles:
            - `AUC`, `Gini`, `Brier`, `R²`, `RMSE` (rappels à expliquer à l'oral)
            """
        ),
        code(
            """
            # Baselines indispensables (si OOF dispo)
            if not oof_df.empty:
                # on prend un split/runs OOF disponible pour obtenir y_true
                d0 = oof_df[(oof_df["is_test"] == 0) & oof_df["y_sev"].notna()].copy()
                if len(d0):
                    y_true = d0.groupby("row_idx")["y_sev"].first().sort_index().to_numpy()
                    mean_pred = np.full_like(y_true, float(np.mean(y_true)), dtype=float)
                    zero_pred = np.zeros_like(y_true, dtype=float)
                    pure_const = np.full_like(y_true, float(np.mean(y_true)), dtype=float)
                    baseline_df = pd.DataFrame([
                        {"baseline": "mean_prime", "rmse": _rmse(y_true, mean_pred)},
                        {"baseline": "all_zero", "rmse": _rmse(y_true, zero_pred)},
                        {"baseline": "pure_premium_constant", "rmse": _rmse(y_true, pure_const)},
                    ])
                    display(baseline_df)
                else:
                    print("OOF vide -> impossible de recalculer les baselines.")
            else:
                print("Artifacts V2 absents -> section baseline en mode descriptif.")
            """
        ),
        code(
            """
            # Comparatif moteurs / familles / splits
            if not run_df.empty:
                rr = run_df.copy()
                rr = rr[rr["level"] == "run"] if "level" in rr.columns else rr
                cols = [c for c in ["feature_set","engine","family","severity_mode","split","rmse_prime","auc_freq","brier_freq","rmse_sev_pos","q99_ratio_pos","distribution_collapse_flag","tail_dispersion_flag"] if c in rr.columns]
                display(rr[cols].sort_values(["split","rmse_prime"]).head(40))

                summary = rr.pivot_table(
                    index=["engine","family","severity_mode"],
                    values=[c for c in ["rmse_prime","q99_ratio_pos","distribution_collapse_flag"] if c in rr.columns],
                    aggfunc="mean",
                ).sort_values("rmse_prime")
                display(summary.head(20))
            """
        ),
        code(
            """
            # Sélection finale et raisons de rejet / acceptation
            if not selection_report_df.empty:
                cols = [c for c in ["run_id","rank","selection_score","accepted","decision_reason","rmse_primary_time","rmse_secondary_group","rmse_aux_blocked5","q99_primary_time","distribution_collapse_flag","tail_dispersion_flag"] if c in selection_report_df.columns]
                display(selection_report_df[cols].head(30))
            else:
                print("selection_report_v2.csv absent.")
            """
        ),
        code(
            """
            # Choix du run de diagnostic principal
            if not selected_df.empty and "run_id" in selected_df.columns:
                run_id_main = str(selected_df.iloc[0]["run_id"])
            elif not run_df.empty:
                tmp = run_df.copy()
                if "level" in tmp.columns:
                    tmp = tmp[tmp["level"] == "run"]
                tmp = tmp.sort_values("rmse_prime")
                run_id_main = str(tmp.iloc[0]["run_id"]) if len(tmp) else None
            else:
                run_id_main = None
            print("run_id_main:", run_id_main)
            """
        ),
        code(
            """
            # Diagnostics OOF détaillés (métriques, calibration, déciles, résidus)
            diag_tables = {}
            if run_id_main is not None and not oof_df.empty:
                diag_tables = compute_oof_model_diagnostics(oof_df, run_id=run_id_main, split="primary_time")
                for name, df_ in diag_tables.items():
                    print(f"[{name}] ->", getattr(df_, "shape", None))
                    if isinstance(df_, pd.DataFrame) and len(df_):
                        display(df_.head(20))
            else:
                print("OOF ou run_id indisponible.")
            """
        ),
        code(
            """
            # Métriques fréquence, sévérité, prime (mise en forme soutenance)
            if diag_tables:
                metrics_df = diag_tables.get("metrics", pd.DataFrame())
                if not metrics_df.empty:
                    display(metrics_df)
                    if "auc_freq" in metrics_df.columns:
                        metrics_df = metrics_df.copy()
                        metrics_df["gini_freq_check"] = 2 * metrics_df["auc_freq"] - 1
                        display(metrics_df[["auc_freq","gini_freq","gini_freq_check","brier_freq","logloss_freq","pr_auc_freq","rmse_prime","mae_prime","r2_prime","rmse_sev_pos","q99_ratio_pos"]])
            """
        ),
        code(
            """
            # Distribution des prédictions et audit de collapse de queue
            if not pred_dist_df.empty:
                display(pred_dist_df.sort_values(["sample","pred_q99"], ascending=[True, False]).head(30))
            elif diag_tables and "distribution" in diag_tables and len(diag_tables["distribution"]):
                display(diag_tables["distribution"])
            else:
                print("pred_distribution_audit_v2.csv absent.")
            """
        ),
        md(
            """
            **À commenter (distribution prédite)**
            - `Constat`: comparer `q90/q99/max` OOF vs test, détecter écrasement de queue.
            - `Interprétation`: une distribution trop compressée peut améliorer artificiellement la stabilité locale mais dégrader Kaggle.
            - `Décision`: garder des garde-fous distributionnels avant soumission.
            """
        ),
        code(
            """
            # Calibration fréquence (reliability-like)
            cal_df = diag_tables.get("calibration_freq", pd.DataFrame()) if diag_tables else pd.DataFrame()
            if not cal_df.empty:
                display(cal_df)
                plt.figure(figsize=(6, 6))
                plt.plot([0,1],[0,1], '--', color='gray')
                plt.plot(cal_df["p_mean"], cal_df["y_rate"], marker='o')
                plt.xlabel("Proba prédite moyenne (bin)")
                plt.ylabel("Taux observé")
                plt.title("Calibration fréquence (OOF)")
                plt.tight_layout()
            else:
                print("Calibration table indisponible.")
            """
        ),
        code(
            """
            # Diagnostics résiduels prime
            res_df = diag_tables.get("residuals", pd.DataFrame()) if diag_tables else pd.DataFrame()
            if not res_df.empty:
                fig, axes = plt.subplots(1, 2, figsize=(14,5))
                sns.scatterplot(data=res_df.sample(min(len(res_df), 5000), random_state=SEED), x="pred_prime", y="residual", alpha=0.4, s=15, ax=axes[0])
                axes[0].axhline(0, ls="--", color="black")
                axes[0].set_title("Résidus vs prime prédite")
                sns.scatterplot(data=res_df.sample(min(len(res_df), 5000), random_state=SEED), x="y_true", y="abs_error", alpha=0.4, s=15, ax=axes[1])
                axes[1].set_title("Erreur absolue vs coût réel")
                plt.tight_layout()
            """
        ),
        code(
            """
            # Error analysis par déciles (réel et prédit)
            if diag_tables:
                err_true = diag_tables.get("error_by_decile_true", pd.DataFrame())
                err_pred = diag_tables.get("error_by_decile_pred", pd.DataFrame())
                if not err_true.empty:
                    display(err_true)
                if not err_pred.empty:
                    display(err_pred)
            """
        ),
        code(
            """
            # Error analysis segmentaire (merge OOF + features train si possible)
            if run_id_main is not None and not oof_df.empty:
                train_raw, test_raw = load_project_datasets(DATA_DIR)
                d = oof_df[(oof_df["is_test"] == 0) & (oof_df["split"] == "primary_time") & (oof_df["run_id"].astype(str) == str(run_id_main))].copy()
                d = d[["row_idx","y_sev","pred_prime"]].dropna()
                train_aug = v2.add_engineered_features(train_raw.copy()).reset_index(drop=True)
                merged = d.merge(train_aug.reset_index().rename(columns={"index":"row_idx"}), on="row_idx", how="left")
                merged["abs_error"] = (merged["pred_prime"] - merged["y_sev"]).abs()
                seg_cols = [c for c in ["utilisation","type_contrat","cp2","cp3","marque_vehicule","type_vehicule"] if c in merged.columns]
                seg_err_tables = {}
                for c in seg_cols:
                    g = merged.groupby(c).agg(
                        n=("y_sev","size"),
                        y_mean=("y_sev","mean"),
                        pred_mean=("pred_prime","mean"),
                        bias=("pred_prime", lambda s, y=merged.loc[s.index, 'y_sev']: float(np.mean(s.to_numpy() - y.to_numpy()))),
                        mae=("abs_error","mean"),
                    ).reset_index()
                    seg_err_tables[c] = g[g["n"] >= 100].sort_values("mae", ascending=False)
                    print(c)
                    display(seg_err_tables[c].head(10))
            """
        ),
        code(
            """
            # Extrêmes (top 1% coûts réels et grosses erreurs)
            if diag_tables and "residuals" in diag_tables and len(diag_tables["residuals"]):
                res_df = diag_tables["residuals"].copy()
                q99_true = res_df["y_true"].quantile(0.99)
                extreme_true = res_df[res_df["y_true"] >= q99_true].sort_values("y_true", ascending=False).head(20)
                extreme_err = res_df.sort_values("abs_error", ascending=False).head(20)
                print("Top 1% coûts réels")
                display(extreme_true)
                print("Plus grosses erreurs absolues")
                display(extreme_err)
            """
        ),
        code(
            """
            # Feature importance fallback (sans SHAP) : permutation importance sur modèle proxy
            try:
                from sklearn.ensemble import RandomForestRegressor
                train_raw, _ = load_project_datasets(DATA_DIR)
                bundles = v2.prepare_feature_sets(train_raw, test_raw if 'test_raw' in globals() else load_project_datasets(DATA_DIR)[1], rare_min_count=30, drop_identifiers=True)
                b = bundles["robust_v2"]
                X = b.X_train.copy()
                y = b.y_sev.to_numpy(dtype=float)
                n_fit = 6000 if QUICK_ANALYSIS else min(len(X), 20000)
                idx = np.random.RandomState(SEED).choice(len(X), size=n_fit, replace=False)
                Xs = X.iloc[idx].copy()
                ys = y[idx]
                num_cols = [c for c in Xs.columns if c in b.num_cols]
                cat_cols = [c for c in Xs.columns if c in b.cat_cols]
                Xmat, feat_names = _prepare_mixed_matrix(Xs, num_cols=num_cols, cat_cols=cat_cols)
                rf = RandomForestRegressor(n_estimators=80 if QUICK_ANALYSIS else 150, random_state=SEED, n_jobs=-1, max_depth=10)
                rf.fit(Xmat, ys)
                pi = permutation_importance(rf, Xmat, ys, n_repeats=3 if QUICK_ANALYSIS else 5, random_state=SEED, n_jobs=1)
                pi_df = pd.DataFrame({"feature": feat_names, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
                display(pi_df.head(30))
            except Exception as e:
                print("Permutation importance fallback indisponible:", type(e).__name__, e)
            """
        ),
        code(
            """
            # PDP/ICE (optionnel, fallback safe)
            try:
                from sklearn.inspection import PartialDependenceDisplay
                # Si la cellule précédente a créé rf/Xmat/feat_names
                if "rf" in globals() and "Xmat" in globals() and "feat_names" in globals():
                    top_idx = list(np.argsort(-np.asarray(pi.importances_mean))[:2]) if "pi" in globals() else [0, 1]
                    fig, ax = plt.subplots(figsize=(10,4))
                    PartialDependenceDisplay.from_estimator(rf, Xmat, features=top_idx, feature_names=feat_names, ax=ax)
                    plt.tight_layout()
                else:
                    print("Modèle proxy indisponible.")
            except Exception as e:
                print("PDP/ICE non disponible:", type(e).__name__, e)
            """
        ),
        md(
            """
            ## SHAP (section optionnelle)

            - Si `shap` est installé: produire une analyse globale + locale sur un sous-échantillon.
            - Sinon: conserver permutation importance + PDP/ICE comme fallback.

            **Décision pratique**: dans ce repo, `shap` n'est pas une dépendance obligatoire.
            """
        ),
        code(
            """
            # Stabilité des features importantes selon folds / seeds (proxy via runs V2 + catégories)
            if not run_df.empty:
                cols = [c for c in ["engine","family","feature_set","seed","split","rmse_prime","q99_ratio_pos"] if c in run_df.columns]
                stability_view = run_df[cols].copy()
                summary = stability_view.groupby(["engine","family","feature_set","split"]).agg(
                    n_runs=("rmse_prime","size"),
                    rmse_mean=("rmse_prime","mean"),
                    rmse_std=("rmse_prime","std"),
                    q99_mean=("q99_ratio_pos","mean"),
                ).reset_index()
                display(summary.sort_values(["split","rmse_mean"]).head(30))
            """
        ),
        code(
            """
            # Shake-up public/private (standard + tail stratified)
            for name in ["shakeup_v2_ensemble.parquet", "shakeup_v2_ensemble_tail.parquet", "shakeup_v2_single.parquet"]:
                p = ARTIFACT_V2 / name
                if p.exists():
                    sh = pd.read_parquet(p)
                    print(name, sh.shape)
                    display(sh.describe())
            """
        ),
        md(
            """
            ## Contrôles anti-fuite (checklist explicite)

            - [ ] `index` / IDs exclus comme features brutes
            - [ ] target encoding uniquement fit sur train fold (cross-fit)
            - [ ] validation `GroupKFold(id_client)` pour anti-fuite client
            - [ ] contrôle OOD train/test (nouvelles modalités, drift)
            - [ ] garde-fous sur la queue de sévérité / collapse de distribution

            **Décision pratique**: cette checklist doit être présentée explicitement en soutenance.
            """
        ),
        code(
            """
            # Synthèse business (classes de risque, quantiles, segments)
            train_raw, test_raw = load_project_datasets(DATA_DIR)
            y = train_raw[v2.TARGET_SEV_COL].astype(float)
            risk_class = pd.qcut(y.rank(method="first"), q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
            business_table = pd.DataFrame({"y": y, "risk_class": risk_class}).groupby("risk_class").agg(
                n=("y","size"),
                mean_cost=("y","mean"),
                median_cost=("y","median"),
                q95=("y", lambda s: float(np.quantile(s, 0.95))),
                q99=("y", lambda s: float(np.quantile(s, 0.99))),
            ).reset_index()
            display(business_table)
            """
        ),
        md(
            """
            ## Limites & risques (à compléter)

            - `Constat`: score public != score privé (seulement 1/3 public)
            - `Interprétation`: sélection opportuniste sur public très risquée
            - `Décision`: piloter les soumissions par robustesse locale (multi-splits + shake-up)

            Limites typiques à mentionner:
            - OOD catégoriel
            - queue lourde / événements rares
            - bruit cible et variables proxy imparfaites
            """
        ),
        md(
            """
            ## Décisions finales de modélisation (à synthétiser)

            À remplir explicitement:
            - Pourquoi **fréquence + sévérité** et pas seulement un modèle direct
            - Pourquoi ces **splits** de CV
            - Pourquoi `log1p` / Tweedie / tail mapping safe
            - Pourquoi single model vs ensemble (selon stabilité)
            """
        ),
        code(
            """
            # Exports diagnostics notebook 3 + storytelling summary
            if diag_tables:
                export_tables = {}
                for k, v in diag_tables.items():
                    if isinstance(v, pd.DataFrame):
                        export_tables[f"oof_model_diagnostics_{k}"] = v
                if export_tables:
                    export_analysis_tables(export_tables, ARTIFACT_DS)

            storytelling_md = ARTIFACT_DS / "storytelling_summary.md"
            storytelling_md.write_text(
                "\\n".join(
                    [
                        "# Storytelling summary (draft)",
                        "",
                        "## 1) Problème métier",
                        "- Tarifer une prime juste et robuste.",
                        "",
                        "## 2) Démarche data science",
                        "- Cadrage métier -> qualité -> EDA -> FE -> CV -> diagnostics -> robustesse.",
                        "",
                        "## 3) Choix techniques justifiés",
                        "- Two-part model (fréquence × sévérité)",
                        "- Multi-splits anti-fuite",
                        "- Garde-fous de queue et de distribution",
                        "",
                        "## 4) Points de vigilance",
                        "- OOD catégoriel et événements extrêmes",
                        "- Risque de shake-up public/privé",
                        "",
                    ]
                ),
                encoding="utf-8"
            )
            print("Exports DS notebook 3 ->", ARTIFACT_DS)
            print("Storytelling summary:", storytelling_md)
            """
        ),
        md(
            """
            ## Conclusion soutenance (10 points à dire oralement)

            1. Le problème est un problème de **tarification** avec coût d'erreur asymétrique (RMSE pertinent).
            2. La décomposition **fréquence × sévérité** est standard actuarielle et explicable.
            3. La sévérité présente une **queue lourde**, donc les choix de transformation et de robustesse sont centraux.
            4. Le dataset montre du **drift/OOD** sur certaines catégories fines.
            5. Le split `GroupKFold(id_client)` est indispensable pour éviter la fuite client.
            6. Les scores sont pilotés par **OOF multi-splits**, pas par un seul score public Kaggle.
            7. La calibration fréquence compte car la prime finale multiplie probabilité × montant.
            8. L'analyse d'erreurs par segments et déciles révèle où le modèle sous-/sur-tarife.
            9. Les contrôles anti-fuite et anti-collapse de distribution sont intégrés à la démarche.
            10. La stratégie de soumission privilégie la **robustesse privée** plutôt que l'optimisation opportuniste du public.
            """
        ),
    ]
    return _nb(cells)


def write_readme_ds() -> None:
    readme = textwrap.dedent(
        """
        # DS notebooks order

        1. `07_ds_cadrage_qualite_cv.ipynb`
        2. `08_ds_eda_segmentation_preprocessing.ipynb`
        3. `09_ds_model_diagnostics_storytelling.ipynb`

        ## Objectif

        Suite de notebooks "démarche data science" pour:
        - cadrage métier assurance,
        - compréhension et qualité des données,
        - EDA / segmentation / preprocessing,
        - diagnostics modèle / robustesse / storytelling de soutenance.

        ## Pré-requis

        - Notebook 07 et 08: utilisables directement avec `data/train.csv` et `data/test.csv`
        - Notebook 09: idéalement après génération des artefacts V2 (`artifacts/v2/*`)

        ## Modes d'exécution

        Les notebooks exposent:
        - `QUICK_ANALYSIS = True`: exécution rapide sur sous-échantillons
        - `FULL_ANALYSIS = False`: passer à `True` pour analyses plus lourdes

        ## Temps estimés (ordre de grandeur)

        - Notebook 07: 5-15 min
        - Notebook 08: 10-30 min (selon clustering / plots)
        - Notebook 09: 10-40 min (selon taille artefacts OOF)

        ## Sorties

        Les tables analytiques sont exportées sous `artifacts/ds/`.
        """
    ).strip() + "\n"
    (NB_DIR / "README_ds.md").write_text(readme, encoding="utf-8")


def main() -> None:
    notebooks = {
        "07_ds_cadrage_qualite_cv.ipynb": build_nb_07(),
        "08_ds_eda_segmentation_preprocessing.ipynb": build_nb_08(),
        "09_ds_model_diagnostics_storytelling.ipynb": build_nb_09(),
    }
    for name, nb in notebooks.items():
        path = NB_DIR / name
        _write_notebook(nb, path)
        print("written:", path)
    write_readme_ds()
    print("written:", NB_DIR / "README_ds.md")


if __name__ == "__main__":
    main()

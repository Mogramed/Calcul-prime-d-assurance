from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(text).strip() + "\n",
    }


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(text).strip() + "\n",
    }


def _nb(cells):
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


def build_nb_01() -> dict:
    cells = [
        md(
            """
            # 01 — EDA + CV Design (V1 robuste anti-overfitting)

            Objectifs:
            - Audit données train/test.
            - Feature engineering robuste/stable.
            - Construction de 2 schémas de validation:
              - `primary_time` (forward-chaining via `index`)
              - `secondary_group` (`GroupKFold` via `id_client`)
            - Export des artefacts de base pour la modélisation.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path
            import json
            import numpy as np
            import pandas as pd

            ROOT = Path.cwd()
            if not (ROOT / "src").exists():
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.v1_pipeline import (
                INDEX_COL,
                TARGET_FREQ_COL,
                TARGET_SEV_COL,
                load_train_test,
                prepare_datasets,
                build_primary_time_folds,
                build_secondary_group_folds,
                validate_folds_disjoint,
                export_fold_artifacts,
                ensure_dir,
            )

            DATA_DIR = ROOT / "data"
            ARTIFACT_DIR = ensure_dir(ROOT / "artifacts")
            """
        ),
        code(
            """
            train_raw, test_raw = load_train_test(DATA_DIR)
            bundle = prepare_datasets(train_raw, test_raw, drop_identifiers=True)

            print("train_raw:", train_raw.shape)
            print("test_raw :", test_raw.shape)
            print("X_train  :", bundle.X_train.shape)
            print("X_test   :", bundle.X_test.shape)
            print("n_features:", len(bundle.feature_cols))
            print("n_cat    :", len(bundle.cat_cols))
            """
        ),
        code(
            """
            # Sanity: target consistency + schema mismatch
            assert TARGET_SEV_COL in train_raw.columns, "Missing montant_sinistre in train"
            assert TARGET_FREQ_COL in train_raw.columns, "Missing nombre_sinistres in train"

            mismatch = sorted(set(bundle.X_train.columns) ^ set(bundle.X_test.columns))
            print("Schema mismatch columns:", len(mismatch))
            if mismatch:
                print(mismatch[:20])

            target_inconsistency = (
                ((train_raw[TARGET_FREQ_COL] == 0) & (train_raw[TARGET_SEV_COL] > 0)).sum()
                + ((train_raw[TARGET_FREQ_COL] > 0) & (train_raw[TARGET_SEV_COL] == 0)).sum()
            )
            print("Target inconsistency rows:", int(target_inconsistency))
            """
        ),
        code(
            """
            # Distribution cible
            y_freq = bundle.y_freq
            y_sev = bundle.y_sev
            pos = y_freq == 1

            print("Claim rate:", round(float(y_freq.mean()), 6), f"({int(pos.sum())}/{len(y_freq)})")
            print("Severity mean (all):", round(float(y_sev.mean()), 3))
            print("Severity mean (pos):", round(float(y_sev[pos].mean()), 3))
            print("Severity max:", round(float(y_sev.max()), 3))
            for q in [0.5, 0.75, 0.9, 0.95, 0.99]:
                print(f"sev_pos_q{int(q*100):02d}:", round(float(np.quantile(y_sev[pos], q)), 3))
            """
        ),
        code(
            """
            # Missingness + shift principal sur code_postal
            missing_top = bundle.X_train.isna().mean().sort_values(ascending=False).head(15)
            print("Top missing rates")
            display(missing_top.to_frame("missing_rate"))

            if "code_postal" in bundle.X_train.columns:
                tr_cp = bundle.X_train["code_postal"].astype(str)
                te_cp = bundle.X_test["code_postal"].astype(str)
                unseen_cp = (~te_cp.isin(set(tr_cp))).mean()
                print("Unseen code_postal ratio in test:", round(float(unseen_cp), 4))

            if {"cp2", "cp3"}.issubset(bundle.X_train.columns):
                for c in ["cp2", "cp3"]:
                    tr = bundle.X_train[c].astype(str)
                    te = bundle.X_test[c].astype(str)
                    unseen = (~te.isin(set(tr))).mean()
                    print(f"Unseen {c} ratio in test:", round(float(unseen), 4))
            """
        ),
        code(
            """
            # Folds primaires + secondaires
            folds_primary = build_primary_time_folds(train_raw, n_blocks=5, index_col=INDEX_COL)
            folds_secondary = build_secondary_group_folds(train_raw, n_splits=5, group_col="id_client")

            validate_folds_disjoint(folds_primary, check_full_coverage=False)
            validate_folds_disjoint(folds_secondary, check_full_coverage=True, n_rows=len(train_raw))

            print("Primary folds:", {k: (len(v[0]), len(v[1])) for k, v in folds_primary.items()})
            print("Secondary folds:", {k: (len(v[0]), len(v[1])) for k, v in folds_secondary.items()})
            """
        ),
        code(
            """
            export_fold_artifacts(
                train=train_raw,
                primary_folds=folds_primary,
                secondary_folds=folds_secondary,
                output_dir=ARTIFACT_DIR,
            )

            # Artefacts complémentaires
            pd.DataFrame({"cat_col": bundle.cat_cols}).to_csv(ARTIFACT_DIR / "cat_cols.csv", index=False)
            pd.DataFrame({"feature_col": bundle.feature_cols}).to_csv(
                ARTIFACT_DIR / "feature_cols.csv", index=False
            )

            target_df = pd.DataFrame({
                "row_idx": np.arange(len(bundle.y_freq), dtype=int),
                "index": train_raw[INDEX_COL].to_numpy(),
                "y_freq": bundle.y_freq.to_numpy(),
                "y_sev": bundle.y_sev.to_numpy(),
            })
            target_df.to_parquet(ARTIFACT_DIR / "y_train_targets.parquet", index=False)

            meta = {
                "data_dir": str(DATA_DIR),
                "n_train": int(len(train_raw)),
                "n_test": int(len(test_raw)),
                "n_features": int(len(bundle.feature_cols)),
                "n_cat": int(len(bundle.cat_cols)),
                "primary_folds": [int(k) for k in sorted(folds_primary.keys())],
                "secondary_folds": [int(k) for k in sorted(folds_secondary.keys())],
            }
            (ARTIFACT_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            print("Saved:")
            print("-", ARTIFACT_DIR / "folds_primary.parquet")
            print("-", ARTIFACT_DIR / "folds_secondary.parquet")
            print("-", ARTIFACT_DIR / "cat_cols.csv")
            print("-", ARTIFACT_DIR / "feature_cols.csv")
            print("-", ARTIFACT_DIR / "y_train_targets.parquet")
            print("-", ARTIFACT_DIR / "dataset_meta.json")
            """
        ),
        md(
            """
            ## Notes
            - `folds_primary` est un schéma temporel forward-chaining (4 folds valides sur 5 blocs).
            - `folds_secondary` couvre 100% des lignes avec disjonction par `id_client`.
            - On garde ces 2 splits pour la suite (`02_modeling_3_engines.ipynb`).
            """
        ),
    ]
    return _nb(cells)


def build_nb_02() -> dict:
    cells = [
        md(
            """
            # 02 — Modeling 3 Engines (CatBoost / LightGBM / XGBoost)

            Ce notebook exécute:
            - pipeline fréquence + gravité 2-parties,
            - variantes gravité (`classic`, `weighted_tail`),
            - calibration fréquence (`none`, `isotonic`, `platt`),
            - évaluations primaire + secondaire,
            - logging artefacts (`run_registry.csv`, `oof_predictions.parquet`).
            """
        ),
        code(
            """
            import sys
            from pathlib import Path
            import itertools
            import json
            import numpy as np
            import pandas as pd

            ROOT = Path.cwd()
            if not (ROOT / "src").exists():
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.v1_pipeline import (
                COARSE_CONFIGS,
                INDEX_COL,
                ensure_dir,
                load_train_test,
                prepare_datasets,
                run_cv_experiment,
                pick_top_configs,
                save_json,
            )

            DATA_DIR = ROOT / "data"
            ARTIFACT_DIR = ensure_dir(ROOT / "artifacts")

            # Runtime controls
            RUN_FULL = False      # True => 3 moteurs robustes complets (2-4h+)
            QUICK_TOPK_CONFIG = 1 # en mode rapide, ne garde que N config/moteur

            SEEDS = [42, 2026] if RUN_FULL else [42]
            SEVERITY_MODES = ["classic", "weighted_tail"] if RUN_FULL else ["classic", "weighted_tail"]
            CALIBRATION_METHODS = ["none", "isotonic", "platt"] if RUN_FULL else ["none", "isotonic"]
            """
        ),
        code(
            """
            def frame_to_folds(df: pd.DataFrame):
                folds = {}
                for fold_id, g in df.groupby("fold_id"):
                    tr = g.loc[g["role"] == "train", "row_idx"].to_numpy(dtype=int)
                    va = g.loc[g["role"] == "valid", "row_idx"].to_numpy(dtype=int)
                    folds[int(fold_id)] = (tr, va)
                return folds

            train_raw, test_raw = load_train_test(DATA_DIR)
            bundle = prepare_datasets(train_raw, test_raw, drop_identifiers=True)

            folds_primary_df = pd.read_parquet(ARTIFACT_DIR / "folds_primary.parquet")
            folds_secondary_df = pd.read_parquet(ARTIFACT_DIR / "folds_secondary.parquet")

            folds_primary = frame_to_folds(folds_primary_df)
            folds_secondary = frame_to_folds(folds_secondary_df)

            splits = {
                "primary_time": folds_primary,
                "secondary_group": folds_secondary,
            }

            print("Splits loaded:", {k: len(v) for k, v in splits.items()})
            """
        ),
        code(
            """
            # Execution loop
            all_fold_metrics = []
            all_run_metrics = []
            all_pred_frames = []

            for split_name, folds in splits.items():
                for engine, cfgs in COARSE_CONFIGS.items():
                    engine_cfgs = cfgs if RUN_FULL else cfgs[:QUICK_TOPK_CONFIG]
                    for cfg in engine_cfgs:
                        for severity_mode in SEVERITY_MODES:
                            for seed in SEEDS:
                                print(
                                    f"[RUN] split={split_name} engine={engine} cfg={cfg['config_id']} "
                                    f"sev={severity_mode} seed={seed}"
                                )

                                fold_df, run_df, pred_df = run_cv_experiment(
                                    split_name=split_name,
                                    engine=engine,
                                    config_id=cfg["config_id"],
                                    X=bundle.X_train,
                                    y_freq=bundle.y_freq,
                                    y_sev=bundle.y_sev,
                                    folds=folds,
                                    X_test=bundle.X_test,
                                    cat_cols=bundle.cat_cols,
                                    seed=seed,
                                    severity_mode=severity_mode,
                                    calibration_methods=CALIBRATION_METHODS,
                                    freq_params=cfg["freq_params"],
                                    sev_params=cfg["sev_params"],
                                )
                                all_fold_metrics.append(fold_df)
                                all_run_metrics.append(run_df)
                                all_pred_frames.append(pred_df)
            """
        ),
        code(
            """
            fold_metrics = pd.concat(all_fold_metrics, ignore_index=True) if all_fold_metrics else pd.DataFrame()
            run_metrics = pd.concat(all_run_metrics, ignore_index=True) if all_run_metrics else pd.DataFrame()
            preds = pd.concat(all_pred_frames, ignore_index=True) if all_pred_frames else pd.DataFrame()

            registry = pd.concat([fold_metrics, run_metrics], ignore_index=True)
            registry.to_csv(ARTIFACT_DIR / "run_registry.csv", index=False)
            preds.to_parquet(ARTIFACT_DIR / "oof_predictions.parquet", index=False)

            test_preds = preds[preds["is_test"] == 1].copy()
            test_preds.to_parquet(ARTIFACT_DIR / "test_predictions.parquet", index=False)

            print("Saved:")
            print("-", ARTIFACT_DIR / "run_registry.csv")
            print("-", ARTIFACT_DIR / "oof_predictions.parquet")
            print("-", ARTIFACT_DIR / "test_predictions.parquet")
            print("Registry rows:", len(registry), "| Pred rows:", len(preds))
            """
        ),
        code(
            """
            # Sélection top configs par moteur (sur split primaire)
            selected_configs = pick_top_configs(
                run_registry=run_metrics,
                split_name="primary_time",
                top_k_per_engine=2 if RUN_FULL else 1,
            )
            save_json(selected_configs, ARTIFACT_DIR / "selected_configs.json")
            selected_configs
            """
        ),
        code(
            """
            # Résumé principal
            summary_primary = (
                run_metrics[run_metrics["split"] == "primary_time"]
                .sort_values(["rmse_prime", "brier_freq"])
                .head(20)
            )
            summary_secondary = (
                run_metrics[run_metrics["split"] == "secondary_group"]
                .sort_values(["rmse_prime", "brier_freq"])
                .head(20)
            )

            print("Top primary:")
            display(summary_primary)
            print("Top secondary:")
            display(summary_secondary)
            """
        ),
        md(
            """
            ## Artefacts produits
            - `artifacts/run_registry.csv`
            - `artifacts/oof_predictions.parquet`
            - `artifacts/test_predictions.parquet`
            - `artifacts/selected_configs.json`
            """
        ),
    ]
    return _nb(cells)


def build_nb_03() -> dict:
    cells = [
        md(
            """
            # 03 — Ensemble + Submission (V1)

            Ce notebook:
            - sélectionne les runs finalistes,
            - optimise les poids d’ensemble (>=0, somme=1),
            - valide primaire/secondaire,
            - simule 2000 shake-ups public/private,
            - refit full train des finalistes,
            - exporte `artifacts/submission_v1.csv` (+ `submission.csv`).
            """
        ),
        code(
            """
            import sys
            from pathlib import Path
            import numpy as np
            import pandas as pd

            ROOT = Path.cwd()
            if not (ROOT / "src").exists():
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.v1_pipeline import (
                COARSE_CONFIGS,
                ensure_dir,
                load_json,
                load_train_test,
                optimize_non_negative_weights,
                prepare_datasets,
                fit_full_two_part_predict,
                fit_calibrator,
                apply_calibrator,
                simulate_public_private_shakeup,
                rmse,
                build_submission,
                save_json,
            )

            DATA_DIR = ROOT / "data"
            ARTIFACT_DIR = ensure_dir(ROOT / "artifacts")
            """
        ),
        code(
            """
            run_registry = pd.read_csv(ARTIFACT_DIR / "run_registry.csv")
            oof = pd.read_parquet(ARTIFACT_DIR / "oof_predictions.parquet")
            selected_configs = load_json(ARTIFACT_DIR / "selected_configs.json")

            train_raw, test_raw = load_train_test(DATA_DIR)
            bundle = prepare_datasets(train_raw, test_raw, drop_identifiers=True)

            run_metrics = run_registry[run_registry["level"] == "run"].copy()
            key_cols = ["engine", "config_id", "seed", "severity_mode", "calibration"]

            def mk_run_id(df):
                return (
                    df["engine"].astype(str) + "|" +
                    df["config_id"].astype(str) + "|" +
                    df["seed"].astype(str) + "|" +
                    df["severity_mode"].astype(str) + "|" +
                    df["calibration"].astype(str)
                )

            run_metrics["run_id"] = mk_run_id(run_metrics)
            oof["run_id"] = mk_run_id(oof)

            print("Run metrics:", run_metrics.shape)
            print("OOF rows:", oof.shape)
            """
        ),
        code(
            """
            # Candidate runs: best par moteur sur primaire, avec garde-fou secondaire
            prim = run_metrics[run_metrics["split"] == "primary_time"].copy()
            sec = run_metrics[run_metrics["split"] == "secondary_group"].copy()

            merged = prim.merge(
                sec[key_cols + ["rmse_prime", "q99_ratio_pos"]].rename(
                    columns={
                        "rmse_prime": "rmse_prime_secondary",
                        "q99_ratio_pos": "q99_ratio_secondary",
                    }
                ),
                on=key_cols,
                how="left",
            )
            merged["rmse_gap_secondary_minus_primary"] = (
                merged["rmse_prime_secondary"] - merged["rmse_prime"]
            )

            # limiter aux configs sélectionnées dans notebook 02
            mask_sel = np.zeros(len(merged), dtype=bool)
            for engine, cfg_ids in selected_configs.items():
                mask_sel |= (merged["engine"].eq(engine) & merged["config_id"].isin(cfg_ids))
            merged = merged[mask_sel].copy()

            finalists = []
            for engine, g in merged.groupby("engine"):
                # garde-fou overfit secondaire si disponible
                g_ok = g[g["rmse_gap_secondary_minus_primary"].fillna(0) <= 10.0]
                if g_ok.empty:
                    g_ok = g
                best = g_ok.sort_values(["rmse_prime", "rmse_prime_secondary"]).head(1)
                finalists.append(best)
            finalists = pd.concat(finalists, ignore_index=True)
            finalists["run_id"] = mk_run_id(finalists)

            finalists
            """
        ),
        code(
            """
            # Matrices prédiction pour optimisation des poids
            def build_matrix(pred_df, split_name, run_ids, is_test=0):
                d = pred_df[(pred_df["split"] == split_name) & (pred_df["is_test"] == is_test)].copy()
                d = d[d["run_id"].isin(run_ids)]
                wide = d.pivot_table(index="row_idx", columns="run_id", values="pred_prime", aggfunc="first")
                y = (
                    d.groupby("row_idx")["y_sev"].first()
                    if is_test == 0
                    else pd.Series(index=wide.index, dtype=float)
                )
                return wide, y

            run_ids = finalists["run_id"].tolist()

            Xp, yp = build_matrix(oof, "primary_time", run_ids, is_test=0)
            mask = Xp.notna().all(axis=1)
            Xp_fit = Xp.loc[mask]
            yp_fit = yp.loc[mask]

            weights = optimize_non_negative_weights(Xp_fit.values, yp_fit.values)
            weight_map = {rid: float(w) for rid, w in zip(Xp_fit.columns.tolist(), weights)}
            weight_map
            """
        ),
        code(
            """
            # Validation primaire + secondaire + fallback single
            ens_primary_pred = Xp_fit.values @ weights
            ens_primary_rmse = rmse(yp_fit.values, ens_primary_pred)

            Xs, ys = build_matrix(oof, "secondary_group", run_ids, is_test=0)
            sec_mask = Xs.notna().all(axis=1)
            if sec_mask.any():
                ens_secondary_pred = Xs.loc[sec_mask].values @ weights
                ens_secondary_rmse = rmse(ys.loc[sec_mask].values, ens_secondary_pred)
            else:
                ens_secondary_rmse = np.nan

            # best single (sur secondaire si dispo, sinon primaire)
            if sec_mask.any():
                single_scores = []
                for rid in run_ids:
                    p = Xs.loc[sec_mask, rid].values
                    s = rmse(ys.loc[sec_mask].values, p)
                    single_scores.append((rid, s))
            else:
                single_scores = []
                for rid in run_ids:
                    p = Xp_fit[rid].values
                    s = rmse(yp_fit.values, p)
                    single_scores.append((rid, s))
            best_single_run, best_single_rmse = sorted(single_scores, key=lambda x: x[1])[0]

            print("Ensemble RMSE primary:", round(float(ens_primary_rmse), 6))
            print("Ensemble RMSE secondary:", round(float(ens_secondary_rmse), 6) if not np.isnan(ens_secondary_rmse) else np.nan)
            print("Best single run:", best_single_run)
            print("Best single RMSE :", round(float(best_single_rmse), 6))
            """
        ),
        code(
            """
            # Shake-up simulation (public/private)
            shake_ens = simulate_public_private_shakeup(
                yp_fit.values, ens_primary_pred, n_sim=2000, public_ratio=1/3, seed=42
            )
            shake_single = simulate_public_private_shakeup(
                yp_fit.values, Xp_fit[best_single_run].values, n_sim=2000, public_ratio=1/3, seed=42
            )

            ens_gap_std = float(shake_ens["gap_public_minus_private"].std())
            single_gap_std = float(shake_single["gap_public_minus_private"].std())

            # Décision finale: ensemble seulement s'il n'est pas instable ni pire en secondaire
            use_ensemble = True
            if not np.isnan(ens_secondary_rmse):
                if ens_secondary_rmse > best_single_rmse + 1.0:
                    use_ensemble = False
            if ens_gap_std > single_gap_std * 1.05:
                use_ensemble = False

            print("ens_gap_std:", round(ens_gap_std, 6))
            print("single_gap_std:", round(single_gap_std, 6))
            print("use_ensemble:", use_ensemble)

            shake_ens.to_parquet(ARTIFACT_DIR / "shakeup_ensemble.parquet", index=False)
            shake_single.to_parquet(ARTIFACT_DIR / "shakeup_single.parquet", index=False)
            """
        ),
        code(
            """
            # Refit full train des runs finalistes puis prédiction test
            cfg_lookup = {
                engine: {c["config_id"]: c for c in cfgs}
                for engine, cfgs in COARSE_CONFIGS.items()
            }

            full_test_preds = {}
            y_freq_np = bundle.y_freq.to_numpy(dtype=int)
            y_sev_np = bundle.y_sev.to_numpy(dtype=float)

            for _, r in finalists.iterrows():
                rid = r["run_id"]
                engine = r["engine"]
                config_id = r["config_id"]
                seed = int(r["seed"])
                severity_mode = r["severity_mode"]
                calibration = r["calibration"]

                cfg = cfg_lookup[engine][config_id]
                freq_raw_te, sev_te = fit_full_two_part_predict(
                    engine=engine,
                    X_train=bundle.X_train,
                    y_freq_train=y_freq_np,
                    y_sev_train=y_sev_np,
                    X_test=bundle.X_test,
                    cat_cols=bundle.cat_cols,
                    seed=seed,
                    severity_mode=severity_mode,
                    freq_params=cfg["freq_params"],
                    sev_params=cfg["sev_params"],
                )

                if calibration != "none":
                    # calibrateur appris sur OOF (split primaire)
                    oof_run = oof[
                        (oof["is_test"] == 0)
                        & (oof["split"] == "primary_time")
                        & (oof["run_id"] == rid)
                    ].copy()
                    valid = oof_run["pred_freq"].notna()
                    cal = fit_calibrator(
                        oof_run.loc[valid, "pred_freq"].to_numpy(),
                        oof_run.loc[valid, "y_freq"].to_numpy(),
                        method=calibration,
                    )
                    freq_te = apply_calibrator(cal, freq_raw_te, method=calibration)
                else:
                    freq_te = freq_raw_te

                full_test_preds[rid] = np.maximum(freq_te * sev_te, 0.0)
            """
        ),
        code(
            """
            # Combinaison finale + export submission
            if use_ensemble:
                final_runs = Xp_fit.columns.tolist()
                w = np.array([weight_map[rid] for rid in final_runs], dtype=float)
                test_matrix = np.column_stack([full_test_preds[rid] for rid in final_runs])
                pred_final = test_matrix @ w
                final_strategy = "ensemble"
            else:
                pred_final = full_test_preds[best_single_run]
                final_strategy = "single"

            submission = build_submission(test_raw["index"], pred_final)
            submission.to_csv(ARTIFACT_DIR / "submission_v1.csv", index=False)
            submission.to_csv(ARTIFACT_DIR / "submission.csv", index=False)

            final_meta = {
                "final_strategy": final_strategy,
                "final_runs": finalists["run_id"].tolist(),
                "weights": weight_map,
                "best_single_run": best_single_run,
                "ens_primary_rmse": float(ens_primary_rmse),
                "ens_secondary_rmse": float(ens_secondary_rmse) if not np.isnan(ens_secondary_rmse) else None,
                "best_single_rmse": float(best_single_rmse),
                "ens_gap_std": float(ens_gap_std),
                "single_gap_std": float(single_gap_std),
            }
            save_json(final_meta, ARTIFACT_DIR / "ensemble_weights_v1.json")
            finalists.to_csv(ARTIFACT_DIR / "finalist_runs.csv", index=False)

            print("Saved:")
            print("-", ARTIFACT_DIR / "submission_v1.csv")
            print("-", ARTIFACT_DIR / "submission.csv")
            print("-", ARTIFACT_DIR / "ensemble_weights_v1.json")
            print("-", ARTIFACT_DIR / "finalist_runs.csv")
            submission.head()
            """
        ),
    ]
    return _nb(cells)


def main() -> None:
    nb1 = build_nb_01()
    nb2 = build_nb_02()
    nb3 = build_nb_03()

    _write_notebook(nb1, NB_DIR / "01_eda_cv_design.ipynb")
    _write_notebook(nb2, NB_DIR / "02_modeling_3_engines.ipynb")
    _write_notebook(nb3, NB_DIR / "03_ensemble_submission.ipynb")

    print("Generated:")
    print("-", NB_DIR / "01_eda_cv_design.ipynb")
    print("-", NB_DIR / "02_modeling_3_engines.ipynb")
    print("-", NB_DIR / "03_ensemble_submission.ipynb")


if __name__ == "__main__":
    main()

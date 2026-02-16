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
            # 01 - EDA + CV Design (V2)

            Objectifs:
            - audit data + drift train/test,
            - création feature sets V2,
            - création des 3 splits anti-overfitting,
            - diagnostic OOD + biais segment v1,
            - export des artefacts `artifacts/v2/*`.
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

            from src.v2_pipeline import (
                DEFAULT_V2_DIR,
                INDEX_COL,
                compute_ood_diagnostics,
                compute_segment_bias_from_oof,
                ensure_dir,
                export_split_artifacts_v2,
                load_train_test,
                prepare_feature_sets,
                build_split_registry,
                validate_folds_disjoint,
                validate_group_disjoint,
            )

            DATA_DIR = ROOT / "data"
            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)
            """
        ),
        code(
            """
            train_raw, test_raw = load_train_test(DATA_DIR)
            feature_sets = prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
            print("feature sets:", list(feature_sets.keys()))
            for k, b in feature_sets.items():
                print(k, "X_train", b.X_train.shape, "X_test", b.X_test.shape, "n_cat", len(b.cat_cols))

            bundle = feature_sets["base_v2"]
            """
        ),
        code(
            """
            splits = build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")
            for name, folds in splits.items():
                validate_folds_disjoint(
                    folds,
                    check_full_coverage=(name in {"secondary_group", "aux_blocked5"}),
                    n_rows=len(train_raw),
                )
                if name == "secondary_group":
                    validate_group_disjoint(folds, train_raw["id_client"])
                print(name, {k: (len(v[0]), len(v[1])) for k, v in folds.items()})

            export_split_artifacts_v2(train=train_raw, splits=splits, output_dir=ARTIFACT_V2)
            """
        ),
        code(
            """
            # Drift/OOD
            ood = compute_ood_diagnostics(
                bundle.X_train.join(train_raw[[INDEX_COL]], how="left"),
                bundle.X_test.join(test_raw[[INDEX_COL]], how="left"),
            )
            ood = ood.sort_values("unseen_test_levels", ascending=False)
            display(ood.head(20))
            """
        ),
        code(
            """
            # Diagnostics biais segment sur OOF v1 (si disponible)
            seg = pd.DataFrame()
            oof_v1_path = ROOT / "artifacts" / "oof_predictions.parquet"
            ens_v1_path = ROOT / "artifacts" / "ensemble_weights_v1.json"

            if oof_v1_path.exists():
                oof_v1 = pd.read_parquet(oof_v1_path)
                if ens_v1_path.exists():
                    meta = json.loads(ens_v1_path.read_text(encoding="utf-8"))
                    run_id = meta.get("best_single_run")
                else:
                    oof_v1["run_id"] = (
                        oof_v1["engine"].astype(str) + "|"
                        + oof_v1["config_id"].astype(str) + "|"
                        + oof_v1["seed"].astype(int).astype(str) + "|"
                        + oof_v1["severity_mode"].astype(str) + "|"
                        + oof_v1["calibration"].astype(str)
                    )
                    run_id = oof_v1["run_id"].dropna().iloc[0]
                seg = compute_segment_bias_from_oof(train_raw, oof_v1, run_id=run_id, split_name="primary_time")
                print("segment rows:", len(seg))
                display(seg.head(20))
            else:
                print("OOF v1 absent -> skip segment bias diagnostics.")
            """
        ),
        code(
            """
            diag = pd.concat([ood, seg], ignore_index=True, sort=False)
            diag.to_parquet(ARTIFACT_V2 / "segment_diagnostics_v2.parquet", index=False)

            meta = {
                "n_train": int(len(train_raw)),
                "n_test": int(len(test_raw)),
                "feature_sets": {k: {"n_features": len(v.feature_cols), "n_cat": len(v.cat_cols)} for k, v in feature_sets.items()},
                "splits": {k: sorted([int(x) for x in v.keys()]) for k, v in splits.items()},
            }
            (ARTIFACT_V2 / "dataset_meta_v2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print("saved:", ARTIFACT_V2 / "segment_diagnostics_v2.parquet")
            print("saved:", ARTIFACT_V2 / "dataset_meta_v2.json")
            """
        ),
    ]
    return _nb(cells)


def build_nb_02() -> dict:
    cells = [
        md(
            """
            # 02 - Feature Engineering V2

            Notebook de comparaison des feature sets (`base_v2`, `robust_v2`, `compact_v2`)
            avec un benchmark rapide cohérent anti-overfitting.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path
            import pandas as pd

            ROOT = Path.cwd()
            if not (ROOT / "src").exists():
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.v2_pipeline import (
                DEFAULT_V2_DIR,
                ensure_dir,
                load_train_test,
                prepare_feature_sets,
                build_split_registry,
                run_benchmark,
                V2_COARSE_CONFIGS,
            )
            DATA_DIR = ROOT / "data"
            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)
            """
        ),
        code(
            """
            train_raw, test_raw = load_train_test(DATA_DIR)
            feature_sets = prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
            splits = build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")

            spec_template = {
                "engine": "catboost",
                "family": "two_part_classic",
                "config_id": "cb_v2_c1",
                "severity_mode": "classic",
                "calibration_methods": ["none", "isotonic"],
                "use_tail_mapper": True,
                "use_target_encoding": True,
                "target_encode_cols": ["code_postal", "cp3", "modele_vehicule", "marque_modele"],
                "target_encoding_smoothing": 20.0,
                "freq_params": V2_COARSE_CONFIGS["catboost"][0]["freq_params"],
                "sev_params": V2_COARSE_CONFIGS["catboost"][0]["sev_params"],
                "direct_params": V2_COARSE_CONFIGS["catboost"][0]["direct_params"],
            }
            """
        ),
        code(
            """
            rows = []
            for fs_name, bundle in feature_sets.items():
                spec = dict(spec_template)
                spec["feature_set"] = fs_name
                fold_df, run_df, pred_df = run_benchmark(spec, bundle=bundle, splits=splits, seed=42)
                run_df["feature_set"] = fs_name
                rows.append(run_df)
                print(fs_name, "rows", len(run_df))

            comp = pd.concat(rows, ignore_index=True)
            comp.to_csv(ARTIFACT_V2 / "feature_set_comparison_v2.csv", index=False)
            display(
                comp[comp["level"] == "run"]
                .sort_values(["split", "rmse_prime"])
                .head(30)
            )
            """
        ),
    ]
    return _nb(cells)


def build_nb_03() -> dict:
    cells = [
        md(
            """
            # 03 - Objective Screening V2

            Screening des familles d'objectifs:
            - `two_part_classic`,
            - `two_part_tweedie`,
            - `direct_tweedie`.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path
            import itertools
            import pandas as pd

            ROOT = Path.cwd()
            if not (ROOT / "src").exists():
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.v2_pipeline import (
                DEFAULT_V2_DIR,
                ensure_dir,
                load_train_test,
                prepare_feature_sets,
                build_split_registry,
                run_benchmark,
                V2_COARSE_CONFIGS,
                V2_SCREENING_FAMILIES,
            )
            DATA_DIR = ROOT / "data"
            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)
            """
        ),
        code(
            """
            train_raw, test_raw = load_train_test(DATA_DIR)
            feature_sets = prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
            bundle = feature_sets["robust_v2"]
            splits = build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")
            primary_only = {"primary_time": splits["primary_time"]}

            rows_f = []
            rows_r = []
            rows_p = []
            for engine in ["catboost", "lightgbm", "xgboost"]:
                cfg = V2_COARSE_CONFIGS[engine][0]
                for fam in V2_SCREENING_FAMILIES:
                    spec = {
                        "feature_set": "robust_v2",
                        "engine": engine,
                        "family": fam["family"],
                        "severity_mode": fam["severity_mode"],
                        "tweedie_power": fam["tweedie_power"],
                        "config_id": cfg["config_id"],
                        "calibration_methods": ["none"],
                        "use_tail_mapper": fam["family"] != "direct_tweedie",
                        "use_target_encoding": True,
                        "target_encode_cols": ["code_postal", "cp3", "modele_vehicule", "marque_modele"],
                        "target_encoding_smoothing": 20.0,
                        "freq_params": cfg["freq_params"],
                        "sev_params": cfg["sev_params"],
                        "direct_params": cfg["direct_params"],
                        "split_names": ["primary_time"],
                    }
                    print("[screen]", engine, fam)
                    f, r, p = run_benchmark(spec, bundle=bundle, splits=primary_only, seed=42)
                    rows_f.append(f)
                    rows_r.append(r)
                    rows_p.append(p)

            fold_df = pd.concat(rows_f, ignore_index=True)
            run_df = pd.concat(rows_r, ignore_index=True)
            pred_df = pd.concat(rows_p, ignore_index=True)
            run_df.to_csv(ARTIFACT_V2 / "objective_screening_v2.csv", index=False)
            run_df.sort_values("rmse_prime").head(30)
            """
        ),
    ]
    return _nb(cells)


def build_nb_04() -> dict:
    cells = [
        md(
            """
            # 04 - Modeling 3 Engines V2

            Benchmark principal V2.
            - Phase B: coarse configs (6/moteur en mode full),
            - Phase C: robustesse multi-seeds sur top configs.
            """
        ),
        code(
            """
            import sys
            from pathlib import Path
            import pandas as pd

            ROOT = Path.cwd()
            if not (ROOT / "src").exists():
                ROOT = ROOT.parent
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.v2_pipeline import (
                DEFAULT_V2_DIR,
                ensure_dir,
                load_train_test,
                prepare_feature_sets,
                build_split_registry,
                run_benchmark,
                V2_COARSE_CONFIGS,
            )

            DATA_DIR = ROOT / "data"
            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)

            RUN_FULL = False      # True pour overnight 8-12h
            QUICK_CFG_PER_ENGINE = 1
            QUICK_TOP2_PER_ENGINE = 1
            """
        ),
        code(
            """
            train_raw, test_raw = load_train_test(DATA_DIR)
            feature_sets = prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
            bundle = feature_sets["base_v2"]
            splits = build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")

            if RUN_FULL:
                families = ["two_part_classic", "two_part_tweedie", "direct_tweedie"]
                sev_modes = ["classic", "weighted_tail", "winsorized"]
                tweedie_powers = [1.3, 1.5, 1.7]
                seeds = [42, 2026]
                calibs = ["none", "isotonic"]
                cfg_per_engine = 6
            else:
                families = ["two_part_classic", "two_part_tweedie", "direct_tweedie"]
                sev_modes = ["classic", "weighted_tail"]
                tweedie_powers = [1.5]
                seeds = [42]
                calibs = ["none", "isotonic"]
                cfg_per_engine = QUICK_CFG_PER_ENGINE
            """
        ),
        code(
            """
            all_f = []
            all_r = []
            all_p = []

            for engine in ["catboost", "lightgbm", "xgboost"]:
                cfgs = V2_COARSE_CONFIGS[engine][:cfg_per_engine]
                for cfg in cfgs:
                    for fam in families:
                        for sev_mode in sev_modes:
                            if fam == "direct_tweedie" and sev_mode != "classic":
                                continue
                            for tw_power in (tweedie_powers if fam == "two_part_tweedie" else [1.5]):
                                for seed in seeds:
                                    spec = {
                                        "feature_set": "base_v2",
                                        "engine": engine,
                                        "family": fam,
                                        "severity_mode": sev_mode,
                                        "tweedie_power": tw_power,
                                        "config_id": cfg["config_id"],
                                        "calibration_methods": calibs,
                                        "use_tail_mapper": fam != "direct_tweedie",
                                        "use_target_encoding": True,
                                        "target_encode_cols": ["code_postal", "cp3", "modele_vehicule", "marque_modele"],
                                        "target_encoding_smoothing": 20.0,
                                        "freq_params": cfg["freq_params"],
                                        "sev_params": cfg["sev_params"],
                                        "direct_params": cfg["direct_params"],
                                    }
                                    print("[run]", engine, cfg["config_id"], fam, sev_mode, tw_power, "seed", seed)
                                    f, r, p = run_benchmark(spec, bundle=bundle, splits=splits, seed=seed)
                                    all_f.append(f)
                                    all_r.append(r)
                                    all_p.append(p)

            fold_df = pd.concat(all_f, ignore_index=True)
            run_df = pd.concat(all_r, ignore_index=True)
            pred_df = pd.concat(all_p, ignore_index=True)

            run_df.to_csv(ARTIFACT_V2 / "run_registry_v2.csv", index=False)
            pred_df.to_parquet(ARTIFACT_V2 / "oof_predictions_v2.parquet", index=False)
            print("saved:", ARTIFACT_V2 / "run_registry_v2.csv")
            print("saved:", ARTIFACT_V2 / "oof_predictions_v2.parquet")
            run_df.sort_values(["split", "rmse_prime"]).head(50)
            """
        ),
    ]
    return _nb(cells)


def build_nb_05() -> dict:
    cells = [
        md(
            """
            # 05 - Ensemble + Robustness V2

            Sélection de runs robustes, optimisation des poids, et stress tests public/private.
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

            from src.v2_pipeline import (
                DEFAULT_V2_DIR,
                ensure_dir,
                load_json,
                optimize_non_negative_weights,
                rmse,
                select_final_models,
                simulate_public_private_shakeup_v2,
                build_model_cards,
            )
            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)
            """
        ),
        code(
            """
            run_df = pd.read_csv(ARTIFACT_V2 / "run_registry_v2.csv")
            oof = pd.read_parquet(ARTIFACT_V2 / "oof_predictions_v2.parquet")

            selected = select_final_models(run_df, risk_policy="stability_private")
            selected.to_csv(ARTIFACT_V2 / "selected_models_v2.csv", index=False)
            selected.head(10)
            """
        ),
        code(
            """
            def build_matrix(pred_df, split_name, run_ids, is_test=0):
                d = pred_df[(pred_df["split"] == split_name) & (pred_df["is_test"] == is_test)].copy()
                d = d[d["run_id"].isin(run_ids)]
                wide = d.pivot_table(index="row_idx", columns="run_id", values="pred_prime", aggfunc="first")
                y = d.groupby("row_idx")["y_sev"].first() if is_test == 0 else pd.Series(index=wide.index, dtype=float)
                return wide, y

            run_ids = selected["run_id"].tolist()
            Xp, yp = build_matrix(oof, "primary_time", run_ids, is_test=0)
            mask = Xp.notna().all(axis=1)
            Xp_fit = Xp.loc[mask]
            yp_fit = yp.loc[mask]

            w = optimize_non_negative_weights(Xp_fit.values, yp_fit.values)
            weight_map = {rid: float(v) for rid, v in zip(Xp_fit.columns.tolist(), w)}
            ens_primary = Xp_fit.values @ w
            ens_primary_rmse = rmse(yp_fit.values, ens_primary)

            print("models:", len(run_ids))
            print("primary rmse:", ens_primary_rmse)
            weight_map
            """
        ),
        code(
            """
            # fallback single + shakeup
            single_scores = [(rid, rmse(yp_fit.values, Xp_fit[rid].values)) for rid in Xp_fit.columns]
            best_single_run, best_single_rmse = sorted(single_scores, key=lambda x: x[1])[0]

            sh_ens = simulate_public_private_shakeup_v2(
                yp_fit.values, ens_primary, n_sim=2000, public_ratio=1/3, seed=42
            )
            sh_ens_tail = simulate_public_private_shakeup_v2(
                yp_fit.values,
                ens_primary,
                n_sim=2000,
                public_ratio=1/3,
                seed=42,
                stratified_tail=True,
                tail_quantile=0.9,
                tail_public_share=0.5,
            )
            sh_single = simulate_public_private_shakeup_v2(
                yp_fit.values, Xp_fit[best_single_run].values, n_sim=2000, public_ratio=1/3, seed=42
            )

            ens_gap_std = float(sh_ens["gap_public_minus_private"].std())
            ens_tail_gap_std = float(sh_ens_tail["gap_public_minus_private"].std())
            single_gap_std = float(sh_single["gap_public_minus_private"].std())

            use_ensemble = not (ens_gap_std > single_gap_std * 1.05 or ens_tail_gap_std > single_gap_std * 1.10)
            strategy = "ensemble" if use_ensemble else "single"

            sh_ens.to_parquet(ARTIFACT_V2 / "shakeup_v2_ensemble.parquet", index=False)
            sh_ens_tail.to_parquet(ARTIFACT_V2 / "shakeup_v2_ensemble_tail.parquet", index=False)
            sh_single.to_parquet(ARTIFACT_V2 / "shakeup_v2_single.parquet", index=False)

            meta = {
                "strategy": strategy,
                "run_ids": run_ids,
                "weights": weight_map,
                "best_single_run": best_single_run,
                "ens_primary_rmse": float(ens_primary_rmse),
                "best_single_rmse": float(best_single_rmse),
                "ens_gap_std": ens_gap_std,
                "ens_tail_gap_std": ens_tail_gap_std,
                "single_gap_std": single_gap_std,
            }
            (ARTIFACT_V2 / "ensemble_weights_v2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            cards = build_model_cards(run_df, selected)
            cards.to_csv(ARTIFACT_V2 / "model_cards_v2.csv", index=False)
            print("strategy:", strategy)
            """
        ),
    ]
    return _nb(cells)


def build_nb_06() -> dict:
    cells = [
        md(
            """
            # 06 - Submission Report V2

            Refit full train des runs sélectionnés et génération:
            - `submission_v2_robust.csv`
            - `submission_v2_single.csv`
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

            from src.v2_pipeline import (
                DEFAULT_V2_DIR,
                ensure_dir,
                load_train_test,
                prepare_feature_sets,
                fit_full_predict,
                fit_calibrator,
                apply_calibrator,
                fit_tail_mapper,
                apply_tail_mapper,
                build_submission,
                V2_COARSE_CONFIGS,
            )
            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)
            DATA_DIR = ROOT / "data"
            """
        ),
        code(
            """
            run_df = pd.read_csv(ARTIFACT_V2 / "run_registry_v2.csv")
            oof = pd.read_parquet(ARTIFACT_V2 / "oof_predictions_v2.parquet")
            selected = pd.read_csv(ARTIFACT_V2 / "selected_models_v2.csv")
            ens_meta = json.loads((ARTIFACT_V2 / "ensemble_weights_v2.json").read_text(encoding="utf-8"))

            train_raw, test_raw = load_train_test(DATA_DIR)
            feature_sets = prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)

            cfg_lookup = {
                engine: {c["config_id"]: c for c in cfgs}
                for engine, cfgs in V2_COARSE_CONFIGS.items()
            }
            """
        ),
        code(
            """
            preds = {}
            for _, r in selected.iterrows():
                run_id = r["run_id"]
                fs_name = r["feature_set"]
                bundle = feature_sets[fs_name]
                engine = r["engine"]
                config_id = r["config_id"]
                seed = int(r["seed"])
                family = r["family"]
                sev_mode = r["severity_mode"]
                calib = r["calibration"]
                tail_name = r["tail_mapper"]
                cfg = cfg_lookup[engine][config_id]

                spec = {
                    "engine": engine,
                    "family": family,
                    "severity_mode": sev_mode,
                    "config_id": config_id,
                    "freq_params": cfg["freq_params"],
                    "sev_params": cfg["sev_params"],
                    "direct_params": cfg["direct_params"],
                    "use_target_encoding": True,
                    "target_encode_cols": ["code_postal", "cp3", "modele_vehicule", "marque_modele"],
                    "target_encoding_smoothing": 20.0,
                }
                out = fit_full_predict(spec=spec, bundle=bundle, seed=seed, valid_ratio=0.1)
                test_freq = out["test_freq"].copy()
                test_sev = out["test_sev"].copy()
                test_prime = out["test_prime"].copy()

                if calib != "none":
                    o = oof[(oof["is_test"] == 0) & (oof["split"] == "primary_time") & (oof["run_id"] == run_id)].copy()
                    ok = o["pred_freq"].notna()
                    if ok.any():
                        cal = fit_calibrator(
                            o.loc[ok, "pred_freq"].to_numpy(),
                            o.loc[ok, "y_freq"].to_numpy(),
                            method=calib,
                        )
                        test_freq = apply_calibrator(cal, test_freq, method=calib)

                if tail_name != "none" and family != "direct_tweedie":
                    o = oof[(oof["is_test"] == 0) & (oof["split"] == "primary_time") & (oof["run_id"] == run_id)].copy()
                    pos = (o["y_freq"] == 1) & o["pred_sev"].notna()
                    if pos.any():
                        mapper = fit_tail_mapper(
                            o.loc[pos, "pred_sev"].to_numpy(),
                            o.loc[pos, "y_sev"].to_numpy(),
                        )
                        test_sev = apply_tail_mapper(mapper, test_sev)

                if family == "direct_tweedie":
                    pred = np.maximum(test_prime, 0.0)
                else:
                    pred = np.maximum(test_freq * test_sev, 0.0)
                preds[run_id] = pred

            print("ready predictions:", len(preds))
            """
        ),
        code(
            """
            strategy = ens_meta.get("strategy", "single")
            run_ids = ens_meta.get("run_ids", list(preds.keys()))
            weights = ens_meta.get("weights", {})
            best_single = ens_meta.get("best_single_run")

            if strategy == "ensemble":
                mat = np.column_stack([preds[rid] for rid in run_ids])
                w = np.array([weights.get(rid, 0.0) for rid in run_ids], dtype=float)
                if w.sum() <= 0:
                    w = np.full(len(run_ids), 1.0 / len(run_ids))
                else:
                    w = w / w.sum()
                pred_robust = mat @ w
            else:
                pred_robust = preds[best_single]

            pred_single = preds[best_single]

            sub_robust = build_submission(test_raw["index"], pred_robust)
            sub_single = build_submission(test_raw["index"], pred_single)
            sub_robust.to_csv(ARTIFACT_V2 / "submission_v2_robust.csv", index=False)
            sub_single.to_csv(ARTIFACT_V2 / "submission_v2_single.csv", index=False)

            report = {
                "strategy": strategy,
                "n_runs": len(run_ids),
                "best_single_run": best_single,
                "submission_robust": str(ARTIFACT_V2 / "submission_v2_robust.csv"),
                "submission_single": str(ARTIFACT_V2 / "submission_v2_single.csv"),
            }
            (ARTIFACT_V2 / "submission_report_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(report)
            """
        ),
    ]
    return _nb(cells)


def main() -> None:
    notebooks = {
        "01_eda_cv_design.ipynb": build_nb_01(),
        "02_feature_engineering_v2.ipynb": build_nb_02(),
        "03_objective_screening_v2.ipynb": build_nb_03(),
        "04_modeling_3_engines_v2.ipynb": build_nb_04(),
        "05_ensemble_robustness_v2.ipynb": build_nb_05(),
        "06_submission_report_v2.ipynb": build_nb_06(),
    }
    for name, nb in notebooks.items():
        path = NB_DIR / name
        _write_notebook(nb, path)
        print("written:", path)


if __name__ == "__main__":
    main()

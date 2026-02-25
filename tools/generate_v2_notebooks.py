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


def build_nb_01() -> dict:
    cells = [
        md(
            """
            # 01 - EDA + CV design (V2.1)

            Objectifs:
            - verifier le data contract train/test,
            - auditer NA, zeros techniques, distribution cible, extremes,
            - diagnostiquer le drift/OOD (categoriel + numerique),
            - valider les 3 splits anti-overfitting,
            - exporter les diagnostics versionnes.
            """
        ),
        code(
            """
            import sys
            import json
            from pathlib import Path
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
                TARGET_SEV_COL,
                ensure_dir,
                load_train_test,
                prepare_feature_sets,
                build_split_registry,
                validate_folds_disjoint,
                validate_group_disjoint,
                export_split_artifacts_v2,
                compute_ood_diagnostics,
                compute_segment_bias_from_oof,
            )

            DATA_DIR = ROOT / "data"
            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)
            """
        ),
        code(
            """
            train_raw, test_raw = load_train_test(DATA_DIR)
            feature_sets = prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
            bundle = feature_sets["base_v2"]

            print("train shape:", train_raw.shape, "test shape:", test_raw.shape)
            print("feature sets:", list(feature_sets.keys()))
            for fs_name, b in feature_sets.items():
                print(fs_name, "features", len(b.feature_cols), "cat", len(b.cat_cols), "num", len(b.num_cols))
            """
        ),
        md("## Data contract"),
        code(
            """
            train_cols = set(train_raw.columns)
            test_cols = set(test_raw.columns)
            common_cols = sorted(train_cols.intersection(test_cols))
            train_only = sorted(train_cols - test_cols)
            test_only = sorted(test_cols - train_cols)

            contract = pd.DataFrame(
                [
                    {"item": "train_rows", "value": int(len(train_raw))},
                    {"item": "test_rows", "value": int(len(test_raw))},
                    {"item": "common_columns", "value": int(len(common_cols))},
                    {"item": "train_only_columns", "value": int(len(train_only))},
                    {"item": "test_only_columns", "value": int(len(test_only))},
                ]
            )
            display(contract)
            print("train_only:", train_only)
            print("test_only:", test_only)
            """
        ),
        md("## Target, missing values and technical zeros"),
        code(
            """
            y = train_raw[TARGET_SEV_COL].astype(float)
            y_pos = y[y > 0]
            target_stats = pd.DataFrame(
                [
                    {"metric": "n_train", "value": int(len(y))},
                    {"metric": "claim_rate", "value": float((y > 0).mean())},
                    {"metric": "mean_positive", "value": float(y_pos.mean()) if len(y_pos) else np.nan},
                    {"metric": "q95_positive", "value": float(y_pos.quantile(0.95)) if len(y_pos) else np.nan},
                    {"metric": "q99_positive", "value": float(y_pos.quantile(0.99)) if len(y_pos) else np.nan},
                    {"metric": "max_positive", "value": float(y_pos.max()) if len(y_pos) else np.nan},
                ]
            )
            display(target_stats)

            na_train = train_raw.isna().mean().sort_values(ascending=False).rename("na_ratio_train")
            na_test = test_raw.isna().mean().sort_values(ascending=False).rename("na_ratio_test")
            na_table = pd.concat([na_train, na_test], axis=1).fillna(0.0).reset_index().rename(columns={"index": "feature"})
            display(na_table.head(20))

            zero_cols = [c for c in ["poids_vehicule", "cylindre_vehicule"] if c in train_raw.columns]
            zero_rows = []
            for c in zero_cols:
                zero_rows.append(
                    {
                        "feature": c,
                        "zero_ratio_train": float((train_raw[c] == 0).mean()),
                        "zero_ratio_test": float((test_raw[c] == 0).mean()),
                    }
                )
            display(pd.DataFrame(zero_rows))
            """
        ),
        md("## Splits anti-overfitting"),
        code(
            """
            splits = build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")
            for split_name, folds in splits.items():
                validate_folds_disjoint(
                    folds,
                    check_full_coverage=(split_name in {"secondary_group", "aux_blocked5"}),
                    n_rows=len(train_raw),
                )
                if split_name == "secondary_group":
                    validate_group_disjoint(folds, train_raw["id_client"])
                print(split_name, {k: (len(v[0]), len(v[1])) for k, v in folds.items()})

            export_split_artifacts_v2(train=train_raw, splits=splits, output_dir=ARTIFACT_V2)
            print("saved fold artifacts under", ARTIFACT_V2)
            """
        ),
        md("## Drift and OOD diagnostics"),
        code(
            """
            ood = compute_ood_diagnostics(bundle.X_train, bundle.X_test)
            ood_focus = ood[ood["feature"].isin(["code_postal", "cp3", "cp2", "modele_vehicule", "marque_vehicule", "marque_modele"])]
            display(ood_focus.sort_values("unseen_test_levels", ascending=False))

            numeric_cols = [c for c in bundle.num_cols if c in bundle.X_test.columns]
            drift_rows = []
            for c in numeric_cols:
                tr = pd.to_numeric(bundle.X_train[c], errors="coerce")
                te = pd.to_numeric(bundle.X_test[c], errors="coerce")
                m_tr = float(np.nanmean(tr))
                m_te = float(np.nanmean(te))
                s_tr = float(np.nanstd(tr))
                drift_rows.append(
                    {
                        "diagnostic_type": "numeric_drift",
                        "feature": c,
                        "mean_train": m_tr,
                        "mean_test": m_te,
                        "std_train": s_tr,
                        "std_shift": float((m_te - m_tr) / max(s_tr, 1e-9)),
                    }
                )
            drift_df = pd.DataFrame(drift_rows).sort_values("std_shift", key=lambda s: s.abs(), ascending=False)
            display(drift_df.head(20))
            """
        ),
        md("## Segment bias from V1 OOF (if available)"),
        code(
            """
            seg = pd.DataFrame()
            oof_v1_path = ROOT / "artifacts" / "oof_predictions.parquet"
            ens_v1_path = ROOT / "artifacts" / "ensemble_weights_v1.json"

            if oof_v1_path.exists():
                oof_v1 = pd.read_parquet(oof_v1_path)
                if "run_id" not in oof_v1.columns:
                    oof_v1["run_id"] = (
                        oof_v1["engine"].astype(str) + "|"
                        + oof_v1["config_id"].astype(str) + "|"
                        + oof_v1["seed"].astype(int).astype(str) + "|"
                        + oof_v1["severity_mode"].astype(str) + "|"
                        + oof_v1["calibration"].astype(str)
                    )
                if ens_v1_path.exists():
                    meta = json.loads(ens_v1_path.read_text(encoding="utf-8"))
                    run_id = meta.get("best_single_run", oof_v1["run_id"].iloc[0])
                else:
                    run_id = oof_v1["run_id"].iloc[0]
                seg = compute_segment_bias_from_oof(train_raw, oof_v1, run_id=run_id, split_name="primary_time")
                print("segment diagnostics rows:", len(seg))
                display(seg.head(20))
            else:
                print("OOF v1 not found, skip segment bias diagnostic.")
            """
        ),
        md("## Risks and mitigation"),
        code(
            """
            risk_table = pd.DataFrame(
                [
                    {"risk": "Fine-grain categorical OOD", "impact": "Public/private shake-up", "mitigation": "hierarchy cp2/cp3, robust feature sets"},
                    {"risk": "Tail under-dispersion in severity", "impact": "RMSE degradation", "mitigation": "safe tail mapper + distribution audit"},
                    {"risk": "Client leakage", "impact": "over-optimistic CV", "mitigation": "secondary GroupKFold(id_client)"},
                    {"risk": "Single split selection bias", "impact": "unstable ranking", "mitigation": "primary+secondary+aux weighted selection"},
                ]
            )
            display(risk_table)
            """
        ),
        code(
            """
            diags = []
            if not ood.empty:
                diags.append(ood.assign(diagnostic_type="ood"))
            if "drift_df" in globals() and not drift_df.empty:
                diags.append(drift_df)
            if not seg.empty:
                diags.append(seg)
            diag_all = pd.concat(diags, ignore_index=True, sort=False) if diags else pd.DataFrame()
            diag_all.to_parquet(ARTIFACT_V2 / "segment_diagnostics_v2.parquet", index=False)

            meta = {
                "n_train": int(len(train_raw)),
                "n_test": int(len(test_raw)),
                "feature_sets": {k: {"n_features": len(v.feature_cols), "n_cat": len(v.cat_cols)} for k, v in feature_sets.items()},
                "splits": {k: sorted([int(fid) for fid in v.keys()]) for k, v in splits.items()},
                "train_only_cols": train_only,
                "test_only_cols": test_only,
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
            # 02 - Feature engineering V2

            Objectifs:
            - ablations par blocs de features,
            - comparaison `base_v2`, `robust_v2`, `compact_v2`,
            - impact sur RMSE et couverture de queue.
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
                build_prediction_distribution_table,
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
            primary_only = {"primary_time": splits["primary_time"]}

            experiments = [
                {"name": "base_with_te", "feature_set": "base_v2", "use_te": True},
                {"name": "base_no_te", "feature_set": "base_v2", "use_te": False},
                {"name": "robust_with_te", "feature_set": "robust_v2", "use_te": True},
                {"name": "compact_with_te", "feature_set": "compact_v2", "use_te": True},
            ]
            """
        ),
        code(
            """
            rows_run = []
            rows_dist = []
            cfg = V2_COARSE_CONFIGS["catboost"][0]

            for exp in experiments:
                spec = {
                    "feature_set": exp["feature_set"],
                    "engine": "catboost",
                    "family": "two_part_classic",
                    "severity_mode": "weighted_tail",
                    "tweedie_power": 1.5,
                    "config_id": cfg["config_id"],
                    "calibration_methods": ["none", "isotonic"],
                    "use_tail_mapper": True,
                    "use_target_encoding": exp["use_te"],
                    "target_encode_cols": ["code_postal", "cp3", "modele_vehicule", "marque_modele"],
                    "target_encoding_smoothing": 20.0,
                    "freq_params": cfg["freq_params"],
                    "sev_params": cfg["sev_params"],
                    "direct_params": cfg["direct_params"],
                    "split_names": ["primary_time"],
                }
                _, r, p = run_benchmark(spec, bundle=feature_sets[exp["feature_set"]], splits=primary_only, seed=42)
                r["experiment"] = exp["name"]
                rows_run.append(r)
                d = build_prediction_distribution_table(p)
                d["experiment"] = exp["name"]
                rows_dist.append(d)
                print("done", exp["name"], "rows", len(r))

            run_cmp = pd.concat(rows_run, ignore_index=True)
            dist_cmp = pd.concat(rows_dist, ignore_index=True)

            run_cmp.to_csv(ARTIFACT_V2 / "feature_set_comparison_v2.csv", index=False)
            display(
                run_cmp[run_cmp["level"] == "run"]
                .sort_values(["rmse_prime", "q99_ratio_pos"], ascending=[True, False])
                .head(20)
            )
            """
        ),
        code(
            """
            q = dist_cmp[(dist_cmp["sample"] == "oof") & (dist_cmp["split"] == "primary_time")][
                ["run_id", "experiment", "pred_q90", "pred_q99", "pred_q99_q90_ratio", "distribution_collapse_flag"]
            ]
            view = (
                run_cmp[run_cmp["level"] == "run"][["run_id", "experiment", "rmse_prime", "q99_ratio_pos"]]
                .merge(q, on=["run_id", "experiment"], how="left")
                .sort_values(["rmse_prime", "pred_q99_q90_ratio"], ascending=[True, False])
            )
            display(view.head(20))
            """
        ),
    ]
    return _nb(cells)


def build_nb_03() -> dict:
    cells = [
        md(
            """
            # 03 - Objective screening V2

            Screening moteur x famille x mode severite, pilote par:
            - RMSE prime,
            - q99_ratio_pos,
            - flags de collapse de distribution.
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

            rows_r = []
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
                    _, r, _ = run_benchmark(spec, bundle=bundle, splits=primary_only, seed=42)
                    rows_r.append(r)
                    print("[screen]", engine, fam["family"], fam["severity_mode"], fam["tweedie_power"])

            run_df = pd.concat(rows_r, ignore_index=True)
            run_df["screening_score"] = (
                run_df["rmse_prime"]
                + 2.5 * (1.0 - run_df["q99_ratio_pos"].fillna(0.0)).abs()
                + 2.0 * run_df["distribution_collapse_flag"].fillna(0.0)
            )
            run_df.to_csv(ARTIFACT_V2 / "objective_screening_v2.csv", index=False)
            display(run_df.sort_values("screening_score").head(30))
            """
        ),
        code(
            """
            matrix = (
                run_df[run_df["level"] == "run"]
                .pivot_table(
                    index=["engine", "family", "severity_mode"],
                    values=["rmse_prime", "q99_ratio_pos", "distribution_collapse_flag", "screening_score"],
                    aggfunc="mean",
                )
                .sort_values("screening_score")
            )
            display(matrix.head(30))
            """
        ),
    ]
    return _nb(cells)


def build_nb_04() -> dict:
    cells = [
        md(
            """
            # 04 - Modeling 3 engines V2.1

            Phases:
            - A: screening deja fait en notebook 03,
            - B: coarse search multi-configs,
            - C: finalists robustesse multi-seeds.
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
                build_prediction_distribution_table,
                V2_COARSE_CONFIGS,
            )

            DATA_DIR = ROOT / "data"
            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)

            RUN_FULL = False  # set True for overnight 10-14h
            QUICK_CFG_PER_ENGINE = 3
            """
        ),
        code(
            """
            train_raw, test_raw = load_train_test(DATA_DIR)
            feature_sets = prepare_feature_sets(train_raw, test_raw, rare_min_count=30, drop_identifiers=True)
            splits = build_split_registry(train_raw, n_blocks_time=5, n_splits_group=5, group_col="id_client")

            if RUN_FULL:
                cfg_per_engine = 6
                seeds = [42, 2026]
                feature_set_list = ["base_v2", "robust_v2", "compact_v2"]
                families = ["two_part_classic", "two_part_tweedie", "direct_tweedie"]
                sev_modes = ["classic", "weighted_tail", "winsorized"]
                tweedie_powers = [1.3, 1.5, 1.7]
                calibrations = ["none", "isotonic"]
            else:
                cfg_per_engine = QUICK_CFG_PER_ENGINE
                seeds = [42]
                feature_set_list = ["base_v2", "robust_v2", "compact_v2"]
                families = ["two_part_classic", "two_part_tweedie", "direct_tweedie"]
                sev_modes = ["classic", "weighted_tail"]
                tweedie_powers = [1.5]
                calibrations = ["none", "isotonic"]
            """
        ),
        code(
            """
            all_f, all_r, all_p = [], [], []

            for engine in ["catboost", "lightgbm", "xgboost"]:
                cfgs = V2_COARSE_CONFIGS[engine][:cfg_per_engine]
                for cfg in cfgs:
                    for fam in families:
                        for sev_mode in sev_modes:
                            if fam == "direct_tweedie" and sev_mode != "classic":
                                continue
                            powers = tweedie_powers if fam == "two_part_tweedie" else [1.5]
                            for tw_power in powers:
                                for seed in seeds:
                                    spec = {
                                        "feature_sets": feature_set_list,
                                        "engine": engine,
                                        "family": fam,
                                        "severity_mode": sev_mode,
                                        "tweedie_power": tw_power,
                                        "config_id": cfg["config_id"],
                                        "calibration_methods": calibrations,
                                        "use_tail_mapper": fam != "direct_tweedie",
                                        "use_target_encoding": True,
                                        "target_encode_cols": ["code_postal", "cp3", "modele_vehicule", "marque_modele"],
                                        "target_encoding_smoothing": 20.0,
                                        "freq_params": cfg["freq_params"],
                                        "sev_params": cfg["sev_params"],
                                        "direct_params": cfg["direct_params"],
                                    }
                                    print("[run]", engine, cfg["config_id"], fam, sev_mode, tw_power, "seed", seed)
                                    f, r, p = run_benchmark(spec, bundle=feature_sets, splits=splits, seed=seed)
                                    all_f.append(f)
                                    all_r.append(r)
                                    all_p.append(p)

            fold_df = pd.concat(all_f, ignore_index=True)
            run_df = pd.concat(all_r, ignore_index=True)
            pred_df = pd.concat(all_p, ignore_index=True)
            dist_df = build_prediction_distribution_table(pred_df)

            run_df.to_csv(ARTIFACT_V2 / "run_registry_v2.csv", index=False)
            pred_df.to_parquet(ARTIFACT_V2 / "oof_predictions_v2.parquet", index=False)
            dist_df.to_csv(ARTIFACT_V2 / "pred_distribution_audit_v2.csv", index=False)

            print("saved:", ARTIFACT_V2 / "run_registry_v2.csv")
            print("saved:", ARTIFACT_V2 / "oof_predictions_v2.parquet")
            print("saved:", ARTIFACT_V2 / "pred_distribution_audit_v2.csv")

            base_view = run_df[run_df["level"] == "run"].copy()
            sort_cols = [c for c in ["split", "rmse_prime", "selection_score"] if c in base_view.columns]
            display(base_view.sort_values(sort_cols).head(40))
            """
        ),
    ]
    return _nb(cells)


def build_nb_05() -> dict:
    cells = [
        md(
            """
            # 05 - Ensemble and robustness V2.1

            Objectifs:
            - selection multi-splits avec penalites de dispersion,
            - comparaison ensemble vs single,
            - stress tests shake-up (standard + tail stratified),
            - export selection report + poids d'ensemble.
            """
        ),
        code(
            """
            import sys
            import json
            from pathlib import Path
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
                optimize_non_negative_weights,
                rmse,
                select_final_models,
                simulate_public_private_shakeup_v2,
                build_model_cards,
                build_prediction_distribution_table,
            )

            ARTIFACT_V2 = ensure_dir(ROOT / DEFAULT_V2_DIR)
            """
        ),
        code(
            """
            run_df = pd.read_csv(ARTIFACT_V2 / "run_registry_v2.csv")
            oof = pd.read_parquet(ARTIFACT_V2 / "oof_predictions_v2.parquet")

            selection_report = select_final_models(run_df, risk_policy="stability_private", return_report=True)
            selection_report.to_csv(ARTIFACT_V2 / "selection_report_v2.csv", index=False)

            selected = select_final_models(run_df, risk_policy="stability_private", return_report=False)
            selected.to_csv(ARTIFACT_V2 / "selected_models_v2.csv", index=False)

            print("selection report rows:", len(selection_report))
            print("selected rows:", len(selected))
            display(selection_report.head(20))
            display(selected.head(20))
            """
        ),
        code(
            """
            def build_split_matrix(pred_df, split_name, run_ids):
                d = pred_df[(pred_df["is_test"] == 0) & (pred_df["split"] == split_name)].copy()
                d = d[d["run_id"].isin(run_ids)]
                wide = d.pivot_table(index="row_idx", columns="run_id", values="pred_prime", aggfunc="first")
                y = d.groupby("row_idx")["y_sev"].first()
                mask = wide.notna().all(axis=1) & y.notna()
                return wide.loc[mask], y.loc[mask]

            run_ids = selected["run_id"].drop_duplicates().tolist()
            Xp, yp = build_split_matrix(oof, "primary_time", run_ids)
            w = optimize_non_negative_weights(Xp.values, yp.values)
            weight_map = {rid: float(v) for rid, v in zip(Xp.columns.tolist(), w)}

            ens_primary_pred = Xp.values @ w
            ens_primary_rmse = rmse(yp.values, ens_primary_pred)

            single_scores = [(rid, rmse(yp.values, Xp[rid].values)) for rid in Xp.columns]
            best_single_run, best_single_rmse = sorted(single_scores, key=lambda x: x[1])[0]

            print("ensemble primary rmse:", ens_primary_rmse)
            print("best single primary rmse:", best_single_rmse, best_single_run)
            """
        ),
        code(
            """
            def eval_on_split(split_name, run_ids, weight_map, single_run):
                Xs, ys = build_split_matrix(oof, split_name, run_ids)
                if len(Xs) == 0:
                    return {"split": split_name, "rmse_ensemble": np.nan, "rmse_single": np.nan}
                ww = np.array([weight_map.get(c, 0.0) for c in Xs.columns], dtype=float)
                ww = ww / ww.sum() if ww.sum() > 0 else np.full(len(ww), 1.0 / len(ww))
                ens = Xs.values @ ww
                rmse_ens = rmse(ys.values, ens)
                rmse_s = rmse(ys.values, Xs[single_run].values) if single_run in Xs.columns else np.nan
                return {"split": split_name, "rmse_ensemble": float(rmse_ens), "rmse_single": float(rmse_s)}

            split_eval = pd.DataFrame(
                [
                    eval_on_split("primary_time", run_ids, weight_map, best_single_run),
                    eval_on_split("secondary_group", run_ids, weight_map, best_single_run),
                    eval_on_split("aux_blocked5", run_ids, weight_map, best_single_run),
                ]
            )
            display(split_eval)
            """
        ),
        code(
            """
            sh_ens = simulate_public_private_shakeup_v2(
                yp.values, ens_primary_pred, n_sim=2000, public_ratio=1/3, seed=42
            )
            sh_ens_tail = simulate_public_private_shakeup_v2(
                yp.values,
                ens_primary_pred,
                n_sim=2000,
                public_ratio=1/3,
                seed=42,
                stratified_tail=True,
                tail_quantile=0.9,
                tail_public_share=0.5,
            )
            sh_single = simulate_public_private_shakeup_v2(
                yp.values, Xp[best_single_run].values, n_sim=2000, public_ratio=1/3, seed=42
            )

            ens_gap_std = float(sh_ens["gap_public_minus_private"].std())
            ens_tail_gap_std = float(sh_ens_tail["gap_public_minus_private"].std())
            single_gap_std = float(sh_single["gap_public_minus_private"].std())

            sec_row = split_eval[split_eval["split"] == "secondary_group"].iloc[0]
            aux_row = split_eval[split_eval["split"] == "aux_blocked5"].iloc[0]
            gain_primary = float(best_single_rmse - ens_primary_rmse)
            degrade_secondary = float(sec_row["rmse_ensemble"] - sec_row["rmse_single"]) if pd.notna(sec_row["rmse_single"]) else 0.0
            degrade_aux = float(aux_row["rmse_ensemble"] - aux_row["rmse_single"]) if pd.notna(aux_row["rmse_single"]) else 0.0

            use_ensemble = (
                (gain_primary > 0.0)
                and (degrade_secondary <= 1.0)
                and (degrade_aux <= 1.0)
                and (ens_gap_std <= single_gap_std * 1.05)
                and (ens_tail_gap_std <= single_gap_std * 1.10)
            )
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
                "gain_primary": gain_primary,
                "degrade_secondary": degrade_secondary,
                "degrade_aux": degrade_aux,
                "ens_gap_std": ens_gap_std,
                "ens_tail_gap_std": ens_tail_gap_std,
                "single_gap_std": single_gap_std,
            }
            (ARTIFACT_V2 / "ensemble_weights_v2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            cards = build_model_cards(run_df, selected)
            cards.to_csv(ARTIFACT_V2 / "model_cards_v2.csv", index=False)

            pred_dist = build_prediction_distribution_table(oof)
            pred_dist.to_csv(ARTIFACT_V2 / "pred_distribution_audit_v2.csv", index=False)

            print("strategy:", strategy)
            print("saved:", ARTIFACT_V2 / "selection_report_v2.csv")
            print("saved:", ARTIFACT_V2 / "ensemble_weights_v2.json")
            """
        ),
    ]
    return _nb(cells)


def build_nb_06() -> dict:
    cells = [
        md(
            """
            # 06 - Submission report V2.1

            Refit 100% train (sans holdout 90/10), calibration/tail safe,
            generation de:
            - `submission_v2_robust.csv`
            - `submission_v2_single.csv`
            et audit pre-submission.
            """
        ),
        code(
            """
            import sys
            import json
            from pathlib import Path
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
                fit_full_predict_fulltrain,
                fit_calibrator,
                apply_calibrator,
                fit_tail_mapper_safe,
                apply_tail_mapper_safe,
                build_submission,
                compute_prediction_distribution_audit,
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
                engine: {cfg["config_id"]: cfg for cfg in cfgs}
                for engine, cfgs in V2_COARSE_CONFIGS.items()
            }

            dist_audit_path = ARTIFACT_V2 / "pred_distribution_audit_v2.csv"
            dist_audit = pd.read_csv(dist_audit_path) if dist_audit_path.exists() else pd.DataFrame()
            """
        ),
        code(
            """
            preds = {}
            for _, row in selected.iterrows():
                run_id = row["run_id"]
                engine = row["engine"]
                fs_name = row["feature_set"]
                family = row["family"]
                tweedie_power = float(row.get("tweedie_power", 1.5))
                config_id = row["config_id"]
                seed = int(row["seed"])
                severity_mode = row["severity_mode"]
                calibration = row["calibration"]
                tail_mapper_name = row["tail_mapper"]

                cfg = cfg_lookup[engine][config_id]
                bundle = feature_sets[fs_name]

                spec = {
                    "engine": engine,
                    "family": family,
                    "severity_mode": severity_mode,
                    "tweedie_power": tweedie_power,
                    "config_id": config_id,
                    "freq_params": cfg["freq_params"],
                    "sev_params": cfg["sev_params"],
                    "direct_params": cfg["direct_params"],
                    "use_target_encoding": True,
                    "target_encode_cols": ["code_postal", "cp3", "modele_vehicule", "marque_modele"],
                    "target_encoding_smoothing": 20.0,
                }

                out = fit_full_predict_fulltrain(spec=spec, bundle=bundle, seed=seed, complexity={})
                test_freq = out["test_freq"].copy()
                test_sev = out["test_sev"].copy()

                o = oof[(oof["is_test"] == 0) & (oof["split"] == "primary_time") & (oof["run_id"] == run_id)].copy()

                if calibration != "none" and len(o):
                    ok = o["pred_freq"].notna()
                    if ok.any():
                        cal = fit_calibrator(
                            o.loc[ok, "pred_freq"].to_numpy(),
                            o.loc[ok, "y_freq"].to_numpy(),
                            method=calibration,
                        )
                        test_freq = apply_calibrator(cal, test_freq, method=calibration)

                if tail_mapper_name != "none" and family != "direct_tweedie" and len(o):
                    pos = (o["y_freq"] == 1) & o["pred_sev"].notna()
                    if pos.sum() >= 80:
                        mapper = fit_tail_mapper_safe(
                            o.loc[pos, "pred_sev"].to_numpy(),
                            o.loc[pos, "y_sev"].to_numpy(),
                        )
                        sev_before = test_sev.copy()
                        test_sev = apply_tail_mapper_safe(mapper, test_sev)
                        std_ratio = float(np.std(test_sev) / max(np.std(sev_before), 1e-9))

                        q99_oof = float(np.nanquantile(o.loc[pos, "pred_sev"].to_numpy(), 0.99))
                        q99_test = float(np.nanquantile(test_sev, 0.99))
                        if (std_ratio < 0.70) or (q99_test < 0.60 * q99_oof):
                            test_sev = sev_before

                if family == "direct_tweedie":
                    pred = np.maximum(out["test_prime"], 0.0)
                else:
                    pred = np.maximum(test_freq * test_sev, 0.0)
                preds[run_id] = pred

            print("generated predictions for runs:", len(preds))
            """
        ),
        code(
            """
            strategy = ens_meta.get("strategy", "single")
            run_ids = ens_meta.get("run_ids", list(preds.keys()))
            weights = ens_meta.get("weights", {})
            best_single = ens_meta.get("best_single_run", run_ids[0])

            if strategy == "ensemble":
                mat = np.column_stack([preds[rid] for rid in run_ids if rid in preds])
                used_ids = [rid for rid in run_ids if rid in preds]
                w = np.array([weights.get(rid, 0.0) for rid in used_ids], dtype=float)
                if w.sum() <= 0:
                    w = np.full(len(used_ids), 1.0 / len(used_ids))
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

            print("saved:", ARTIFACT_V2 / "submission_v2_robust.csv")
            print("saved:", ARTIFACT_V2 / "submission_v2_single.csv")
            """
        ),
        code(
            """
            robust_audit = compute_prediction_distribution_audit(
                sub_robust["pred"].to_numpy(),
                run_id="submission_v2_robust",
                split="test",
                sample="test",
            )
            single_audit = compute_prediction_distribution_audit(
                sub_single["pred"].to_numpy(),
                run_id="submission_v2_single",
                split="test",
                sample="test",
            )

            q99_oof_ref = np.nan
            if not dist_audit.empty:
                rr = dist_audit[(dist_audit["sample"] == "oof") & (dist_audit["split"] == "primary_time")]
                rr = rr[rr["run_id"].isin(selected["run_id"].tolist())]
                if len(rr):
                    q99_oof_ref = float(rr["pred_q99"].median())

            submission_audit = {
                "n_rows_robust": int(len(sub_robust)),
                "n_rows_single": int(len(sub_single)),
                "columns_robust": sub_robust.columns.tolist(),
                "columns_single": sub_single.columns.tolist(),
                "robust_non_negative": bool((sub_robust["pred"] >= 0).all()),
                "single_non_negative": bool((sub_single["pred"] >= 0).all()),
                "robust_no_nan": bool(sub_robust["pred"].notna().all()),
                "single_no_nan": bool(sub_single["pred"].notna().all()),
                "robust_distribution": robust_audit,
                "single_distribution": single_audit,
                "q99_oof_reference_primary": None if not np.isfinite(q99_oof_ref) else q99_oof_ref,
                "q99_ratio_robust_test_over_oof_ref": None if not np.isfinite(q99_oof_ref) else float(robust_audit["pred_q99"] / max(q99_oof_ref, 1e-9)),
                "q99_ratio_single_test_over_oof_ref": None if not np.isfinite(q99_oof_ref) else float(single_audit["pred_q99"] / max(q99_oof_ref, 1e-9)),
            }

            (ARTIFACT_V2 / "submission_audit_v2.json").write_text(json.dumps(submission_audit, indent=2), encoding="utf-8")

            report = {
                "strategy": strategy,
                "n_models_selected": int(len(selected)),
                "best_single_run": best_single,
                "submission_robust": str(ARTIFACT_V2 / "submission_v2_robust.csv"),
                "submission_single": str(ARTIFACT_V2 / "submission_v2_single.csv"),
                "submission_audit": str(ARTIFACT_V2 / "submission_audit_v2.json"),
            }
            (ARTIFACT_V2 / "submission_report_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

            print("saved:", ARTIFACT_V2 / "submission_audit_v2.json")
            print("saved:", ARTIFACT_V2 / "submission_report_v2.json")
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

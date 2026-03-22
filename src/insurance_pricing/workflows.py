from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from insurance_pricing.training.config import TrainingConfig


def _build_benchmark_spec(cfg: TrainingConfig) -> dict[str, Any]:
    return {
        "feature_set": cfg.feature_set,
        "feature_sets": [cfg.feature_set],
        "engine": cfg.freq.engine,
        "family": cfg.sev.family,
        "severity_mode": cfg.sev.severity_mode,
        "tweedie_power": cfg.sev.tweedie_power,
        "config_id": cfg.run_name,
        "calibration_methods": [cfg.freq.calibration],
        "use_tail_mapper": bool(cfg.sev.use_tail_mapper),
        "use_target_encoding": bool(cfg.use_target_encoding),
        "target_encode_cols": list(cfg.target_encode_cols),
        "target_encoding_smoothing": float(cfg.target_encoding_smoothing),
        "freq_params": dict(cfg.freq.params),
        "sev_params": dict(cfg.sev.params),
        "direct_params": dict(cfg.sev.params),
        "split_names": list(cfg.split.split_names),
    }


def train_run(config_path: str) -> dict[str, Any]:
    from insurance_pricing.cv.integrity import build_splits, validate_split_integrity
    from insurance_pricing.data.datasets import build_feature_sets, load_datasets, select_bundle
    from insurance_pricing.data.schema import TARGET_SEV_COL
    from insurance_pricing.evaluation.metrics import summarize_prime_metrics
    from insurance_pricing.features.schema import build_feature_schema
    from insurance_pricing.models.calibration import fit_calibrator
    from insurance_pricing.models.frequency import fit_frequency_model
    from insurance_pricing.models.prime import PrimeModel
    from insurance_pricing.models.severity import fit_severity_model
    from insurance_pricing.models.tail import fit_tail_mapper
    from insurance_pricing.runtime.persistence import save_model_bundle
    from insurance_pricing.training.benchmark import run_benchmark
    from insurance_pricing.training.config import load_training_config
    from insurance_pricing.training.selection import score_multi_split, select_best_run

    cfg = load_training_config(config_path)
    train, test = load_datasets(cfg.data_dir)
    feature_sets = build_feature_sets(train, test, drop_identifiers=cfg.drop_identifiers)
    bundle = select_bundle(feature_sets, cfg.feature_set)
    splits = build_splits(train, cfg.split)
    split_report = validate_split_integrity(splits, train=train, group_col=cfg.split.group_col)

    spec = _build_benchmark_spec(cfg)
    fold_df, run_df, pred_df = run_benchmark(
        spec=spec, bundle=feature_sets, splits=splits, seed=cfg.seed
    )

    if run_df.empty:
        raise RuntimeError("run_benchmark returned empty run_df.")
    best_run = select_best_run(run_df, split="primary_time")
    run_id = str(best_run["run_id"])

    pred_sel = pred_df[
        (pred_df["run_id"].astype(str) == run_id)
        & (pred_df["split"].astype(str) == "primary_time")
        & (pred_df["is_test"] == 0)
    ].copy()

    freq_model = fit_frequency_model(
        bundle.X_train,
        bundle.y_freq.to_numpy(dtype=int),
        cat_cols=bundle.cat_cols,
        engine=cfg.freq.engine,
        seed=cfg.seed,
        params=cfg.freq.params,
    )
    sev_model = fit_severity_model(
        bundle.X_train,
        bundle.y_sev.to_numpy(dtype=float),
        bundle.y_freq.to_numpy(dtype=int),
        cat_cols=bundle.cat_cols,
        engine=cfg.sev.engine,
        family=cfg.sev.family,
        severity_mode=cfg.sev.severity_mode,
        tweedie_power=cfg.sev.tweedie_power,
        seed=cfg.seed,
        params=cfg.sev.params,
    )

    calibrator = None
    if str(cfg.freq.calibration).lower() != "none" and not pred_sel.empty:
        calibrator = fit_calibrator(
            y_true=pred_sel["y_freq"].to_numpy(dtype=int),
            p_pred=pred_sel["pred_freq"].to_numpy(dtype=float),
            method=cfg.freq.calibration,
        )

    tail_mapper = None
    if bool(cfg.sev.use_tail_mapper) and not pred_sel.empty:
        pos = pred_sel["y_sev"].to_numpy(dtype=float) > 0
        if int(np.sum(pos)) >= 80:
            tail_mapper = fit_tail_mapper(
                y_true_pos=pred_sel.loc[pos, "y_sev"].to_numpy(dtype=float),
                pred_sev_pos=pred_sel.loc[pos, "pred_sev"].to_numpy(dtype=float),
            )

    prime_model = PrimeModel(
        freq_model=freq_model,
        sev_model=sev_model,
        calibration_method=cfg.freq.calibration,
        calibrator=calibrator,
        tail_mapper=tail_mapper,
        non_negative=cfg.prime.non_negative,
    )

    # Metrics on train with final prime model
    pred_train = prime_model.predict_components(train)
    metrics: dict[str, Any] = summarize_prime_metrics(
        train[TARGET_SEV_COL].to_numpy(dtype=float),
        pred_train["pred_prime"].to_numpy(dtype=float),
    )
    if "q99_ratio_pos" in best_run:
        metrics["q99_ratio_pos"] = float(best_run.get("q99_ratio_pos", np.nan))
    metrics["selected_run_id"] = run_id
    metrics["selected_rmse_primary_time"] = float(best_run.get("rmse_prime", np.nan))

    feature_schema = build_feature_schema(bundle.feature_cols, bundle.cat_cols)
    artifacts = save_model_bundle(
        freq_model=freq_model,
        sev_model=sev_model,
        prime_model=prime_model,
        run_id=run_id,
        feature_schema=feature_schema,
        metrics=metrics,
        config=asdict(cfg),
        notes=cfg.notes,
    )

    # Save benchmark tables near model folder for traceability
    fold_df.to_csv(artifacts.run_dir / "benchmark_folds.csv", index=False)
    run_df.to_csv(artifacts.run_dir / "benchmark_runs.csv", index=False)
    pred_df.to_parquet(artifacts.run_dir / "benchmark_oof_predictions.parquet", index=False)
    score_multi_split(run_df).to_csv(
        artifacts.run_dir / "benchmark_multi_split_summary.csv", index=False
    )

    return {
        "run_id": artifacts.run_id,
        "run_dir": str(artifacts.run_dir),
        "metrics": metrics,
        "split_integrity": split_report,
    }


def evaluate_run(run_id: str) -> dict[str, Any]:
    from insurance_pricing.data.datasets import load_datasets
    from insurance_pricing.data.schema import TARGET_SEV_COL
    from insurance_pricing.evaluation.metrics import summarize_prime_metrics
    from insurance_pricing.runtime.persistence import load_model_bundle

    loaded = load_model_bundle(run_id)
    cfg = loaded["manifest"].get("config", {})
    data_dir = str(cfg.get("data_dir", "data"))
    train, test = load_datasets(data_dir)

    prime_model = loaded["prime_model"]
    pred_train = prime_model.predict_components(train)
    pred_test = prime_model.predict_components(test)

    y_train = train[TARGET_SEV_COL].to_numpy(dtype=float)
    m: dict[str, Any] = summarize_prime_metrics(
        y_train, pred_train["pred_prime"].to_numpy(dtype=float)
    )
    m["pred_test_mean"] = float(np.mean(pred_test["pred_prime"].to_numpy(dtype=float)))
    m["pred_test_q99"] = float(np.quantile(pred_test["pred_prime"].to_numpy(dtype=float), 0.99))
    m["run_id"] = str(run_id)

    run_dir = Path(loaded["run_dir"])
    (run_dir / "evaluation.json").write_text(pd.Series(m).to_json(indent=2), encoding="utf-8")
    return dict(m)


def predict_from_run(run_id: str, input_df: pd.DataFrame) -> pd.DataFrame:
    from insurance_pricing.inference.predict import predict_from_run as _predict_from_run

    return _predict_from_run(run_id, input_df)


def build_submission(run_id: str, test_df: pd.DataFrame) -> pd.DataFrame:
    from insurance_pricing.inference.predict import (
        build_submission_from_run as _build_submission,
    )

    return _build_submission(run_id, test_df)

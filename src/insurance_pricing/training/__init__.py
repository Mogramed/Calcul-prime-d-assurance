from insurance_pricing.cv.splits import (
    build_split_registry,
    export_split_artifacts_v2,
    validate_folds_disjoint,
    validate_group_disjoint,
)
from insurance_pricing.cv.integrity import build_splits, validate_split_integrity
from insurance_pricing.data.io import (
    build_targets,
    ensure_dir,
    load_json,
    load_train_test,
    save_json,
    validate_data_contract,
)
from insurance_pricing.data.datasets import build_feature_sets, load_datasets, select_bundle
from insurance_pricing.data.schema import (
    DEFAULT_V2_DIR,
    ID_COLS,
    INDEX_COL,
    TARGET_FREQ_COL,
    TARGET_SEV_COL,
    DatasetBundle,
)
from insurance_pricing.evaluation.diagnostics import (
    build_model_cards,
    build_prediction_distribution_table,
    compute_ood_diagnostics,
    compute_prediction_distribution_audit,
    compute_segment_bias_from_oof,
    simulate_public_private_shakeup_v2,
)
from insurance_pricing.evaluation.metrics import rmse, summarize_prime_metrics
from insurance_pricing.features.engineering import (
    add_engineered_features,
    add_engineered_features_v2,
    prepare_datasets,
    prepare_feature_sets,
)
from insurance_pricing.features.schema import build_feature_frame_for_inference, build_feature_schema
from insurance_pricing.models.calibration import apply_calibrator, fit_calibrator
from insurance_pricing.models.frequency import FrequencyModel, fit_frequency_model
from insurance_pricing.models.prime import PrimeModel
from insurance_pricing.models.severity import SeverityModel, fit_severity_model
from insurance_pricing.models.tail import apply_tail_mapper_safe, fit_tail_mapper_safe
from insurance_pricing.training.benchmark import make_run_id, run_benchmark
from insurance_pricing.training.config import (
    ModelSpecFreq,
    ModelSpecPrime,
    ModelSpecSev,
    SplitConfig,
    TrainingConfig,
    load_training_config,
    save_training_config,
)
from insurance_pricing.training.fulltrain import fit_full_predict_fulltrain
from insurance_pricing.training.presets import V2_COARSE_CONFIGS, V2_SCREENING_FAMILIES
from insurance_pricing.training.selection import (
    optimize_non_negative_weights,
    pick_top_configs,
    select_final_models,
)
from insurance_pricing.inference.submission import build_submission

# Temporary one-cycle compatibility for V1 notebook symbols.
# Removal target: next cleanup release after confirming notebooks no longer import
# COARSE_CONFIGS/run_cv_experiment/fit_full_two_part_predict/simulate_public_private_shakeup.
from insurance_pricing.legacy.v1_pipeline import (
    COARSE_CONFIGS,
    fit_full_two_part_predict,
    run_cv_experiment,
    simulate_public_private_shakeup,
)

__all__ = [
    "DEFAULT_V2_DIR",
    "ID_COLS",
    "INDEX_COL",
    "TARGET_FREQ_COL",
    "TARGET_SEV_COL",
    "DatasetBundle",
    "ensure_dir",
    "save_json",
    "load_json",
    "load_train_test",
    "build_targets",
    "validate_data_contract",
    "load_datasets",
    "build_feature_sets",
    "select_bundle",
    "rmse",
    "summarize_prime_metrics",
    "prepare_datasets",
    "prepare_feature_sets",
    "add_engineered_features",
    "add_engineered_features_v2",
    "build_feature_frame_for_inference",
    "build_feature_schema",
    "build_split_registry",
    "build_splits",
    "validate_split_integrity",
    "export_split_artifacts_v2",
    "validate_folds_disjoint",
    "validate_group_disjoint",
    "TrainingConfig",
    "SplitConfig",
    "ModelSpecFreq",
    "ModelSpecSev",
    "ModelSpecPrime",
    "load_training_config",
    "save_training_config",
    "FrequencyModel",
    "SeverityModel",
    "PrimeModel",
    "fit_frequency_model",
    "fit_severity_model",
    "fit_calibrator",
    "apply_calibrator",
    "fit_tail_mapper_safe",
    "apply_tail_mapper_safe",
    "make_run_id",
    "run_benchmark",
    "fit_full_predict_fulltrain",
    "build_prediction_distribution_table",
    "compute_prediction_distribution_audit",
    "compute_ood_diagnostics",
    "compute_segment_bias_from_oof",
    "simulate_public_private_shakeup_v2",
    "optimize_non_negative_weights",
    "select_final_models",
    "build_model_cards",
    "build_submission",
    "V2_COARSE_CONFIGS",
    "V2_SCREENING_FAMILIES",
    "COARSE_CONFIGS",
    "run_cv_experiment",
    "fit_full_two_part_predict",
    "simulate_public_private_shakeup",
    "pick_top_configs",
]

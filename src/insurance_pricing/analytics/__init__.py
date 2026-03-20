from insurance_pricing.analytics.association import compute_cramers_v_table
from insurance_pricing.analytics.dictionary import (
    build_data_dictionary,
    classify_columns,
)
from insurance_pricing.analytics.drift import (
    compute_drift_categorical_chi2,
    compute_drift_numeric_ks_psi,
)
from insurance_pricing.analytics.exports import export_analysis_tables
from insurance_pricing.analytics.loading import (
    DEFAULT_DS_DIR,
    load_project_datasets,
)
from insurance_pricing.analytics.model_diagnostics import (
    compute_calibration_table,
    compute_error_by_deciles,
    compute_oof_model_diagnostics,
)
from insurance_pricing.analytics.preprocessing import (
    build_cardinality_report,
    build_feature_blocks,
    build_feature_engineering_catalog,
    compute_preprocessing_recommendations,
)
from insurance_pricing.analytics.quality import (
    compute_missingness_report,
    compute_outlier_report,
    compute_rule_violations,
    detect_leakage_risk_columns,
)
from insurance_pricing.analytics.segmentation import (
    compute_linkage_from_distance,
    compute_gower_like_distance_sample,
    compute_segment_target_tables,
    fit_kmeans_exploration,
    fit_mixed_embedding_proxy,
    sample_for_exploration,
)

__all__ = [
    "DEFAULT_DS_DIR",
    "load_project_datasets",
    "build_data_dictionary",
    "classify_columns",
    "detect_leakage_risk_columns",
    "compute_missingness_report",
    "compute_rule_violations",
    "compute_outlier_report",
    "compute_drift_numeric_ks_psi",
    "compute_drift_categorical_chi2",
    "compute_cramers_v_table",
    "compute_segment_target_tables",
    "build_feature_blocks",
    "compute_preprocessing_recommendations",
    "sample_for_exploration",
    "compute_gower_like_distance_sample",
    "compute_linkage_from_distance",
    "fit_mixed_embedding_proxy",
    "fit_kmeans_exploration",
    "compute_oof_model_diagnostics",
    "compute_error_by_deciles",
    "compute_calibration_table",
    "export_analysis_tables",
    "build_cardinality_report",
    "build_feature_engineering_catalog",
]

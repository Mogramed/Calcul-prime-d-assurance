from __future__ import annotations

"""Optimized compatibility helpers built on top of modular V2 pipeline."""

from typing import Any, Mapping, Tuple  # noqa: E402

import pandas as pd  # noqa: E402

from insurance_pricing.evaluation.diagnostics import (  # noqa: E402
    build_model_cards,
    build_prediction_distribution_table,
    compute_ood_diagnostics,
    compute_prediction_distribution_audit,
    compute_segment_bias_from_oof,
    simulate_public_private_shakeup_v2,
)
from insurance_pricing.features.engineering import prepare_feature_sets  # noqa: E402
from insurance_pricing.inference.submission import build_submission  # noqa: E402
from insurance_pricing.models.calibration import apply_calibrator, fit_calibrator  # noqa: E402
from insurance_pricing.models.tail import apply_tail_mapper_safe, fit_tail_mapper_safe  # noqa: E402
from insurance_pricing.cv.splits import build_split_registry  # noqa: E402
from insurance_pricing.training.benchmark import run_benchmark  # noqa: E402
from insurance_pricing.training.fulltrain import fit_full_predict_fulltrain  # noqa: E402
from insurance_pricing.training.presets import (  # noqa: E402
    V2_COARSE_CONFIGS,
    V2_SCREENING_FAMILIES,
)
from insurance_pricing.training.selection import (  # noqa: E402
    optimize_non_negative_weights,
    select_final_models,
)


def run_benchmark_optimized(
    spec: Mapping[str, Any],
    bundle: Any,
    splits: Mapping[str, Mapping[int, tuple]],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop-in helper used by old optimized experiments."""
    return run_benchmark(spec=spec, bundle=bundle, splits=splits, seed=seed)


__all__ = [
    "run_benchmark_optimized",
    "run_benchmark",
    "fit_full_predict_fulltrain",
    "prepare_feature_sets",
    "build_split_registry",
    "fit_calibrator",
    "apply_calibrator",
    "fit_tail_mapper_safe",
    "apply_tail_mapper_safe",
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
]

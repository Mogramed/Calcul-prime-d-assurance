from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "rmse": ("insurance_pricing.evaluation.metrics", "rmse"),
    "_safe_auc": ("insurance_pricing.evaluation.metrics", "_safe_auc"),
    "make_tail_weights": ("insurance_pricing.evaluation.metrics", "make_tail_weights"),
    "compute_metric_row": ("insurance_pricing.evaluation.metrics", "compute_metric_row"),
    "build_prediction_distribution_table": (
        "insurance_pricing.evaluation.diagnostics",
        "build_prediction_distribution_table",
    ),
    "compute_prediction_distribution_audit": (
        "insurance_pricing.evaluation.diagnostics",
        "compute_prediction_distribution_audit",
    ),
    "compute_ood_diagnostics": (
        "insurance_pricing.evaluation.diagnostics",
        "compute_ood_diagnostics",
    ),
    "compute_segment_bias_from_oof": (
        "insurance_pricing.evaluation.diagnostics",
        "compute_segment_bias_from_oof",
    ),
    "simulate_public_private_shakeup": (
        "insurance_pricing.evaluation.diagnostics",
        "simulate_public_private_shakeup",
    ),
    "simulate_public_private_shakeup_v2": (
        "insurance_pricing.evaluation.diagnostics",
        "simulate_public_private_shakeup_v2",
    ),
    "build_model_cards": ("insurance_pricing.evaluation.diagnostics", "build_model_cards"),
}

__all__ = [
    "rmse",
    "_safe_auc",
    "make_tail_weights",
    "compute_metric_row",
    "build_prediction_distribution_table",
    "compute_prediction_distribution_audit",
    "compute_ood_diagnostics",
    "compute_segment_bias_from_oof",
    "simulate_public_private_shakeup",
    "simulate_public_private_shakeup_v2",
    "build_model_cards",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

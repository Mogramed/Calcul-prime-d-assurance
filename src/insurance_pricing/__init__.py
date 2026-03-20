from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "train_run": ("insurance_pricing.workflows", "train_run"),
    "evaluate_run": ("insurance_pricing.workflows", "evaluate_run"),
    "predict_from_run": ("insurance_pricing.workflows", "predict_from_run"),
    "build_submission": ("insurance_pricing.workflows", "build_submission"),
    "save_model_bundle": ("insurance_pricing.runtime.persistence", "save_model_bundle"),
    "load_model_bundle": ("insurance_pricing.runtime.persistence", "load_model_bundle"),
    "export_ds_tables_and_figures": (
        "insurance_pricing.runtime.ds_reporting",
        "export_ds_tables_and_figures",
    ),
    "TrainingConfig": ("insurance_pricing.training.config", "TrainingConfig"),
    "SplitConfig": ("insurance_pricing.training.config", "SplitConfig"),
    "ModelSpecFreq": ("insurance_pricing.training.config", "ModelSpecFreq"),
    "ModelSpecSev": ("insurance_pricing.training.config", "ModelSpecSev"),
    "ModelSpecPrime": ("insurance_pricing.training.config", "ModelSpecPrime"),
    "training": ("insurance_pricing.training", None),
    "analytics": ("insurance_pricing.analytics", None),
}

__version__ = "0.1.0"

__all__ = [
    "train_run",
    "evaluate_run",
    "predict_from_run",
    "build_submission",
    "save_model_bundle",
    "load_model_bundle",
    "export_ds_tables_and_figures",
    "TrainingConfig",
    "SplitConfig",
    "ModelSpecFreq",
    "ModelSpecSev",
    "ModelSpecPrime",
    "training",
    "analytics",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

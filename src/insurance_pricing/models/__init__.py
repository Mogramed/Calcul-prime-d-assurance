from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "FrequencyModel": ("insurance_pricing.models.frequency", "FrequencyModel"),
    "SeverityModel": ("insurance_pricing.models.severity", "SeverityModel"),
    "PrimeModel": ("insurance_pricing.models.prime", "PrimeModel"),
    "fit_frequency_model": ("insurance_pricing.models.frequency", "fit_frequency_model"),
    "fit_severity_model": ("insurance_pricing.models.severity", "fit_severity_model"),
    "fit_calibrator": ("insurance_pricing.models.calibration", "fit_calibrator"),
    "apply_calibrator": ("insurance_pricing.models.calibration", "apply_calibrator"),
    "crossfit_calibrate_oof": (
        "insurance_pricing.models.calibration",
        "crossfit_calibrate_oof",
    ),
    "fit_tail_mapper": ("insurance_pricing.models.tail", "fit_tail_mapper"),
    "apply_tail_mapper": ("insurance_pricing.models.tail", "apply_tail_mapper"),
    "fit_tail_mapper_safe": ("insurance_pricing.models.tail", "fit_tail_mapper_safe"),
    "apply_tail_mapper_safe": ("insurance_pricing.models.tail", "apply_tail_mapper_safe"),
    "crossfit_tail_mapper_oof": (
        "insurance_pricing.models.tail",
        "crossfit_tail_mapper_oof",
    ),
}

__all__ = [
    "FrequencyModel",
    "SeverityModel",
    "PrimeModel",
    "fit_frequency_model",
    "fit_severity_model",
    "fit_calibrator",
    "apply_calibrator",
    "crossfit_calibrate_oof",
    "fit_tail_mapper",
    "apply_tail_mapper",
    "fit_tail_mapper_safe",
    "apply_tail_mapper_safe",
    "crossfit_tail_mapper_oof",
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

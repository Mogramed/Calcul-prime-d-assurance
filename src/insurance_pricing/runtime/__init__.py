from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "RunArtifacts": ("insurance_pricing.runtime.persistence", "RunArtifacts"),
    "save_model_bundle": ("insurance_pricing.runtime.persistence", "save_model_bundle"),
    "load_model_bundle": ("insurance_pricing.runtime.persistence", "load_model_bundle"),
    "save_table": ("insurance_pricing.runtime.ds_reporting", "save_table"),
    "save_figure": ("insurance_pricing.runtime.ds_reporting", "save_figure"),
    "register_output": ("insurance_pricing.runtime.ds_reporting", "register_output"),
    "export_ds_tables_and_figures": (
        "insurance_pricing.runtime.ds_reporting",
        "export_ds_tables_and_figures",
    ),
}

__all__ = [
    "RunArtifacts",
    "save_model_bundle",
    "load_model_bundle",
    "save_table",
    "save_figure",
    "register_output",
    "export_ds_tables_and_figures",
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

from __future__ import annotations

import importlib
import sys
import pytest


def test_legacy_wrappers_removed():
    module_names = [
        "src.v1_pipeline",
        "src.v2_pipeline",
        "src.v2_pipeline_optimized",
        "src.v2_2_quick_workflow",
        "src.v2_3_dualtrack_quick",
        "src.v2_4_tail_recovery",
        "src.v2_4_1_tail_selection_fix",
        "src.ds_analysis_utils",
        "src.insurance_pricing.core",
        "src.insurance_pricing.modeling",
    ]
    for name in module_names:
        sys.modules.pop(name, None)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_modular_api_importable():
    for name in list(sys.modules):
        if name == "src.insurance_pricing" or name.startswith("src.insurance_pricing.experiments"):
            sys.modules.pop(name, None)
    mod = importlib.import_module("src.insurance_pricing")
    assert hasattr(mod, "train_run")
    assert hasattr(mod, "evaluate_run")
    assert hasattr(mod, "predict_from_run")
    assert hasattr(mod, "build_submission")
    assert "experiments" not in getattr(mod, "__all__", [])

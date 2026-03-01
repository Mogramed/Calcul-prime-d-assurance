from __future__ import annotations

import importlib
import numpy as np


def test_core_package_imports_are_healthy():
    modules = [
        "src.insurance_pricing",
        "src.insurance_pricing.api",
        "src.insurance_pricing.training",
        "src.insurance_pricing.analytics",
        "src.insurance_pricing.evaluation.diagnostics",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None


def test_diagnostics_import_no_partial_init_cycle():
    diagnostics = importlib.import_module("src.insurance_pricing.evaluation.diagnostics")
    assert hasattr(diagnostics, "build_prediction_distribution_table")
    assert hasattr(diagnostics, "compute_prediction_distribution_audit")


def test_fit_calibrator_accepts_positional_probs_y_and_keyword_method():
    calibration = importlib.import_module("src.insurance_pricing.models.calibration")
    probs = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    y_true = np.array([0, 0, 1, 1], dtype=int)
    model = calibration.fit_calibrator(probs, y_true, method="isotonic")
    out = calibration.apply_calibrator(model, probs, method="isotonic")
    assert out.shape == probs.shape

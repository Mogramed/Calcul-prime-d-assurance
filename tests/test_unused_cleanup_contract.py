from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.insurance_pricing.training.optimized",
        "src.insurance_pricing.runtime.common_io",
        "src.insurance_pricing.runtime.inference",
        "src.insurance_pricing.models.engines",
    ],
)
def test_removed_internal_modules_not_importable(module_name: str):
    sys.modules.pop(module_name, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)

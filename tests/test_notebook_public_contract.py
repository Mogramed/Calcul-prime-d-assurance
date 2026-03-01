from __future__ import annotations

import importlib
import json
from pathlib import Path


CONTRACT_PATH = Path("tests/contracts/notebook_public_contract.json")


def test_notebook_public_contract_symbols():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    modules = contract["modules"]
    for module_name, symbols in modules.items():
        mod = importlib.import_module(module_name)
        assert mod is not None
        for sym in symbols:
            assert hasattr(mod, sym), f"{module_name} is missing symbol: {sym}"


def test_notebook_public_contract_modules_importable():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for module_name in contract["modules"].keys():
        mod = importlib.import_module(module_name)
        assert mod is not None

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from insurance_pricing import train_run
from insurance_pricing.runtime.persistence import load_model_bundle


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TRAIN_TEST", "0") != "1",
    reason="Set RUN_HEAVY_TRAIN_TEST=1 to enable end-to-end training test.",
)
def test_train_produces_pickles(tmp_path: Path):
    cfg = {
        "run_name": "test_quick",
        "seed": 42,
        "data_dir": "data",
        "feature_set": "base_v2",
        "split": {
            "n_blocks_time": 3,
            "n_splits_group": 3,
            "group_col": "id_client",
            "split_names": ["primary_time"],
        },
        "freq": {
            "engine": "catboost",
            "calibration": "none",
            "params": {"iterations": 20, "depth": 4},
        },
        "sev": {
            "engine": "catboost",
            "family": "two_part_tweedie",
            "severity_mode": "classic",
            "tweedie_power": 1.3,
            "use_tail_mapper": False,
            "params": {"iterations": 25, "depth": 4},
        },
    }
    cfg_path = tmp_path / "quick_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    out = train_run(str(cfg_path))
    run_id = out["run_id"]
    loaded = load_model_bundle(run_id)
    assert loaded["run_dir"].exists()
    assert (loaded["run_dir"] / "model_freq.pkl").exists()
    assert (loaded["run_dir"] / "model_sev.pkl").exists()
    assert (loaded["run_dir"] / "model_prime.pkl").exists()
    test_df = pd.read_csv("data/test.csv").head(20)
    pred = loaded["prime_model"].predict_components(test_df)
    assert len(pred) == len(test_df)
    assert (pred["pred_prime"] >= 0).all()

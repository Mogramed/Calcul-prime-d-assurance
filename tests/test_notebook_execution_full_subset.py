from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_NOTEBOOK_FULL_TESTS", "0") != "1",
    reason="Set RUN_NOTEBOOK_FULL_TESTS=1 to run full notebook subset execution.",
)
def test_notebook_execution_full_subset():
    subset = [
        "notebooks/v1/01_eda_cv_design.ipynb",
        "notebooks/ds/07_ds_cadrage_qualite_cv.ipynb",
        "notebooks/ds/08_ds_eda_segmentation_preprocessing.ipynb",
        "notebooks/ds/09_ds_model_diagnostics_storytelling.ipynb",
    ]
    cmd = [
        sys.executable,
        "scripts/execute_notebooks_full.py",
        "--mode",
        "full",
        "--timeout",
        "900",
        "--notebooks",
        *subset,
    ]
    subprocess.run(cmd, check=True)

    summary_csv = Path("artifacts/runtime_cache/notebook_audit/full_run_summary.csv")
    assert summary_csv.exists(), "Missing full summary output"
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "Full summary is empty"
    failed = [r for r in rows if r.get("status") != "ok"]
    assert not failed, f"Notebook full subset failures: {[r.get('path') for r in failed]}"

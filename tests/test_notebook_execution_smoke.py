from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def test_notebook_execution_sanity_all():
    cmd = [
        sys.executable,
        "scripts/execute_notebooks_full.py",
        "--mode",
        "sanity",
        "--from-manifest",
        "--only-01-13",
        "--timeout",
        "240",
    ]
    subprocess.run(cmd, check=True)

    summary_csv = Path("artifacts/runtime_cache/notebook_audit/sanity_run_summary.csv")
    assert summary_csv.exists(), "Missing sanity summary output"
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "Sanity summary is empty"
    failed = [r for r in rows if r.get("status") != "ok"]
    assert not failed, f"Notebook sanity failures: {[r.get('path') for r in failed]}"

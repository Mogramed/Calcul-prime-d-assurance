from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_import_graph_contract():
    cmd = [sys.executable, "scripts/check_import_graph.py"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    detail = proc.stdout.strip() or proc.stderr.strip()
    assert proc.returncode == 0, f"Import graph contract failed.\n{detail}"

    # Keep machine-readable parse in the test for easier debug when running locally.
    report = json.loads(proc.stdout)
    assert report["status"] == "ok", report
    assert int(report.get("cycle_count", 1)) == 0, report


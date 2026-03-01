from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run_e2e(tmp_path: Path, *, ds_mode: str, config: str = "configs/train_smoke.json"):
    report_path = tmp_path / f"run_report_{ds_mode}.json"
    submission_path = tmp_path / f"submission_{ds_mode}.csv"
    cmd = [
        sys.executable,
        "scripts/run_end_to_end.py",
        "--config",
        config,
        "--ds-mode",
        ds_mode,
        "--run-report-output",
        str(report_path),
        "--submission-output",
        str(submission_path),
        "--strict-data-contract",
        "true",
        "--fail-if-missing-pickles",
        "true",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc, report_path, submission_path


def _assert_submission_contract(submission_path: Path) -> None:
    assert submission_path.exists(), f"Missing submission output: {submission_path}"
    sub = pd.read_csv(submission_path)
    assert list(sub.columns) == ["index", "pred"]
    assert len(sub) == 50000
    assert (sub["pred"] >= 0).all()


def _assert_model_bundle_present(run_dir: Path) -> None:
    required = [
        "model_freq.pkl",
        "model_sev.pkl",
        "model_prime.pkl",
        "feature_schema.json",
        "metrics.json",
        "manifest.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    assert not missing, f"Missing persisted model artifacts in {run_dir}: {missing}"


def test_run_end_to_end_smoke_ds_off(tmp_path: Path):
    proc, report_path, submission_path = _run_e2e(tmp_path, ds_mode="off")
    assert proc.returncode == 0, f"Script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert report_path.exists(), "Missing run_report.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["ds_exports"]["mode"] == "off"
    run_dir = Path(report["run"]["run_dir"])
    _assert_model_bundle_present(run_dir)
    _assert_submission_contract(submission_path)


def test_run_end_to_end_smoke_ds_quick(tmp_path: Path):
    proc, report_path, submission_path = _run_e2e(tmp_path, ds_mode="quick")
    assert proc.returncode == 0, f"Script failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert report_path.exists(), "Missing run_report.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["ds_exports"]["mode"] == "quick"
    assert int(report["ds_exports"]["n_outputs"]) >= 1
    run_dir = Path(report["run"]["run_dir"])
    _assert_model_bundle_present(run_dir)
    _assert_submission_contract(submission_path)


def test_run_end_to_end_fails_with_missing_config(tmp_path: Path):
    bad_cfg = "configs/does_not_exist.json"
    proc, report_path, _ = _run_e2e(tmp_path, ds_mode="off", config=bad_cfg)
    assert proc.returncode != 0
    assert "Config file not found" in (proc.stdout + proc.stderr)
    assert report_path.exists(), "Expected failure report to be written"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error"]["type"] == "FileNotFoundError"

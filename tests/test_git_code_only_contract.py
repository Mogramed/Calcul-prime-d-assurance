from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _git_ls_files() -> list[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    except Exception as exc:  # pragma: no cover - only when git metadata unavailable
        pytest.skip(f"git metadata unavailable: {exc}")
    return [line.strip() for line in out.splitlines() if line.strip()]


def test_code_only_git_policy():
    tracked = _git_ls_files()
    allowed_artifacts = {"artifacts/.gitkeep", "artifacts/README.md"}

    bad_artifacts = [
        p
        for p in tracked
        if p.startswith("artifacts/") and p not in allowed_artifacts
    ]
    assert not bad_artifacts, f"Tracked runtime artifacts detected: {bad_artifacts}"

    bad_noise = [
        p
        for p in tracked
        if "__pycache__/" in p
        or p.startswith(".pytest_cache/")
        or "/.pytest_cache/" in p
        or p.startswith("catboost_info/")
        or "/catboost_info/" in p
    ]
    assert not bad_noise, f"Tracked runtime/cache noise detected: {bad_noise}"


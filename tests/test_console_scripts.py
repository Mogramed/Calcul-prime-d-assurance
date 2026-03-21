from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_name",
    [
        "insurance-pricing-train",
        "insurance-pricing-evaluate",
        "insurance-pricing-predict",
        "insurance-pricing-make-submission",
        "insurance-pricing-api",
    ],
)
def test_console_scripts_smoke_with_uv(script_name: str):
    result = subprocess.run(
        [sys.executable, "-m", "uv", "run", "--no-sync", script_name, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout

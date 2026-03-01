from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_ROOT = Path("notebooks")


def _read_notebook_text(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        lines.extend(cell.get("source", []))
    return "".join(lines)


def test_no_legacy_or_pipelines_imports_in_notebooks():
    forbidden = [
        "from src.v1_pipeline import",
        "from src.v2_pipeline import",
        "from src.ds_analysis_utils import",
        "from src import v2_pipeline as",
        "from src import v2_2_quick_workflow",
        "from src import v2_3_dualtrack_quick",
        "from src import v2_4_tail_recovery",
        "from src import v2_4_1_tail_selection_fix",
        "src.insurance_pricing.pipelines",
        "src.insurance_pricing.workflows",
        "src.insurance_pricing.services",
        "from src.insurance_pricing.ds_reporting import",
        "from src.insurance_pricing.v1 import",
        "from src.insurance_pricing.v2 import",
        "from src.insurance_pricing.ds import",
        "from src.insurance_pricing.quick import",
    ]
    nb_files = [p for p in NOTEBOOK_ROOT.rglob("*.ipynb") if "archive" not in p.parts]
    assert nb_files, "No notebooks found"
    for path in nb_files:
        text = _read_notebook_text(path)
        for token in forbidden:
            assert token not in text, f"Forbidden import '{token}' in {path}"


def test_notebook_groups_use_new_public_modules():
    # DS notebooks
    ds_files = sorted((NOTEBOOK_ROOT / "ds").glob("*.ipynb"))
    for path in ds_files:
        text = _read_notebook_text(path)
        assert "from src.insurance_pricing.analytics import" in text, f"Missing analytics import in {path}"
        assert "import src.insurance_pricing.training as v2" in text, f"Missing training alias import in {path}"
        assert "from src.insurance_pricing.runtime.ds_reporting import" in text, f"Missing runtime ds_reporting import in {path}"

    # V1 notebooks
    v1_files = sorted((NOTEBOOK_ROOT / "v1").glob("*.ipynb"))
    for path in v1_files:
        text = _read_notebook_text(path)
        assert "from src.insurance_pricing.training import" in text, f"Missing training import in {path}"

    # V2 notebooks
    v2_files = sorted((NOTEBOOK_ROOT / "v2").glob("*.ipynb"))
    for path in v2_files:
        text = _read_notebook_text(path)
        assert "from src.insurance_pricing.training import" in text, f"Missing training import in {path}"

    # quick notebooks
    quick_expect = {
        "10_v2_2_quick_gap_diagnosis_retrain_submission.ipynb": "from src.insurance_pricing.experiments.quick import gap_diagnosis as wf",
        "11_v2_3_dualtrack_tweedie_gap_bridge_quick.ipynb": "from src.insurance_pricing.experiments.quick import dualtrack as wf",
        "12_v2_4_tail_recovery_lab.ipynb": "from src.insurance_pricing.experiments.quick import tail_recovery as tr",
        "13_v2_4_1_tail_selection_fix.ipynb": "from src.insurance_pricing.experiments.quick import tail_selection as fx",
    }
    for name, expected in quick_expect.items():
        path = NOTEBOOK_ROOT / "quick" / name
        text = _read_notebook_text(path)
        assert expected in text, f"Missing quick import in {path}"

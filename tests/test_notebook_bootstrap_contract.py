from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_ROOT = Path("notebooks")


def _first_code_cell_source(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            return "".join(cell.get("source", []))
    return ""


def test_notebook_bootstrap_contract_01_13():
    keep = {
        "01_eda_cv_design.ipynb",
        "02.0_modeling_3_engines.ipynb",
        "03.0_ensemble_submission.ipynb",
        "02_feature_engineering_v2.ipynb",
        "03_objective_screening_v2.ipynb",
        "04_modeling_3_engines_v2.ipynb",
        "04_modeling_3_engines_v2_optimized.ipynb",
        "05_ensemble_robustness_v2.ipynb",
        "06_submission_report_v2.ipynb",
        "07_ds_cadrage_qualite_cv.ipynb",
        "08_ds_eda_segmentation_preprocessing.ipynb",
        "09_ds_model_diagnostics_storytelling.ipynb",
        "10_v2_2_quick_gap_diagnosis_retrain_submission.ipynb",
        "11_v2_3_dualtrack_tweedie_gap_bridge_quick.ipynb",
        "12_v2_4_tail_recovery_lab.ipynb",
        "13_v2_4_1_tail_selection_fix.ipynb",
    }
    files = [p for p in NOTEBOOK_ROOT.rglob("*.ipynb") if p.name in keep]
    assert files, "No target notebooks found"

    required_tokens_modern = [
        "NOTEBOOK_DIR = Path.cwd().resolve()",
        "for _candidate in [NOTEBOOK_DIR, *NOTEBOOK_DIR.parents]:",
        "(_candidate / \"src\").exists() and (_candidate / \"data\").exists()",
        "sys.path.insert(0, str(ROOT))",
    ]
    required_tokens_legacy = [
        "ROOT = Path.cwd()",
        "while ROOT.name !=",
        "if not (ROOT / \"src\").exists():",
        "sys.path.insert(0, str(ROOT))",
    ]
    for path in files:
        src = _first_code_cell_source(path)
        assert src, f"Missing first code cell in {path}"
        has_modern = all(tok in src for tok in required_tokens_modern)
        has_legacy = all(tok in src for tok in required_tokens_legacy)
        assert has_modern or has_legacy, (
            f"Bootstrap contract missing in {path}. "
            "Expected modern or legacy root-resolution pattern."
        )

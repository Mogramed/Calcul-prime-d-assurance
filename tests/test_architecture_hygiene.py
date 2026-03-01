from __future__ import annotations

import ast
from pathlib import Path


SRC_ROOT = Path("src")


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_import_star_in_src():
    for path in SRC_ROOT.rglob("*.py"):
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "*", f"Wildcard import forbidden in {path}"


def test_no_pipelines_module_or_imports():
    assert not (SRC_ROOT / "insurance_pricing" / "pipelines").exists(), "pipelines layer must be removed"
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "insurance_pricing.pipelines" not in text, f"Stale pipelines import found in {path}"

def test_no_bare_insurance_pricing_imports():
    # Enforce explicit package root used by this repository: src.insurance_pricing
    for path in (SRC_ROOT / "insurance_pricing").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from insurance_pricing" not in text, f"Bare import found in {path}"
        assert "import insurance_pricing" not in text, f"Bare import found in {path}"


def test_no_workflows_or_services_or_versioned_modules_in_active_code():
    active_root = SRC_ROOT / "insurance_pricing"
    for forbidden_dir in ["workflows", "services", "v1", "v2", "ds", "quick", "core", "modeling"]:
        assert not (active_root / forbidden_dir).exists(), f"Deprecated module dir should be removed: {forbidden_dir}"

    for path in active_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "\\legacy\\" in str(path):
            continue
        assert "insurance_pricing.workflows" not in text, f"Stale workflows import found in {path}"
        assert "insurance_pricing.services" not in text, f"Stale services import found in {path}"
        assert "insurance_pricing.v1" not in text, f"Stale v1 import found in {path}"
        assert "insurance_pricing.v2" not in text, f"Stale v2 import found in {path}"
        assert "insurance_pricing.ds" not in text, f"Stale ds import found in {path}"
        assert "insurance_pricing.quick" not in text, f"Stale quick import found in {path}"


def test_no_core_or_modeling_imports_in_canonical_code():
    active_root = SRC_ROOT / "insurance_pricing"
    canonical_dirs = [
        "data",
        "features",
        "cv",
        "models",
        "training",
        "evaluation",
        "inference",
        "analytics",
        "experiments",
        "runtime",
    ]
    for d in canonical_dirs:
        base = active_root / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "insurance_pricing.core" not in text, f"Canonical code must not import core: {path}"
            assert "insurance_pricing.modeling" not in text, f"Canonical code must not import modeling: {path}"


def test_no_quick_imports_in_canonical_flow():
    active_root = SRC_ROOT / "insurance_pricing"
    canonical_non_experimental_dirs = [
        "data",
        "features",
        "cv",
        "models",
        "training",
        "evaluation",
        "inference",
        "analytics",
        "runtime",
    ]
    for d in canonical_non_experimental_dirs:
        base = active_root / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "insurance_pricing.experiments.quick" not in text, (
                f"Canonical flow must not depend on archived quick modules: {path}"
            )


def test_no_legacy_imports_in_canonical_flow_except_training_compat():
    active_root = SRC_ROOT / "insurance_pricing"
    canonical_dirs = [
        "data",
        "features",
        "cv",
        "models",
        "training",
        "evaluation",
        "inference",
        "analytics",
        "runtime",
    ]
    for d in canonical_dirs:
        base = active_root / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            rel = path.relative_to(active_root).as_posix()
            if rel == "training/__init__.py" or rel.startswith("training/"):
                # Temporary one-cycle compatibility exception.
                continue
            text = path.read_text(encoding="utf-8")
            assert "src.insurance_pricing.legacy" not in text, (
                f"Canonical code must not import legacy modules outside training compat: {path}"
            )


def test_canonical_file_size_limit():
    canonical_dirs = [
        "data",
        "features",
        "cv",
        "models",
        "training",
        "evaluation",
        "inference",
        "analytics",
        "experiments",
        "runtime",
    ]
    active_root = SRC_ROOT / "insurance_pricing"
    too_large = []
    for d in canonical_dirs:
        base = active_root / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            n_lines = len(path.read_text(encoding="utf-8").splitlines())
            if n_lines > 350:
                too_large.append((path, n_lines))
    assert not too_large, f"Files above 350 lines in canonical code: {too_large}"


def test_no_import_only_modules_in_src_root():
    # No legacy facade modules should remain at src root.
    forbidden = {
        "v1_pipeline.py",
        "v2_pipeline.py",
        "v2_pipeline_optimized.py",
        "ds_analysis_utils.py",
        "v2_2_quick_workflow.py",
        "v2_3_dualtrack_quick.py",
        "v2_4_tail_recovery.py",
        "v2_4_1_tail_selection_fix.py",
    }
    present = {p.name for p in SRC_ROOT.glob("*.py")}
    assert not (present & forbidden), f"Legacy wrapper modules still present: {sorted(present & forbidden)}"

    # Remaining root python modules should not be import-only.
    root_files = [p for p in SRC_ROOT.glob("*.py") if p.name != "__init__.py"]
    for path in root_files:
        tree = _parse(path)
        body = [n for n in tree.body if not isinstance(n, ast.Expr) or not isinstance(getattr(n, "value", None), ast.Constant)]
        assert len(body) > 1, f"Import-only module detected: {path}"


def test_removed_orphan_modules_absent():
    forbidden_files = [
        SRC_ROOT / "insurance_pricing" / "training" / "optimized.py",
        SRC_ROOT / "insurance_pricing" / "runtime" / "common_io.py",
        SRC_ROOT / "insurance_pricing" / "runtime" / "inference.py",
        SRC_ROOT / "insurance_pricing" / "models" / "engines",
        SRC_ROOT / "insurance_pricing" / "experiments" / "quick" / "common.py",
    ]
    for path in forbidden_files:
        assert not path.exists(), f"Deprecated module should not exist: {path}"


def test_no_duplicate_safe_read_helpers_in_legacy_quick_impls():
    quick_dir = SRC_ROOT / "insurance_pricing" / "legacy" / "quick"
    if not quick_dir.exists():
        return
    impl_files = [p for p in quick_dir.glob("*_impl.py") if p.is_file()]
    forbidden_tokens = [
        "def _safe_read_csv(",
        "def _safe_read_parquet(",
        "def _safe_read_json(",
        "def _safe_float(",
        "def _rmse(",
    ]
    for path in impl_files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text, f"Duplicated helper {token} must be centralized in legacy/quick/common.py ({path})"

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "artifacts",
    "catboost_info",
    "notebooks",
    "__pycache__",
}
DEPRECATED_IMPORT_PATTERN = re.compile(
    r"(^|\n)\s*(from|import)\s+src\.insurance_pricing\b",
)


def _iter_python_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*.py") if not any(part in IGNORED_DIRS for part in path.parts)
    ]


def test_repo_code_no_longer_uses_src_import_paths():
    offenders: list[str] = []
    for path in _iter_python_files():
        content = path.read_text(encoding="utf-8")
        if DEPRECATED_IMPORT_PATTERN.search(content):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"Found deprecated src imports in: {offenders}"


def test_package_imports_are_available():
    import insurance_pricing as package
    from insurance_pricing.api import create_app

    assert callable(package.train_run)
    assert callable(package.predict_from_run)
    assert callable(create_app)

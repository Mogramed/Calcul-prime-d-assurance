from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def _scan_dirs(name: str) -> list[Path]:
    return [p for p in ROOT.rglob(name) if p.is_dir()]


def _git_tracked_relpaths() -> set[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    except Exception:
        return set()
    return {line.strip().replace("/", "\\") for line in out.splitlines() if line.strip()}


def main() -> None:
    issues: list[str] = []

    required = [
        ROOT / "src" / "insurance_pricing" / "data",
        ROOT / "src" / "insurance_pricing" / "features",
        ROOT / "src" / "insurance_pricing" / "cv",
        ROOT / "src" / "insurance_pricing" / "models",
        ROOT / "src" / "insurance_pricing" / "training",
        ROOT / "src" / "insurance_pricing" / "evaluation",
        ROOT / "src" / "insurance_pricing" / "inference",
        ROOT / "src" / "insurance_pricing" / "analytics",
        ROOT / "src" / "insurance_pricing" / "experiments",
        ROOT / "src" / "insurance_pricing" / "experiments" / "quick",
        ROOT / "src" / "insurance_pricing" / "legacy",
        ROOT / "src" / "insurance_pricing" / "runtime",
        ROOT / "notebooks" / "v1",
        ROOT / "notebooks" / "v2",
        ROOT / "notebooks" / "ds",
        ROOT / "notebooks" / "quick",
        ROOT / "notebooks" / "archive",
        ROOT / "README_architecture.md",
        ROOT / "artifacts" / "README.md",
    ]
    for p in required:
        if not p.exists():
            issues.append(f"Missing required path: {p.relative_to(ROOT)}")

    removed_paths = [
        ROOT / "src" / "insurance_pricing" / "pipelines",
        ROOT / "src" / "insurance_pricing" / "workflows",
        ROOT / "src" / "insurance_pricing" / "services",
        ROOT / "src" / "insurance_pricing" / "core",
        ROOT / "src" / "insurance_pricing" / "modeling",
        ROOT / "src" / "insurance_pricing" / "v1",
        ROOT / "src" / "insurance_pricing" / "v2",
        ROOT / "src" / "insurance_pricing" / "ds",
        ROOT / "src" / "insurance_pricing" / "quick",
        ROOT / "src" / "insurance_pricing" / "training" / "optimized.py",
        ROOT / "src" / "insurance_pricing" / "runtime" / "common_io.py",
        ROOT / "src" / "insurance_pricing" / "runtime" / "inference.py",
        ROOT / "src" / "insurance_pricing" / "models" / "engines",
        ROOT / "src" / "insurance_pricing" / "experiments" / "quick" / "common.py",
        ROOT / "src" / "v1_pipeline.py",
        ROOT / "src" / "v2_pipeline.py",
        ROOT / "src" / "v2_pipeline_optimized.py",
        ROOT / "src" / "ds_analysis_utils.py",
        ROOT / "src" / "v2_2_quick_workflow.py",
        ROOT / "src" / "v2_3_dualtrack_quick.py",
        ROOT / "src" / "v2_4_tail_recovery.py",
        ROOT / "src" / "v2_4_1_tail_selection_fix.py",
    ]
    for p in removed_paths:
        if p.exists():
            issues.append(f"Deprecated path should not exist: {p.relative_to(ROOT)}")

    manifest = NOTEBOOKS / "MANIFEST.csv"
    if not manifest.exists():
        issues.append("Missing notebooks/MANIFEST.csv")

    top_level_ipynb = sorted(NOTEBOOKS.glob("*.ipynb"))
    if top_level_ipynb:
        issues.append(f"Top-level notebooks found: {[p.name for p in top_level_ipynb]}")

    tracked = _git_tracked_relpaths()
    tracked_runtime_noise = []
    tracked_artifact_files = []
    for rel in tracked:
        if not (ROOT / rel).exists():
            continue
        if "\\catboost_info\\" in f"\\{rel}\\" and not rel.startswith("artifacts\\"):
            tracked_runtime_noise.append(rel)
        if "\\__pycache__\\" in f"\\{rel}\\" and not rel.startswith("artifacts\\"):
            tracked_runtime_noise.append(rel)
        if "\\.pytest_cache\\" in f"\\{rel}\\" and not rel.startswith("artifacts\\"):
            tracked_runtime_noise.append(rel)
        if rel.startswith("artifacts\\"):
            if rel not in {"artifacts\\README.md", "artifacts\\.gitkeep"}:
                tracked_artifact_files.append(rel)
    if tracked_runtime_noise:
        tracked_runtime_noise = sorted(set(tracked_runtime_noise))
        issues.append(f"Tracked runtime/cache files detected: {tracked_runtime_noise}")
    if tracked_artifact_files:
        tracked_artifact_files = sorted(set(tracked_artifact_files))
        issues.append(
            "Tracked artifact outputs detected (code-only policy): "
            f"{tracked_artifact_files}"
        )

    status = "ok" if not issues else "failed"
    report = {
        "status": status,
        "issues": issues,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

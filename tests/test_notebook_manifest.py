from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_notebook_manifest_paths_exist():
    manifest = Path("notebooks/MANIFEST.csv")
    assert manifest.exists()
    df = pd.read_csv(manifest)
    assert not df.empty
    assert {"path", "notebook", "section"}.issubset(df.columns)
    for p in df["path"].astype(str):
        assert Path(p).exists(), f"Missing notebook path in MANIFEST: {p}"


def test_no_top_level_notebook_left_in_notebooks_root():
    root = Path("notebooks")
    top_level = sorted(root.glob("*.ipynb"))
    assert top_level == [], f"Top-level notebooks should be grouped in folders: {top_level}"

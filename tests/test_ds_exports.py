from __future__ import annotations

from pathlib import Path

from src.insurance_pricing.runtime.ds_reporting import export_ds_tables_and_figures


def test_ds_exports_create_index_and_outputs():
    out = export_ds_tables_and_figures(mode="quick")
    assert isinstance(out, dict)
    assert Path("artifacts/ds/ds_outputs_index.csv").exists()
    assert Path("artifacts/ds/tables").exists()
    assert Path("artifacts/ds/figures").exists()

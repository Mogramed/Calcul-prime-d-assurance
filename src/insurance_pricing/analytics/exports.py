from __future__ import annotations

from pathlib import Path

import pandas as pd

from insurance_pricing.data.io import ensure_dir

def export_analysis_tables(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = ensure_dir(out_dir)
    for name, df in tables.items():
        if df is None:
            continue
        if isinstance(df, pd.DataFrame):
            df.to_csv(out / f"{name}.csv", index=False)
        else:
            raise TypeError(f"Unsupported table type for {name}: {type(df)}")


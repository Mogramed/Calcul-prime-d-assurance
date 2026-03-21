from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

DS_ROOT = Path("artifacts") / "ds"
DS_TABLES = DS_ROOT / "tables"
DS_FIGURES = DS_ROOT / "figures"
DS_INDEX = DS_ROOT / "ds_outputs_index.csv"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _register_output(
    *,
    output_type: str,
    notebook: str,
    name: str,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    DS_ROOT.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": _utc_now(),
        "type": output_type,
        "notebook": notebook,
        "name": name,
        "path": str(path),
        **(extra or {}),
    }
    new = pd.DataFrame([row])
    if DS_INDEX.exists():
        old = pd.read_csv(DS_INDEX)
        out = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        out = new
    out.to_csv(DS_INDEX, index=False)
    return DS_INDEX


def save_table(df: pd.DataFrame, name: str, notebook: str) -> Path:
    DS_TABLES.mkdir(parents=True, exist_ok=True)
    nb = notebook.replace(".ipynb", "")
    out_dir = DS_TABLES / nb
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    _register_output(
        output_type="table", notebook=nb, name=name, path=path, extra={"n_rows": int(len(df))}
    )
    return path


def save_figure(fig, name: str, notebook: str, dpi: int = 150) -> Path:
    DS_FIGURES.mkdir(parents=True, exist_ok=True)
    nb = notebook.replace(".ipynb", "")
    out_dir = DS_FIGURES / nb
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    _register_output(
        output_type="figure", notebook=nb, name=name, path=path, extra={"dpi": int(dpi)}
    )
    return path


def register_output(output_type: str, notebook: str, name: str, path: str | Path) -> Path:
    p = Path(path)
    return _register_output(output_type=output_type, notebook=notebook, name=name, path=p)


def _plot_top_missing(miss_df: pd.DataFrame, notebook: str) -> Path | None:
    if miss_df.empty:
        return None
    cols = [c for c in ["column", "missing_rate_train"] if c in miss_df.columns]
    if len(cols) < 2:
        return None
    d = miss_df.copy()
    d["missing_rate_train"] = pd.to_numeric(d["missing_rate_train"], errors="coerce").fillna(0.0)
    d = d.sort_values("missing_rate_train", ascending=False).head(20)
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(d["column"].astype(str), d["missing_rate_train"])
    ax.set_title("Top Missing Rate (train)")
    ax.set_xlabel("missing_rate_train")
    ax.invert_yaxis()
    path = save_figure(fig, "missingness_top20", notebook, dpi=160)
    plt.close(fig)
    return path


def _plot_top_psi(num_drift_df: pd.DataFrame, notebook: str) -> Path | None:
    if num_drift_df.empty or "psi" not in num_drift_df.columns:
        return None
    d = num_drift_df.copy()
    d["psi"] = pd.to_numeric(d["psi"], errors="coerce")
    if "column" not in d.columns:
        return None
    d = d.dropna(subset=["psi"]).sort_values("psi", ascending=False).head(20)
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(d["column"].astype(str), d["psi"])
    ax.set_title("Top Numeric Drift PSI")
    ax.set_xlabel("PSI")
    ax.invert_yaxis()
    path = save_figure(fig, "drift_numeric_top_psi", notebook, dpi=160)
    plt.close(fig)
    return path


def _plot_q99_ratio(metrics_df: pd.DataFrame, notebook: str) -> Path | None:
    if metrics_df.empty or "q99_ratio_pos" not in metrics_df.columns:
        return None
    d = metrics_df.copy()
    d["q99_ratio_pos"] = pd.to_numeric(d["q99_ratio_pos"], errors="coerce")
    d = d.dropna(subset=["q99_ratio_pos"]).head(20)
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d.index, d["q99_ratio_pos"], marker="o")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("q99_ratio_pos")
    ax.set_ylabel("ratio")
    ax.set_xlabel("row")
    path = save_figure(fig, "q99_ratio_pos", notebook, dpi=160)
    plt.close(fig)
    return path


def export_ds_tables_and_figures(mode: str = "full") -> dict[str, Path]:
    DS_ROOT.mkdir(parents=True, exist_ok=True)
    DS_TABLES.mkdir(parents=True, exist_ok=True)
    DS_FIGURES.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    # Export/copy top-level CSV tables into structured DS tables
    for csv_path in sorted(DS_ROOT.glob("*.csv")):
        if csv_path.name == DS_INDEX.name:
            continue
        nb = "export_ds_outputs"
        df = pd.read_csv(csv_path)
        target = save_table(df, csv_path.stem, nb)
        out[f"table::{csv_path.stem}"] = target

    # Standard figures from key tables
    missing = (
        pd.read_csv(DS_ROOT / "missingness_report.csv")
        if (DS_ROOT / "missingness_report.csv").exists()
        else pd.DataFrame()
    )
    drift_num = (
        pd.read_csv(DS_ROOT / "drift_numeric_ks_psi.csv")
        if (DS_ROOT / "drift_numeric_ks_psi.csv").exists()
        else pd.DataFrame()
    )
    metrics = (
        pd.read_csv(DS_ROOT / "oof_model_diagnostics_metrics.csv")
        if (DS_ROOT / "oof_model_diagnostics_metrics.csv").exists()
        else pd.DataFrame()
    )

    p1 = _plot_top_missing(missing, "export_ds_outputs")
    p2 = _plot_top_psi(drift_num, "export_ds_outputs")
    p3 = _plot_q99_ratio(metrics, "export_ds_outputs")
    if p1 is not None:
        out["figure::missingness_top20"] = p1
    if p2 is not None:
        out["figure::drift_numeric_top_psi"] = p2
    if p3 is not None:
        out["figure::q99_ratio_pos"] = p3
    _write_outputs_readme()
    return out


def _write_outputs_readme() -> Path:
    lines: list[str] = []
    lines.append("# DS Outputs Catalog")
    lines.append("")
    lines.append(f"- generated_utc: {_utc_now()}")
    lines.append("- scope: artifacts/ds/tables + artifacts/ds/figures + ds_outputs_index.csv")
    lines.append("")
    idx = pd.read_csv(DS_INDEX) if DS_INDEX.exists() else pd.DataFrame()
    if idx.empty:
        lines.append("No indexed outputs found.")
    else:
        lines.append("## Latest Outputs (top 50)")
        lines.append("")
        cols = [
            c for c in ["timestamp_utc", "type", "notebook", "name", "path"] if c in idx.columns
        ]
        d = idx[cols].tail(50)
        lines.append(d.to_markdown(index=False))
    out = DS_ROOT / "README_outputs.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

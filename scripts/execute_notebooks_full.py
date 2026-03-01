from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

USE_NBCLIENT = True
try:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
except Exception:
    USE_NBCLIENT = False
    class CellExecutionError(Exception):
        """Fallback placeholder when nbclient is unavailable."""


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "notebooks" / "MANIFEST.csv"
OUT_DIR = ROOT / "artifacts" / "runtime_cache" / "notebook_audit"
FAIL_DIR = OUT_DIR / "failures"
EXEC_DIR = OUT_DIR / "executed"


@dataclass
class NotebookResult:
    notebook: str
    path: str
    mode: str
    status: str
    duration_sec: float
    failed_cell: int | None
    error_type: str | None
    error_message: str | None
    traceback_path: str | None
    executed_path: str | None
    memory_mb: float | None = None


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    EXEC_DIR.mkdir(parents=True, exist_ok=True)


def _iter_manifest_paths() -> list[Path]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
    rows: list[Path] = []
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            p = ROOT / str(r["path"])
            if p.suffix == ".ipynb":
                rows.append(p)
    return rows


def _filter_01_13(paths: Iterable[Path]) -> list[Path]:
    keep = set(
        [
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
        ]
    )
    return [p for p in paths if p.name in keep]


def _load_notebook(path: Path):
    if USE_NBCLIENT:
        return nbformat.read(path, as_version=4)
    return json.loads(path.read_text(encoding="utf-8"))


def _trim_for_sanity(nb):
    if USE_NBCLIENT:
        code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
        if not code_cells:
            return nb
        nb.cells = [code_cells[0]]
        return nb

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code_cells:
        return nb
    nb["cells"] = [code_cells[0]]
    return nb


def _execute_notebook(path: Path, mode: str, timeout: int) -> NotebookResult:
    started = time.time()
    nb = _load_notebook(path)
    if mode == "sanity":
        nb = _trim_for_sanity(nb)

    failed_cell = None
    err_type = None
    err_msg = None
    tb_path = None
    status = "ok"
    out_path = EXEC_DIR / f"{path.stem}.{mode}.executed.ipynb"

    try:
        if USE_NBCLIENT:
            client = NotebookClient(
                nb,
                timeout=timeout,
                kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}},
                allow_errors=False,
            )
            client.execute()
            nbformat.write(nb, out_path)
        else:
            def _display(*args, **kwargs):
                for obj in args:
                    print(obj)

            ns: dict[str, object] = {"__name__": "__main__", "display": _display}
            old_cwd = Path.cwd()
            os.chdir(ROOT)
            try:
                for i, cell in enumerate(nb.get("cells", []), start=1):
                    if cell.get("cell_type") != "code":
                        continue
                    src = "".join(cell.get("source", []))
                    # Best effort fallback executor: strip notebook magics/shell escapes.
                    cleaned = []
                    for ln in src.splitlines():
                        ls = ln.lstrip()
                        if ls.startswith("%") or ls.startswith("!"):
                            continue
                        cleaned.append(ln)
                    src = "\n".join(cleaned)
                    if src.strip():
                        exec(compile(src, f"{path}#cell{i}", "exec"), ns, ns)
                out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            finally:
                os.chdir(old_cwd)
    except CellExecutionError as exc:
        status = "failed"
        err_type = "CellExecutionError"
        err_msg = str(exc).splitlines()[-1][:1000]
        if USE_NBCLIENT:
            for i, cell in enumerate(nb.cells):
                if cell.get("cell_type") == "code":
                    for out in cell.get("outputs", []):
                        if out.get("output_type") == "error":
                            failed_cell = i + 1
                            break
                if failed_cell is not None:
                    break
        tb_path = str(FAIL_DIR / f"{path.stem}.{mode}.traceback.txt")
        (FAIL_DIR / f"{path.stem}.{mode}.traceback.txt").write_text(
            "".join(traceback.format_exception(exc)), encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover
        status = "failed"
        err_type = exc.__class__.__name__
        err_msg = str(exc)[:1000]
        # fallback executor doesn't expose cell number directly; parse best effort
        tb_txt = "".join(traceback.format_exception(exc))
        for token in tb_txt.splitlines():
            if "#cell" in token:
                try:
                    failed_cell = int(token.rsplit("#cell", 1)[-1].split('"')[0].split()[0])
                    break
                except Exception:
                    pass
        tb_path = str(FAIL_DIR / f"{path.stem}.{mode}.traceback.txt")
        (FAIL_DIR / f"{path.stem}.{mode}.traceback.txt").write_text(tb_txt, encoding="utf-8")

    dur = time.time() - started
    return NotebookResult(
        notebook=path.name,
        path=str(path.relative_to(ROOT)),
        mode=mode,
        status=status,
        duration_sec=round(dur, 3),
        failed_cell=failed_cell,
        error_type=err_type,
        error_message=err_msg,
        traceback_path=tb_path,
        executed_path=str(out_path) if out_path.exists() else None,
        memory_mb=None,
    )


def _write_summary(rows: list[NotebookResult], mode: str) -> Path:
    out_csv = OUT_DIR / f"{mode}_run_summary.csv"
    fields = [
        "notebook",
        "path",
        "mode",
        "status",
        "duration_sec",
        "failed_cell",
        "error_type",
        "error_message",
        "traceback_path",
        "executed_path",
        "memory_mb",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)
    return out_csv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute notebooks from manifest with clean kernels.")
    ap.add_argument("--mode", choices=["sanity", "full", "retest"], default="sanity")
    ap.add_argument("--from-manifest", action="store_true", default=False)
    ap.add_argument("--notebooks", nargs="*", default=[])
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--only-01-13", action="store_true", default=True)
    ap.add_argument("--failed-from", type=str, default="")
    return ap.parse_args()


def _resolve_paths(args: argparse.Namespace) -> list[Path]:
    if args.mode == "retest":
        if args.failed_from:
            failed_csv = Path(args.failed_from)
        else:
            failed_csv = OUT_DIR / "full_run_summary.csv"
        if not failed_csv.exists():
            return []
        rows = []
        with failed_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("status") == "failed":
                    rows.append(ROOT / r["path"])
        return rows

    if args.notebooks:
        return [ROOT / p for p in args.notebooks]

    if args.from_manifest:
        paths = _iter_manifest_paths()
    else:
        paths = sorted(ROOT.joinpath("notebooks").rglob("*.ipynb"))

    if args.only_01_13:
        paths = _filter_01_13(paths)
    return paths


def main() -> None:
    args = parse_args()
    _ensure_dirs()
    paths = _resolve_paths(args)
    if not paths:
        print("No notebooks to execute.")
        return

    results: list[NotebookResult] = []
    for p in paths:
        res = _execute_notebook(p, mode=args.mode, timeout=args.timeout)
        results.append(res)
        print(f"[{res.status}] {res.path} ({res.duration_sec:.1f}s)")
        if res.status != "ok" and res.error_message:
            print(f"  -> {res.error_type}: {res.error_message}")

    out_csv = _write_summary(results, mode=args.mode)
    summary = {
        "mode": args.mode,
        "count": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "failed": sum(1 for r in results if r.status != "ok"),
        "summary_csv": str(out_csv),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if summary["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

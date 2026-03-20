from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from insurance_pricing import (
    build_submission,
    evaluate_run,
    export_ds_tables_and_figures,
    train_run,
)
from insurance_pricing.data.datasets import load_datasets, validate_data_contract
from insurance_pricing.data.io import ensure_dir
from insurance_pricing.runtime.ds_reporting import save_table


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_exec_id() -> str:
    return datetime.now(timezone.utc).strftime("e2e_%Y%m%dT%H%M%SZ")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_input_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return ROOT / p


def _resolve_report_path(run_report_output: str | None, exec_id: str) -> Path:
    if run_report_output is None:
        return ROOT / "artifacts" / "pipeline_runs" / exec_id / "run_report.json"
    p = _resolve_input_path(run_report_output)
    if p.suffix.lower() == ".json":
        return p
    return p / "run_report.json"


def _read_config_json(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _run_data_contract_check(
    *,
    data_dir: Path,
    test_path: Path,
    strict_data_contract: bool,
) -> dict[str, Any]:
    train_df, _ = load_datasets(data_dir)
    test_df = pd.read_csv(test_path)
    report = validate_data_contract(train_df, test_df)
    if strict_data_contract:
        if report["missing_train_columns"] or report["missing_test_columns"]:
            raise ValueError(f"Data contract failed: {report}")
        if int(report["n_test"]) != 50000:
            raise ValueError(f"Expected 50000 rows in test set, got {report['n_test']}")
    return report


def _run_ds_exports(ds_mode: str, exec_id: str) -> dict[str, str]:
    exports = export_ds_tables_and_figures(mode=ds_mode)
    # Guarantee at least one exported output for quick/full orchestration reports.
    if not exports:
        fallback = pd.DataFrame(
            [{"exec_id": exec_id, "ds_mode": ds_mode, "generated_at_utc": _utc_now()}]
        )
        table_path = save_table(fallback, name=f"run_end_to_end_{exec_id}", notebook="run_end_to_end")
        exports = {"table::run_end_to_end_fallback": table_path}
    return {k: str(v) for k, v in exports.items()}


def _check_persistence_files(run_dir: Path) -> dict[str, Any]:
    required = [
        "model_freq.pkl",
        "model_sev.pkl",
        "model_prime.pkl",
        "feature_schema.json",
        "metrics.json",
        "manifest.json",
    ]
    found = {name: (run_dir / name).exists() for name in required}
    missing = [name for name, ok in found.items() if not ok]
    return {"required_files": required, "found": found, "missing": missing}


def _validate_submission_contract(
    sub_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    strict_data_contract: bool,
) -> dict[str, Any]:
    if list(sub_df.columns) != ["index", "pred"]:
        raise ValueError(f"Submission columns invalid: {list(sub_df.columns)}")
    if len(sub_df) != len(test_df):
        raise ValueError(
            f"Submission row count mismatch: submission={len(sub_df)} test={len(test_df)}"
        )
    if strict_data_contract and len(sub_df) != 50000:
        raise ValueError(f"Expected 50000 rows in submission, got {len(sub_df)}")
    if not np.isfinite(sub_df["pred"].to_numpy(dtype=float)).all():
        raise ValueError("Submission contains non-finite predictions")
    if (sub_df["pred"].to_numpy(dtype=float) < 0).any():
        raise ValueError("Submission contains negative predictions")

    p = sub_df["pred"].to_numpy(dtype=float)
    return {
        "n_rows": int(len(sub_df)),
        "pred_min": float(np.min(p)),
        "pred_mean": float(np.mean(p)),
        "pred_q99": float(np.quantile(p, 0.99)),
        "pred_max": float(np.max(p)),
    }


def _write_report(report: dict[str, Any], report_path: Path) -> None:
    ensure_dir(report_path.parent)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_to_jsonable),
        encoding="utf-8",
    )


def _write_summary_md(report: dict[str, Any], summary_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Run End-to-End Summary")
    lines.append("")
    lines.append(f"- status: {report.get('status', 'unknown')}")
    lines.append(f"- exec_id: {report.get('exec_id', 'na')}")
    lines.append(f"- started_at_utc: {report.get('started_at_utc', 'na')}")
    lines.append(f"- ended_at_utc: {report.get('ended_at_utc', 'na')}")
    lines.append(f"- duration_sec: {report.get('duration_sec', 'na')}")
    lines.append("")
    run_info = report.get("run", {})
    lines.append("## Training")
    lines.append(f"- run_id: {run_info.get('run_id', 'na')}")
    lines.append(f"- run_dir: {run_info.get('run_dir', 'na')}")
    lines.append("")
    eval_info = report.get("evaluation", {})
    lines.append("## Evaluation")
    lines.append(f"- rmse_prime: {eval_info.get('rmse_prime', 'na')}")
    lines.append(f"- mae_prime: {eval_info.get('mae_prime', 'na')}")
    lines.append("")
    sub_info = report.get("submission", {})
    lines.append("## Submission")
    lines.append(f"- output_path: {sub_info.get('output_path', 'na')}")
    lines.append(f"- n_rows: {sub_info.get('n_rows', 'na')}")
    lines.append(f"- pred_q99: {sub_info.get('pred_q99', 'na')}")
    lines.append("")
    ds_info = report.get("ds_exports", {})
    lines.append("## DS Exports")
    lines.append(f"- mode: {ds_info.get('mode', 'off')}")
    lines.append(f"- n_outputs: {ds_info.get('n_outputs', 0)}")
    lines.append("")
    err = report.get("error")
    if err:
        lines.append("## Error")
        lines.append(f"- type: {err.get('type', 'unknown')}")
        lines.append(f"- message: {err.get('message', 'unknown')}")
        lines.append("")
    ensure_dir(summary_path.parent)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end runner: data check -> DS exports -> train -> evaluate -> submission."
    )
    parser.add_argument("--config", default="configs/train_robust.json")
    parser.add_argument("--ds-mode", default="quick", choices=["off", "quick", "full"])
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--submission-output", default=None)
    parser.add_argument("--run-report-output", default=None)
    parser.add_argument("--fail-if-missing-pickles", type=_parse_bool, default=True)
    parser.add_argument("--strict-data-contract", type=_parse_bool, default=True)
    return parser.parse_args()


def run_end_to_end(args: argparse.Namespace) -> dict[str, Any]:
    started_ts = time.time()
    started_at = _utc_now()
    exec_id = _make_exec_id()
    config_path = _resolve_input_path(args.config)
    test_path = _resolve_input_path(args.test_path)
    report_path = _resolve_report_path(args.run_report_output, exec_id)
    summary_path = report_path.parent / "summary.md"
    config_snapshot_path = report_path.parent / "config_used.json"

    report: dict[str, Any] = {
        "exec_id": exec_id,
        "status": "running",
        "started_at_utc": started_at,
        "options": {
            "config": str(config_path),
            "ds_mode": args.ds_mode,
            "test_path": str(test_path),
            "submission_output": args.submission_output,
            "run_report_output": str(report_path),
            "fail_if_missing_pickles": bool(args.fail_if_missing_pickles),
            "strict_data_contract": bool(args.strict_data_contract),
        },
    }

    try:
        config_obj = _read_config_json(config_path)
        ensure_dir(config_snapshot_path.parent)
        config_snapshot_path.write_text(
            json.dumps(config_obj, indent=2, ensure_ascii=False, default=_to_jsonable),
            encoding="utf-8",
        )

        data_dir = _resolve_input_path(str(config_obj.get("data_dir", "data")))
        train_path = data_dir / "train.csv"
        default_test_path = data_dir / "test.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not default_test_path.exists():
            raise FileNotFoundError(f"Default test file not found: {default_test_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Submission test file not found: {test_path}")

        data_contract = _run_data_contract_check(
            data_dir=data_dir,
            test_path=test_path,
            strict_data_contract=bool(args.strict_data_contract),
        )
        report["data_contract"] = data_contract

        if args.ds_mode != "off":
            ds_outputs = _run_ds_exports(args.ds_mode, exec_id)
            report["ds_exports"] = {
                "mode": args.ds_mode,
                "n_outputs": int(len(ds_outputs)),
                "outputs": ds_outputs,
            }
        else:
            report["ds_exports"] = {"mode": "off", "n_outputs": 0, "outputs": {}}

        train_out = train_run(str(config_path))
        run_id = str(train_out["run_id"])
        run_dir = Path(str(train_out["run_dir"]))
        report["run"] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "train_output": train_out,
        }

        persistence = _check_persistence_files(run_dir)
        report["persistence"] = persistence
        if persistence["missing"] and bool(args.fail_if_missing_pickles):
            missing_s = ", ".join(persistence["missing"])
            raise FileNotFoundError(f"Missing persisted artifacts in {run_dir}: {missing_s}")

        eval_out = evaluate_run(run_id)
        report["evaluation"] = eval_out

        test_df = pd.read_csv(test_path)
        sub_df = build_submission(run_id, test_df)
        sub_stats = _validate_submission_contract(
            sub_df, test_df, strict_data_contract=bool(args.strict_data_contract)
        )
        if args.submission_output:
            submission_path = _resolve_input_path(args.submission_output)
        else:
            submission_path = ROOT / "artifacts" / "submissions" / f"submission_{run_id}.csv"
        ensure_dir(submission_path.parent)
        sub_df.to_csv(submission_path, index=False)
        report["submission"] = {"output_path": str(submission_path), **sub_stats}

        report["status"] = "success"
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
        return report
    finally:
        report["ended_at_utc"] = _utc_now()
        report["duration_sec"] = round(time.time() - started_ts, 3)
        report["config_snapshot_path"] = str(config_snapshot_path)
        _write_report(report, report_path)
        _write_summary_md(report, summary_path)


def main() -> None:
    args = parse_args()
    report = run_end_to_end(args)
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "exec_id": report.get("exec_id"),
                "run_id": report.get("run", {}).get("run_id"),
                "submission_output": report.get("submission", {}).get("output_path"),
                "report_path": report.get("options", {}).get("run_report_output"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if report.get("status") != "success":
        err = report.get("error", {})
        msg = f"{err.get('type', 'Error')}: {err.get('message', 'unknown failure')}"
        print(msg, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

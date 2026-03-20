from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from insurance_pricing.data.io import ensure_dir


MODEL_ROOT = Path("artifacts") / "models"


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    model_freq_path: Path
    model_sev_path: Path
    model_prime_path: Path
    feature_schema_path: Path
    metrics_path: Path
    manifest_path: Path


class _CompatibilityUnpickler(pickle.Unpickler):
    """Load legacy artifacts serialized before the package import-path cleanup."""

    _MODULE_PREFIX_MAP = {
        "src.insurance_pricing": "insurance_pricing",
    }

    def find_class(self, module: str, name: str) -> Any:
        target_module = module
        for old_prefix, new_prefix in self._MODULE_PREFIX_MAP.items():
            if module == old_prefix or module.startswith(f"{old_prefix}."):
                target_module = module.replace(old_prefix, new_prefix, 1)
                break
        resolved_module = import_module(target_module)
        return getattr(resolved_module, name)


def _pickle_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _pickle_load(path: Path) -> Any:
    with path.open("rb") as f:
        return _CompatibilityUnpickler(f).load()


def _build_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}"


def _sanitize_run_id(run_id: str) -> str:
    bad = '<>:"/\\|?*'
    out = str(run_id)
    for ch in bad:
        out = out.replace(ch, "_")
    out = out.strip().replace(" ", "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out[:180] if len(out) > 180 else out


def save_model_bundle(
    *,
    freq_model: Any,
    sev_model: Any,
    prime_model: Any,
    run_id: str | None = None,
    feature_schema: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> RunArtifacts:
    rid = _sanitize_run_id(str(run_id or _build_run_id("pricing")))
    run_dir = ensure_dir(MODEL_ROOT / rid)

    model_freq_path = run_dir / "model_freq.pkl"
    model_sev_path = run_dir / "model_sev.pkl"
    model_prime_path = run_dir / "model_prime.pkl"
    feature_schema_path = run_dir / "feature_schema.json"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"

    _pickle_dump(freq_model, model_freq_path)
    _pickle_dump(sev_model, model_sev_path)
    _pickle_dump(prime_model, model_prime_path)

    feature_schema_obj = dict(feature_schema or {})
    metrics_obj = dict(metrics or {})
    config_obj = dict(config or {})

    feature_schema_path.write_text(json.dumps(feature_schema_obj, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_obj, indent=2), encoding="utf-8")

    manifest = {
        "run_id": rid,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_files": {
            "freq": str(model_freq_path),
            "sev": str(model_sev_path),
            "prime": str(model_prime_path),
        },
        "feature_schema_file": str(feature_schema_path),
        "metrics_file": str(metrics_path),
        "config": config_obj,
        "notes": notes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _append_registry(
        {
            "run_id": rid,
            "created_at_utc": manifest["created_at_utc"],
            "run_dir": str(run_dir),
            "model_freq": str(model_freq_path),
            "model_sev": str(model_sev_path),
            "model_prime": str(model_prime_path),
            "rmse_prime": metrics_obj.get("rmse_prime"),
            "q99_ratio_pos": metrics_obj.get("q99_ratio_pos"),
            "status": "ready",
        }
    )

    return RunArtifacts(
        run_id=rid,
        run_dir=run_dir,
        model_freq_path=model_freq_path,
        model_sev_path=model_sev_path,
        model_prime_path=model_prime_path,
        feature_schema_path=feature_schema_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
    )


def load_model_bundle(run_id: str) -> Dict[str, Any]:
    run_dir = MODEL_ROOT / _sanitize_run_id(str(run_id))
    if not run_dir.exists():
        raise FileNotFoundError(f"Unknown run_id: {run_id}")
    return {
        "freq_model": _pickle_load(run_dir / "model_freq.pkl"),
        "sev_model": _pickle_load(run_dir / "model_sev.pkl"),
        "prime_model": _pickle_load(run_dir / "model_prime.pkl"),
        "feature_schema": json.loads((run_dir / "feature_schema.json").read_text(encoding="utf-8")),
        "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
        "manifest": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")),
        "run_dir": run_dir,
    }


def _append_registry(row: Mapping[str, Any]) -> Path:
    reg_path = MODEL_ROOT / "registry.csv"
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([dict(row)])
    if reg_path.exists():
        old = pd.read_csv(reg_path)
        out = pd.concat([old, new_row], ignore_index=True, sort=False)
    else:
        out = new_row
    out.to_csv(reg_path, index=False)
    return reg_path

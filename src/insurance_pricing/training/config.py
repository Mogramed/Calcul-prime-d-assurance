from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SplitConfig:
    n_blocks_time: int = 5
    n_splits_group: int = 5
    group_col: str = "id_client"
    split_names: List[str] = field(
        default_factory=lambda: ["primary_time", "secondary_group", "aux_blocked5"]
    )


@dataclass
class ModelSpecFreq:
    engine: str = "catboost"
    params: Dict[str, Any] = field(default_factory=dict)
    calibration: str = "none"


@dataclass
class ModelSpecSev:
    engine: str = "catboost"
    family: str = "two_part_tweedie"
    severity_mode: str = "weighted_tail"
    tweedie_power: float = 1.3
    params: Dict[str, Any] = field(default_factory=dict)
    use_tail_mapper: bool = True


@dataclass
class ModelSpecPrime:
    non_negative: bool = True


@dataclass
class TrainingConfig:
    run_name: str = "train_robust"
    seed: int = 42
    data_dir: str = "data"
    feature_set: str = "base_v2"
    drop_identifiers: bool = True
    use_target_encoding: bool = True
    target_encode_cols: List[str] = field(
        default_factory=lambda: ["code_postal", "cp3", "modele_vehicule", "marque_modele"]
    )
    target_encoding_smoothing: float = 20.0
    split: SplitConfig = field(default_factory=SplitConfig)
    freq: ModelSpecFreq = field(default_factory=ModelSpecFreq)
    sev: ModelSpecSev = field(default_factory=ModelSpecSev)
    prime: ModelSpecPrime = field(default_factory=ModelSpecPrime)
    notes: Optional[str] = None


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_training_config(path: str | Path) -> TrainingConfig:
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    default_raw = asdict(TrainingConfig())
    merged = _merge_dict(default_raw, raw)
    split = SplitConfig(**merged.get("split", {}))
    freq = ModelSpecFreq(**merged.get("freq", {}))
    sev = ModelSpecSev(**merged.get("sev", {}))
    prime = ModelSpecPrime(**merged.get("prime", {}))

    return TrainingConfig(
        run_name=str(merged.get("run_name", "train_robust")),
        seed=int(merged.get("seed", 42)),
        data_dir=str(merged.get("data_dir", "data")),
        feature_set=str(merged.get("feature_set", "base_v2")),
        drop_identifiers=bool(merged.get("drop_identifiers", True)),
        use_target_encoding=bool(merged.get("use_target_encoding", True)),
        target_encode_cols=list(merged.get("target_encode_cols", default_raw["target_encode_cols"])),
        target_encoding_smoothing=float(
            merged.get("target_encoding_smoothing", default_raw["target_encoding_smoothing"])
        ),
        split=split,
        freq=freq,
        sev=sev,
        prime=prime,
        notes=merged.get("notes"),
    )


def save_training_config(config: TrainingConfig, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return out


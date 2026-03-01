from __future__ import annotations

from src.insurance_pricing.cv.integrity import build_splits, validate_split_integrity
from src.insurance_pricing.data.datasets import load_datasets
from src.insurance_pricing.training.config import SplitConfig


def test_split_integrity():
    train, _ = load_datasets("data")
    cfg = SplitConfig()
    splits = build_splits(train, cfg)
    report = validate_split_integrity(splits, train=train, group_col=cfg.group_col)
    assert "primary_time" in report
    assert "secondary_group" in report
    assert "aux_blocked5" in report
    assert report["primary_time"]["ok"] is True
    assert report["secondary_group"]["ok"] is True

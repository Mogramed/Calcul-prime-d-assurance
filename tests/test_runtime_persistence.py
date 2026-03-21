from __future__ import annotations

from pathlib import Path

import pytest

from insurance_pricing.runtime.persistence import load_model_bundle


def test_load_model_bundle_supports_explicit_model_root(
    monkeypatch: pytest.MonkeyPatch,
    existing_run_id: str,
) -> None:
    monkeypatch.setenv("INSURANCE_PRICING_MODEL_ROOT", str(Path("artifacts") / "models"))

    bundle = load_model_bundle(existing_run_id)

    assert bundle["run_dir"].name == existing_run_id


def test_load_model_bundle_error_lists_searched_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSURANCE_PRICING_MODEL_ROOT", "missing-model-root")

    with pytest.raises(FileNotFoundError, match="Searched model roots:") as exc_info:
        load_model_bundle("missing-run-id")

    assert "Available run_ids:" in str(exc_info.value)

from __future__ import annotations

from insurance_pricing.data.datasets import load_datasets, validate_data_contract


def test_data_contract_basic():
    train, test = load_datasets("data")
    report = validate_data_contract(train, test)
    assert report["n_train"] > 0
    assert report["n_test"] > 0
    assert report["missing_train_columns"] == []
    assert report["missing_test_columns"] == []

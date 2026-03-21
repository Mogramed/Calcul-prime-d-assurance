from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def train_main() -> None:
    from insurance_pricing.workflows import train_run

    parser = argparse.ArgumentParser(
        description="Train insurance pricing models and save artifacts."
    )
    parser.add_argument("--config", required=True, help="Path to the training JSON config file.")
    args = parser.parse_args()
    print(json.dumps(train_run(args.config), indent=2, ensure_ascii=False))


def evaluate_main() -> None:
    from insurance_pricing.workflows import evaluate_run

    parser = argparse.ArgumentParser(description="Evaluate a saved run on train and test data.")
    parser.add_argument(
        "--run-id", required=True, help="Model run_id from artifacts/models/registry.csv."
    )
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.run_id), indent=2, ensure_ascii=False))


def predict_main() -> None:
    from insurance_pricing.workflows import predict_from_run

    parser = argparse.ArgumentParser(
        description="Predict frequency, severity, and prime from a saved run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    output_df = predict_from_run(args.run_id, input_df)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"written: {output_path}")


def make_submission_main() -> None:
    from insurance_pricing.workflows import build_submission

    parser = argparse.ArgumentParser(description="Build a Kaggle submission from a saved run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-test", default="data/test.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    test_df = pd.read_csv(args.input_test)
    submission_df = build_submission(args.run_id, test_df)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"written: {output_path} ({len(submission_df)} rows)")


def serve_api_main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the Insurance Pricing FastAPI application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "insurance_pricing.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

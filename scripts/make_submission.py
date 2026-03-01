from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.insurance_pricing import build_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Kaggle submission from saved run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-test", default="data/test.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    test_df = pd.read_csv(args.input_test)
    sub = build_submission(args.run_id, test_df)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"written: {out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()

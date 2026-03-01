from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.insurance_pricing import predict_from_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict components (freq/sev/prime) from saved run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    args = parser.parse_args()

    input_df = pd.read_csv(args.input)
    pred = predict_from_run(args.run_id, input_df)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    print(f"written: {out}")


if __name__ == "__main__":
    main()

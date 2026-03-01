from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.insurance_pricing import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved run on train/test.")
    parser.add_argument("--run-id", required=True, help="Model run_id from artifacts/models/registry.csv")
    args = parser.parse_args()
    out = evaluate_run(args.run_id)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

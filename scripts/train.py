from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.insurance_pricing import train_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train insurance pricing models and save pickles.")
    parser.add_argument("--config", required=True, help="Path to training JSON config.")
    args = parser.parse_args()

    out = train_run(args.config)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

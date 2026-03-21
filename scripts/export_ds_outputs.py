from __future__ import annotations

import argparse
import json

from insurance_pricing import export_ds_tables_and_figures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DS tables and figures into standardized folders."
    )
    parser.add_argument("--mode", default="full", choices=["full", "quick"])
    args = parser.parse_args()
    out = export_ds_tables_and_figures(mode=args.mode)
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

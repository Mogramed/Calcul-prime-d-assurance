from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/export_openapi.py <output-path>", file=sys.stderr)
        return 1

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from insurance_pricing.api import AppSettings, create_app

    output_path = Path(sys.argv[1]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    app = create_app(
        AppSettings(
            run_id="openapi-preview",
            database_url="postgresql+psycopg://preview:preview@127.0.0.1:5432/preview",
            log_level="INFO",
            log_json=False,
        )
    )
    output_path.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

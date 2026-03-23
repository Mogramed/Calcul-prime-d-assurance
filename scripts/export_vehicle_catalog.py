from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT_DIR / "data" / "train.csv"
DEFAULT_OUTPUT = ROOT_DIR / "web" / "data" / "vehicle-catalog.json"


def display_label(raw_value: str) -> str:
    tokens = raw_value.split()
    rendered_tokens: list[str] = []

    for token in tokens:
        if token.isupper() and len(token) <= 3:
            rendered_tokens.append(token)
            continue

        if any(character.isdigit() for character in token):
            rendered_tokens.append(token.upper())
            continue

        if token.isalpha() and token.isupper():
            rendered_tokens.append(token[:1] + token[1:].lower())
            continue

        rendered_tokens.append(token)

    return " ".join(rendered_tokens)


def build_catalog(source_path: Path) -> dict[str, object]:
    frame = pd.read_csv(source_path, usecols=["marque_vehicule", "modele_vehicule"])
    grouped = (
        frame.assign(record_count=1)
        .groupby(["marque_vehicule", "modele_vehicule"], as_index=False)["record_count"]
        .sum()
    )

    brand_totals = (
        grouped.groupby("marque_vehicule", as_index=False)["record_count"].sum().rename(columns={"record_count": "brand_count"})
    )
    grouped = grouped.merge(brand_totals, on="marque_vehicule", how="left")
    grouped = grouped.sort_values(
        by=["brand_count", "marque_vehicule", "record_count", "modele_vehicule"],
        ascending=[False, True, False, True],
    )

    brands: list[dict[str, object]] = []
    for brand, brand_rows in grouped.groupby("marque_vehicule", sort=False):
        models = [
            {
                "value": str(row.modele_vehicule),
                "label": display_label(str(row.modele_vehicule)),
            }
            for row in brand_rows.itertuples(index=False)
        ]
        brands.append(
            {
                "value": str(brand),
                "label": display_label(str(brand)),
                "models": models,
            }
        )

    return {"brands": brands}


def main() -> None:
    catalog = build_catalog(DEFAULT_SOURCE)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(
        json.dumps(catalog, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote vehicle catalog to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()

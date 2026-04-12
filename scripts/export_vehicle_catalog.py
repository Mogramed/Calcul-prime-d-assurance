from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT_DIR / "data" / "train.csv"
DEFAULT_OUTPUT = ROOT_DIR / "web" / "data" / "vehicle-catalog.json"

CATALOG_COLUMNS = [
    "marque_vehicule",
    "modele_vehicule",
    "essence_vehicule",
    "din_vehicule",
    "vitesse_vehicule",
    "cylindre_vehicule",
    "poids_vehicule",
]

FUEL_LABELS = {
    "Gasoline": "Essence",
    "Diesel": "Diesel",
    "Hybrid": "Hybride",
}

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


def _to_int(value: object) -> int:
    return int(float(value))


def _variant_label(row: pd.Series) -> str:
    fuel = FUEL_LABELS.get(str(row["essence_vehicule"]), str(row["essence_vehicule"]))
    din = _to_int(row["din_vehicule"])
    speed = _to_int(row["vitesse_vehicule"])
    displacement = _to_int(row["cylindre_vehicule"])
    weight = _to_int(row["poids_vehicule"])
    return f"{fuel} | {din} ch DIN | {speed} km/h | {displacement} cm3 | {weight} kg"


def _variant_id(row: pd.Series) -> str:
    return "|".join(
        [
            str(row["essence_vehicule"]),
            str(_to_int(row["din_vehicule"])),
            str(_to_int(row["vitesse_vehicule"])),
            str(_to_int(row["cylindre_vehicule"])),
            str(_to_int(row["poids_vehicule"])),
        ]
    )


def build_catalog(source_path: Path) -> dict[str, object]:
    frame = pd.read_csv(source_path, usecols=CATALOG_COLUMNS)
    grouped = (
        frame.assign(record_count=1)
        .groupby(CATALOG_COLUMNS, as_index=False)["record_count"]
        .sum()
    )

    model_totals = (
        grouped.groupby(["marque_vehicule", "modele_vehicule"], as_index=False)["record_count"]
        .sum()
        .rename(columns={"record_count": "model_count"})
    )
    brand_totals = (
        grouped.groupby("marque_vehicule", as_index=False)["record_count"]
        .sum()
        .rename(columns={"record_count": "brand_count"})
    )
    grouped = grouped.merge(model_totals, on=["marque_vehicule", "modele_vehicule"], how="left")
    grouped = grouped.merge(brand_totals, on="marque_vehicule", how="left")
    grouped = grouped.sort_values(
        by=[
            "brand_count",
            "marque_vehicule",
            "model_count",
            "modele_vehicule",
            "record_count",
            "essence_vehicule",
            "din_vehicule",
            "vitesse_vehicule",
        ],
        ascending=[False, True, False, True, False, True, True, True],
    )

    brands: list[dict[str, object]] = []
    for brand, brand_rows in grouped.groupby("marque_vehicule", sort=False):
        models: list[dict[str, object]] = []
        for model, model_rows in brand_rows.groupby("modele_vehicule", sort=False):
            variants = [
                {
                    "id": _variant_id(row),
                    "label": _variant_label(row),
                    "essence_vehicule": str(row["essence_vehicule"]),
                    "din_vehicule": _to_int(row["din_vehicule"]),
                    "vitesse_vehicule": _to_int(row["vitesse_vehicule"]),
                    "cylindre_vehicule": _to_int(row["cylindre_vehicule"]),
                    "poids_vehicule": _to_int(row["poids_vehicule"]),
                    "record_count": _to_int(row["record_count"]),
                }
                for _, row in model_rows.iterrows()
            ]
            models.append(
                {
                    "value": str(model),
                    "label": display_label(str(model)),
                    "variants": variants,
                }
            )
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

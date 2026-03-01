# DS notebooks order

1. `07_ds_cadrage_qualite_cv.ipynb`
2. `08_ds_eda_segmentation_preprocessing.ipynb`
3. `09_ds_model_diagnostics_storytelling.ipynb`

## Objective

These notebooks document the full data science approach:
- business framing,
- data understanding and quality checks,
- EDA/segmentation/preprocessing,
- model diagnostics and storytelling.

## Prerequisites

- Notebook 07 and 08: run directly with `data/train.csv` and `data/test.csv`.
- Notebook 09: best used after V2 artifacts are present in `artifacts/v2/`.

## Execution modes

- `QUICK_ANALYSIS = True`: faster run on bounded samples.
- `FULL_ANALYSIS = True`: full statistical checks and heavier diagnostics.

## Standard exports

- Tables: `artifacts/ds/tables/`
- Figures: `artifacts/ds/figures/`
- Index: `artifacts/ds/ds_outputs_index.csv`
- Catalog: `artifacts/ds/README_outputs.md`

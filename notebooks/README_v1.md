# V1 Robust Pipeline - Run Order

1. Run `01_eda_cv_design.ipynb`
   - Builds feature engineering base
   - Exports folds and dataset artifacts

2. Run `02_modeling_3_engines.ipynb`
   - Trains CatBoost / LightGBM / XGBoost frequency+severity models
   - Saves `artifacts/run_registry.csv`, `artifacts/oof_predictions.parquet`, `artifacts/test_predictions.parquet`
   - Saves selected configs in `artifacts/selected_configs.json`

3. Run `03_ensemble_submission.ipynb`
   - Optimizes ensemble weights on OOF
   - Runs shake-up simulation (2000 public/private draws)
   - Refits finalists on full train and generates:
     - `artifacts/submission_v1.csv`
     - `artifacts/submission.csv`
     - `artifacts/ensemble_weights_v1.json`

## Runtime switch

In notebook 02:
- `RUN_FULL = False` (default): compact run for faster iteration
- `RUN_FULL = True`: robust 2-4h+ run, full scope


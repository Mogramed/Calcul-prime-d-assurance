# Architecture Projet (V6.2)

## Regle simple

1. Nouveau code: uniquement dans les modules canoniques.
2. `experiments/quick/`: archive lecture seule (pas d'evolution fonctionnelle).
3. Le root `src/` ne contient plus de wrappers legacy.
4. Compat legacy V1 exportee via `training` pour 1 cycle uniquement.

## Zones du projet

Code canonique actif:

1. `src/insurance_pricing/data/`
2. `src/insurance_pricing/features/`
3. `src/insurance_pricing/cv/`
4. `src/insurance_pricing/models/`
5. `src/insurance_pricing/training/`
6. `src/insurance_pricing/evaluation/`
7. `src/insurance_pricing/inference/`
8. `src/insurance_pricing/analytics/`
9. `src/insurance_pricing/runtime/`
10. `src/insurance_pricing/api.py`

Archive historique:

1. `src/insurance_pricing/legacy/`
2. `src/insurance_pricing/experiments/quick/`

## Ou coder

1. Entrainement/selection: `src/insurance_pricing/training/`
2. Modeles/calibration/tail: `src/insurance_pricing/models/`
3. Metriques/diagnostics: `src/insurance_pricing/evaluation/`
4. DS analysis: `src/insurance_pricing/analytics/`
5. Inference/submission: `src/insurance_pricing/inference/`
6. Persistence/export DS: `src/insurance_pricing/runtime/`

## API publique

1. `src.insurance_pricing.train_run`
2. `src.insurance_pricing.evaluate_run`
3. `src.insurance_pricing.predict_from_run`
4. `src.insurance_pricing.build_submission`
5. `src.insurance_pricing.export_ds_tables_and_figures`

## Politique Git

1. Le repo suit le code et la configuration.
2. Les artefacts runtime sont ignores (`artifacts/**`).
3. Les caches/logs (`__pycache__`, `.pytest_cache`, `catboost_info`) sont ignores.

## Checks recommandes

1. `python scripts/check_import_graph.py`
2. `python scripts/validate_repo_structure.py`
3. `pytest -q`

## Correction encoding notebook

1. Audit sans modification:
`python scripts/fix_notebook_encoding.py --dry-run`
2. Application des corrections:
`python scripts/fix_notebook_encoding.py --apply`
3. Verification stricte apres correction:
`python scripts/validate_repo_structure.py --encoding-strict`
4. Test dedie:
`pytest -q tests/test_notebook_encoding_health.py`

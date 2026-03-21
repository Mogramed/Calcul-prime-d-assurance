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
10. `src/insurance_pricing/workflows.py`
11. `src/insurance_pricing/api/`

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
7. Facade Python stable: `src/insurance_pricing/workflows.py`
8. Couche HTTP FastAPI: `src/insurance_pricing/api/`

## API publique

1. `insurance_pricing.train_run`
2. `insurance_pricing.evaluate_run`
3. `insurance_pricing.predict_from_run`
4. `insurance_pricing.build_submission`
5. `insurance_pricing.export_ds_tables_and_figures`

## Politique Git

1. Le repo suit le code et la configuration.
2. Les artefacts runtime sont ignores (`artifacts/**`).
3. Les caches/logs (`__pycache__`, `.pytest_cache`, `catboost_info`) sont ignores.

## Checks recommandes

1. `uv sync --group api --group train --group test --group lint`
2. `uv run pytest -q`
3. `uv run ruff check src tests scripts`
4. `uv run mypy`
5. `uv run insurance-pricing-api --help`

## Typage statique

1. Le gate mypy strict couvre la surface industrialisee de l'API FastAPI.
2. Le package historique hors API reste exploitable, mais n'est pas encore entierement amene au niveau de typage strict.
3. La commande `uv run mypy` applique donc le niveau strict sur le package cible configure dans `pyproject.toml`.

## Logging et PostgreSQL API

1. Le logging API est structure en JSON sur stdout.
2. Les predictions et erreurs API sont persistees en PostgreSQL.
3. `GET /health` reste une liveness probe.
4. `GET /ready` valide le modele charge et la connectivite PostgreSQL.
5. En Docker, l'image embarque aussi une copie de secours de `artifacts/models` pour les postes Windows ou Docker Desktop monte un bind volume vide.

## Workflow local DB/API

1. Demarrer PostgreSQL:
`docker compose up -d postgres`
2. Appliquer les migrations:
`docker compose run --rm migrate`
3. Lancer les tests d'integration:
`set INSURANCE_PRICING_TEST_DATABASE_URL=postgresql+psycopg://insurance_pricing:insurance_pricing@127.0.0.1:54329/insurance_pricing`
`uv run pytest -q -m integration`
4. Lancer l'API en local hors Docker:
`uv run insurance-pricing-api --host 127.0.0.1 --port 8000`
5. Lancer l'API via compose:
`docker compose up api`
6. Si Docker Desktop monte un dossier `artifacts/` vide sur Windows, definir `INSURANCE_PRICING_ARTIFACTS_DIR` dans `.env` vers un chemin partage par Docker.

## Correction encoding notebook

1. Audit sans modification:
`python scripts/fix_notebook_encoding.py --dry-run`
2. Application des corrections:
`python scripts/fix_notebook_encoding.py --apply`
3. Verification package/API apres correction:
`uv run pytest -q`

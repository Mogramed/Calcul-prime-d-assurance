# Project Architecture (V6.2)

## Simple Rules

1. New code: only in canonical modules.
2. `experiments/quick/`: read-only archive (no functional evolution).
3. Root `src/` no longer contains legacy wrappers.
4. Legacy V1 compatibility exported via `training` for 1 cycle only.

## Project Zones

Active canonical code:

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
12. `web/`

Historical archive:

1. `src/insurance_pricing/legacy/`
2. `src/insurance_pricing/experiments/quick/`

## Where to Code

1. Training/selection: `src/insurance_pricing/training/`
2. Models/calibration/tail: `src/insurance_pricing/models/`
3. Metrics/diagnostics: `src/insurance_pricing/evaluation/`
4. DS analysis: `src/insurance_pricing/analytics/`
5. Inference/submission: `src/insurance_pricing/inference/`
6. Persistence/export DS: `src/insurance_pricing/runtime/`
7. Stable Python facade: `src/insurance_pricing/workflows.py`
8. FastAPI HTTP layer: `src/insurance_pricing/api/`
9. Next.js product frontend: `web/`

## Public API

1. `insurance_pricing.train_run`
2. `insurance_pricing.evaluate_run`
3. `insurance_pricing.predict_from_run`
4. `insurance_pricing.build_submission`
5. `insurance_pricing.export_ds_tables_and_figures`

## Git Policy

1. The repo follows code and configuration.
2. Runtime artifacts are ignored (`artifacts/**`).
3. Caches/logs (`__pycache__`, `.pytest_cache`, `catboost_info`) are ignored.

## Recommended Checks

1. `uv sync --group api --group train --group test --group lint`
2. `uv run pytest -q`
3. `uv run ruff check src tests scripts`
4. `uv run mypy`
5. `uv run insurance-pricing-api --help`
6. `cd web && npm install`
7. `cd web && npm run codegen`
8. `cd web && npm run typecheck && npm run build`

## Static Typing

1. The mypy strict gate covers all `src/insurance_pricing`.
2. The only remaining tolerances in `pyproject.toml` concern untyped external dependencies.
3. The command `uv run mypy` is therefore a true repo-wide gate on the application package.

## Logging and PostgreSQL API

1. API logging is structured in JSON on stdout.
2. API predictions and errors are persisted in PostgreSQL.
3. `GET /health` remains a liveness probe.
4. `GET /ready` validates the loaded model and PostgreSQL connectivity.
5. In Docker, the image also includes a backup copy of `artifacts/models` for Windows machines or Docker Desktop that mounts an empty bind volume.

## Local DB/API Workflow

1. Start PostgreSQL:
`docker compose up -d postgres`
2. Apply migrations:
`docker compose run --rm migrate`
3. Run integration tests:
`set INSURANCE_PRICING_TEST_DATABASE_URL=postgresql+psycopg://insurance_pricing:insurance_pricing@127.0.0.1:54329/insurance_pricing`
`uv run pytest -q -m integration`
4. Run API locally outside Docker:
`uv run insurance-pricing-api --host 127.0.0.1 --port 8000`
5. Run API via compose:
`docker compose up api`
6. If Docker Desktop mounts an empty `artifacts/` folder on Windows, set `INSURANCE_PRICING_ARTIFACTS_DIR` in `.env` to a path shared by Docker.
7. Run Next.js frontend locally:
`cd web`
`npm install`
`npm run codegen`
`npm run catalog:vehicles`
`npm run dev`
8. Run full-stack preview via compose:
`docker compose up --build api web`

## Cloud Run Deployment

1. The repo contains a dedicated GitHub Actions workflow for Cloud Run deployment: `.github/workflows/deploy-cloud-run.yml`.
2. Details on GCP bootstrap, Workload Identity Federation, GitHub variables, and Neon branching are documented in `docs/deploy_cloud_run.md`.
3. A simplified checklist focused on `Repository variables` and `Repository secrets` is available in `docs/github_only_deploy.md`.
4. A post-deployment smoke test is available:
`uv run --group test python scripts/smoke_web_app.py --base-url https://nova-web-xxxxx-ew.a.run.app`

## Notebook Encoding Fix

1. Audit without modification:
`python scripts/fix_notebook_encoding.py --dry-run`
2. Apply corrections:
`python scripts/fix_notebook_encoding.py --apply`
3. Verify package/API after correction:
`uv run pytest -q`
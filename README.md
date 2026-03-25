# Nova Assurances - Insurance Pricing Platform

Complete auto insurance pricing project covering:

1. a Python data science package for training, evaluation, and inference
2. a FastAPI API for prediction and quote management
3. a product-oriented Next.js frontend for clients and administrators
4. a PostgreSQL persistence layer with Alembic migrations
5. a GitHub Actions CI, Docker images, and Cloud Run deployment

The project has evolved from a model/benchmark foundation to a complete web application named `Nova Assurances`, featuring authentication, quote history, PDF generation, an admin area, and cloud deployment.

## Overview

### What the project does

The pricing engine calculates an auto premium from raw business data. It exposes:

1. Python workflows to train and compare models
2. HTTP endpoints to score a record for frequency, severity, or final premium
3. a client website to create a quote, view its history, and download a PDF
4. admin features to monitor accounts and moderate quotes

### Main features

#### Data science

1. loading and preparation of training and test datasets
2. feature engineering and column schemas
3. multi-split benchmark and best run selection
4. persistence of model bundles in `artifacts/`
5. offline prediction and submission generation

#### Backend API

1. unitary and batch prediction endpoints
2. persistent quote endpoints
3. email + password authentication
4. PDF generation for each quote
5. admin endpoints for account and quote management
6. structured logs, readiness checks, and PostgreSQL persistence

#### Product frontend

1. "Nova Assurances" client landing page
2. guided quote flow
3. mandatory login before creating or viewing a quote
4. client area and quote history
5. admin console reserved for admin accounts

#### DevOps

1. Python + frontend CI in GitHub Actions
2. Docker Hub build and publish
3. Cloud Run deployment
4. post-deployment web flow smoke test

## Architecture

```mermaid
flowchart LR
    U["User"] --> W["Next.js web app"]
    W --> BFF["Next.js /api/* routes"]
    BFF --> API["FastAPI API"]
    API --> DB["PostgreSQL / Neon"]
    API --> ART["Model artifacts"]
    DS["Python workflows"] --> ART
    DS --> DB
    CI["GitHub Actions"] --> DH["Docker Hub"]
    CI --> CR["Cloud Run"]
    CR --> W
    CR --> API
Repository structurePathRolesrc/insurance_pricing/Main Python packagesrc/insurance_pricing/api/FastAPI API, auth, quotes, admin, persistencesrc/insurance_pricing/training/Training configs and orchestrationsrc/insurance_pricing/models/Frequency, severity, premium, and calibration modelssrc/insurance_pricing/evaluation/Metrics and diagnosticssrc/insurance_pricing/inference/Offline prediction and submissionsrc/insurance_pricing/runtime/Bundle persistence and DS exportssrc/insurance_pricing/workflows.pyStable Python facadeweb/Product Next.js frontendscripts/Utility tools, OpenAPI exports, smoke testtests/Unit and integration testsalembic/PostgreSQL migrations.github/workflows/CI, Docker publishing, Cloud Run deploymentdocs/GitHub / Cloud Run deployment documentationTech stackBackend and data sciencePython 3.13uv for dependency managementPandas, NumPy, scikit-learn, CatBoostFastAPI + UvicornSQLAlchemy + Psycopg + AlembicArgon2 for password hashingReportLab for PDF generationFrontendNext.js 16React 19TypeScriptReact Hook Form + ZodTanStack QueryTailwind CSSOpenAPI client generated with @hey-api/openapi-tsOpsDocker / Docker ComposeGitHub ActionsDocker HubGoogle Cloud RunNeon PostgreSQLBusiness workflows1. TrainingThe Python package allows training a pricing run from a JSON configuration file.Typical workflow:load datasetsbuild splits and verify their integritybenchmark multiple models / settingsselect the best runtrain the final modelssave the bundle in artifacts/models/Command:Bashuv run insurance-pricing-train --config configs/<my-run>.json
2. EvaluationAllows evaluating a saved run on the training / test sets.Bashuv run insurance-pricing-evaluate --run-id <run-id>
3. Offline predictionAllows scoring a CSV with a given run.Bashuv run insurance-pricing-predict --run-id <run-id> --input data/test.csv --output outputs/predictions.csv
4. SubmissionAllows building a submission from a run.Bashuv run insurance-pricing-make-submission --run-id <run-id> --output outputs/submission.csv
FastAPI APIDocumentationWhen the API is running:Swagger UI: /docsReDoc: /redocOpenAPI JSON: /openapi.jsonMain endpointsMetadata and healthGET /GET /versionGET /models/currentGET /healthGET /readyPredictionGET /predict/schemaPOST /predict/frequencyPOST /predict/frequency/batchPOST /predict/severityPOST /predict/severity/batchPOST /predict/primePOST /predict/prime/batchAuthenticationPOST /auth/registerPOST /auth/loginGET /auth/sessionPOST /auth/logoutQuotesPOST /quotesGET /quotesGET /quotes/{quote_id}GET /quotes/{quote_id}/report.pdfAdministrationGET /admin/usersDELETE /admin/users/{user_id}GET /admin/quotesDELETE /admin/quotes/{quote_id}API notesthe API persists errors, sessions, and quotes in PostgreSQLGET /ready verifies model loading and database connectivitythe quote endpoints are used by the Next.js frontend via its /api/* routesin the current state of the project, the Cloud Run API is configured as publicNext.js FrontendThe web/ frontend is the "Nova Assurances" product layer.Client flowpublic landing pageregistration / loginprotected access to quotingquote creationhistory consultationPDF report downloadAdmin flowlogin with an account whose email is listed in INSURANCE_PRICING_ADMIN_EMAILSaccess to /adminaccount consultationsoft deletion of users and quotesFrontend specificsquotes are blocked as long as no session is openthe browser only calls the frontend's same-origin /api/* routessession cookies are managed server-sidethe OpenAPI client is regenerated from web/openapi.jsonSee also: web/README.mdLocal installationPrerequisitesPython 3.13Node.js 22Docker DesktopuvDependency installationBashuv sync --all-groups --frozen
For the frontend:Bashcd web
npm install
npm run codegen
npm run catalog:vehicles
Local startupOption 1 - classic development mode1. Start PostgreSQLBashdocker compose up -d postgres
2. Apply migrationsBashdocker compose run --rm migrate
3. Start the APIBashuv run insurance-pricing-api --host 127.0.0.1 --port 8000
4. Start the frontendBashcd web
npm run dev
Option 2 - full stack preview with Docker ComposeBashdocker compose --profile ops up --build postgres migrate api web
Useful local URLsfrontend: http://127.0.0.1:3000API: http://127.0.0.1:8000Swagger: http://127.0.0.1:8000/docsReDoc: http://127.0.0.1:8000/redocImportant environment variablesBackendVariableRoleINSURANCE_PRICING_RUN_IDmodel bundle to loadINSURANCE_PRICING_DATABASE_URLPostgreSQL URLINSURANCE_PRICING_LOG_LEVELlog levelINSURANCE_PRICING_LOG_JSONJSON logsINSURANCE_PRICING_CORS_ALLOWED_ORIGINSallowed originsINSURANCE_PRICING_ADMIN_EMAILSauthorized admin emailsINSURANCE_PRICING_SESSION_TTL_HOURSsession durationFrontendVariableRoleAPI_BASE_URLupstream API URLAPI_AUDIENCEoptional Cloud Run audienceCOOKIE_SECUREfalse in local HTTP, true in HTTPSExample variables are provided in:.env.exampleweb/.env.exampleTests and qualityMain checksBashuv run ruff check src tests scripts
uv run mypy
uv run pytest -m "not integration"
uv run pytest -m integration
Frontend checksBashcd web
npm run codegen
npm run lint
npm run typecheck
npm run build
DockerBackend imageThe root Dockerfile builds the API / Python runtime image.Frontend imageweb/Dockerfile builds the production Next.js image.Composedocker-compose.yml orchestrates:postgresmigrateapiwebCI / CDCIThe ci.yml workflow executes:Python and Node installationfrontend OpenAPI client generationfrontend lintfrontend typecheckfrontend buildRuffMyPyAlembic migrationsunit testsintegration testsDocker smoke testDocker Hub image publishingDocker Hub publishingTwo images are published:API: <dockerhub-user>/calcul-prime-assuranceWeb: <dockerhub-user>/nova-assurances-webCloud Run DeploymentThe deploy-cloud-run.yml workflow manages:GCP authentication via Workload Identity FederationArtifact Registry bootstrap if necessaryimage build and pushmigration job deploymentAPI deploymentweb deploymentpost-deployment smoke testIn the current state:nova-web is publicnova-api is publicthe smoke test validates the web flow and authenticationRelated documentation:docs/deploy_cloud_run.mddocs/github_only_deploy.mdWeb smoke testThe smoke_web_app.py script allows validating a web deployment.Example:Bashuv run --group test python scripts/smoke_web_app.py --base-url [https://nova-web-xxxxx.a.run.app](https://nova-web-xxxxx.a.run.app)
This test notably verifies:landing page renderingprotection of /devis before loginregistrationquote creationhistoryPDF downloadAdditional documentationREADME_architecture.md: architecture / conventions overviewdocs/deploy_cloud_run.md: GCP bootstrap and deploymentdocs/github_only_deploy.md: GitHub-only configurationweb/README.md: frontend detailsProject statusThe project today covers an almost complete cycle:experimentation and model selectionindustrial HTTP exposureclient/admin web applicationpersistence and PDFCI, Docker, Cloud RunIt is therefore suitable for:an end-of-studies projecta full-stack / MLOps portfolio demoa technical base to move towards a more complete product

# Deploy Nova Assurances on Cloud Run

For a GitHub-first setup with only repository variables and secrets to manage day-to-day
configuration, see `docs/github_only_deploy.md`.

This repository now targets a two-service Cloud Run architecture:

1. `nova-web`: public Next.js frontend
2. `nova-api`: private FastAPI backend
3. `insurance-pricing-migrate`: Cloud Run Job for Alembic migrations

The web service calls the private API through server-side route handlers. On Cloud Run, the web
service obtains an identity token for the API audience and forwards quote requests server to server.
The browser only calls the public Next.js service. Anonymous quote history is scoped through the
`nova_client_id` cookie managed by the web service.

## Deployment model

1. GitHub Actions authenticates to Google Cloud through Workload Identity Federation.
2. The workflow builds and pushes two images to Artifact Registry: API and web.
3. The workflow deploys and executes the migration job with the API image.
4. The workflow deploys the private FastAPI service.
5. The workflow grants the runtime service account permission to invoke the private API.
6. The workflow resolves the private API URL.
7. The workflow deploys the public Next.js service with `API_BASE_URL`, `API_AUDIENCE`, and `COOKIE_SECURE=true`.

## Google Cloud resources to create

Create these resources once before the first deployment:

1. One Artifact Registry Docker repository
2. One runtime service account for the API, web service, and migration job
3. One deployer service account used by GitHub Actions through Workload Identity Federation

Neon resources to create:

1. One Neon project
2. One Neon PostgreSQL database

## Recommended bootstrap commands

Replace the placeholders before running the commands.

```bash
export PROJECT_ID="your-gcp-project"
export REGION="europe-west9"
export REPOSITORY="insurance-pricing"
export RUNTIME_SA="insurance-pricing-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
export DEPLOYER_SA="insurance-pricing-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
```

Enable the required Google APIs:

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  iamcredentials.googleapis.com
```

Create Artifact Registry:

```bash
gcloud artifacts repositories create "${REPOSITORY}" \
  --project "${PROJECT_ID}" \
  --location "${REGION}" \
  --repository-format docker \
  --description "Nova Assurances images"
```

Create service accounts:

```bash
gcloud iam service-accounts create insurance-pricing-runtime \
  --project "${PROJECT_ID}"

gcloud iam service-accounts create insurance-pricing-deployer \
  --project "${PROJECT_ID}"
```

Grant deployer permissions:

```bash
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${DEPLOYER_SA}" \
  --role roles/run.admin

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${DEPLOYER_SA}" \
  --role roles/artifactregistry.writer

gcloud iam service-accounts add-iam-policy-binding "${RUNTIME_SA}" \
  --member "serviceAccount:${DEPLOYER_SA}" \
  --role roles/iam.serviceAccountUser
```

## Workload Identity Federation for GitHub Actions

The deployment workflow is designed for Workload Identity Federation, not long-lived JSON keys.

Create the pool and provider:

```bash
export GITHUB_OWNER="your-github-owner"
export GITHUB_REPOSITORY="${GITHUB_OWNER}/Calcul-prime-d-assurance"

gcloud iam workload-identity-pools create "github" \
  --project "${PROJECT_ID}" \
  --location "global" \
  --display-name "GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc "calcul-prime-d-assurance" \
  --project "${PROJECT_ID}" \
  --location "global" \
  --workload-identity-pool "github" \
  --display-name "Calcul prime d assurance GitHub provider" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition "assertion.repository == '${GITHUB_REPOSITORY}'"
```

Read back the provider full resource name:

```bash
gcloud iam workload-identity-pools providers describe "calcul-prime-d-assurance" \
  --project "${PROJECT_ID}" \
  --location "global" \
  --workload-identity-pool "github" \
  --format "value(name)"
```

Allow GitHub Actions to impersonate the deployer service account:

```bash
export WORKLOAD_IDENTITY_POOL_NAME="$(gcloud iam workload-identity-pools describe github \
  --project "${PROJECT_ID}" \
  --location global \
  --format 'value(name)')"

gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_SA}" \
  --project "${PROJECT_ID}" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/${WORKLOAD_IDENTITY_POOL_NAME}/attribute.repository/${GITHUB_REPOSITORY}"
```

## Neon setup

1. Create a Neon account and a new project.
2. Create the target database if you do not want to use the default one.
3. Copy the PostgreSQL connection string from the Neon console.
4. Keep SSL enabled if Neon includes it in the URL.

The deployment workflow expects the complete Neon connection string exactly as provided by Neon, stored in GitHub Actions as a secret named `INSURANCE_PRICING_DATABASE_URL`.

## GitHub repository variables and secrets

Set these repository variables in GitHub:

```text
ENABLE_CLOUD_RUN_DEPLOY=true                       # optional, only for automatic deploy after CI
GCP_PROJECT_ID=your-gcp-project
GCP_REGION=europe-west9
GCP_ARTIFACT_REGISTRY_LOCATION=europe-west9
GCP_ARTIFACT_REGISTRY_REPOSITORY=insurance-pricing
GCP_WORKLOAD_IDENTITY_PROVIDER=projects/123456789/locations/global/workloadIdentityPools/github/providers/calcul-prime-d-assurance
GCP_DEPLOYER_SERVICE_ACCOUNT=insurance-pricing-deployer@your-gcp-project.iam.gserviceaccount.com
CLOUD_RUN_API_SERVICE=nova-api
CLOUD_RUN_WEB_SERVICE=nova-web
CLOUD_RUN_MIGRATION_JOB=insurance-pricing-migrate
CLOUD_RUN_RUNTIME_SERVICE_ACCOUNT=insurance-pricing-runtime@your-gcp-project.iam.gserviceaccount.com
INSURANCE_PRICING_RUN_ID=base_v2_catboost_two_part_tweedie_1.3_train_smoke_42_classic_none_none
```

Create this GitHub Actions secret:

```text
INSURANCE_PRICING_DATABASE_URL=postgresql+psycopg://...
```

Optional tuning variables:

```text
CLOUD_RUN_API_IMAGE_NAME=insurance-pricing-api
CLOUD_RUN_WEB_IMAGE_NAME=nova-assurances-web
CLOUD_RUN_API_CPU=1
CLOUD_RUN_API_MEMORY=512Mi
CLOUD_RUN_API_TIMEOUT=300
CLOUD_RUN_API_CONCURRENCY=10
CLOUD_RUN_API_MIN_INSTANCES=0
CLOUD_RUN_API_MAX_INSTANCES=1
CLOUD_RUN_WEB_CPU=1
CLOUD_RUN_WEB_MEMORY=512Mi
CLOUD_RUN_WEB_TIMEOUT=300
CLOUD_RUN_WEB_CONCURRENCY=10
CLOUD_RUN_WEB_MIN_INSTANCES=0
CLOUD_RUN_WEB_MAX_INSTANCES=2
CLOUD_RUN_JOB_CPU=1
CLOUD_RUN_JOB_MEMORY=512Mi
CLOUD_RUN_JOB_TIMEOUT=600s
INSURANCE_PRICING_ADMIN_EMAILS=admin@nova-assurances.fr
INSURANCE_PRICING_SESSION_TTL_HOURS=720
ENABLE_CLOUD_RUN_SMOKE_TEST=false
CLOUD_RUN_SMOKE_TEST_ADMIN_EMAIL=admin@nova-assurances.fr
```

Optional GitHub Actions secret for the smoke test:

```text
CLOUD_RUN_SMOKE_TEST_ADMIN_PASSWORD=your-admin-password
```

## IAM note for the private API

The workflow grants `roles/run.invoker` on the private API service to the runtime service account.
This allows the public web service to call the API with an identity token obtained from the Cloud Run
metadata server.

## Deployments

Manual deployment:

1. Open GitHub Actions
2. Run `Deploy to Cloud Run`
3. Wait for the migration job, API rollout, and web rollout to finish

Automatic deployment:

1. Set `ENABLE_CLOUD_RUN_DEPLOY=true` in repository variables
2. Push to `main`
3. The deployment workflow will trigger automatically after the `CI and Docker Publish` workflow succeeds

## First admin bootstrap

The admin console becomes available when the account email is listed in
`INSURANCE_PRICING_ADMIN_EMAILS`.

Recommended first bootstrap:

1. Set `INSURANCE_PRICING_ADMIN_EMAILS=admin@nova-assurances.fr` in GitHub repository variables
2. Deploy the stack once
3. Open the public web URL
4. Create the admin account from `/inscription` with the same email address
5. Confirm that `/admin` is now visible after login

If you enable the optional smoke test with `CLOUD_RUN_SMOKE_TEST_ADMIN_EMAIL`, the workflow can
also bootstrap this admin account automatically on its first run when the email already appears in
`INSURANCE_PRICING_ADMIN_EMAILS`.

## Post-deployment smoke test

The repository includes `scripts/smoke_web_app.py` to validate the public customer flow against the
deployed web URL:

```bash
uv run --group test python scripts/smoke_web_app.py \
  --base-url "https://nova-web-xxxxx-ew.a.run.app"
```

This checks:

1. public pages load correctly
2. account registration works
3. quote creation reaches the private API through the Next.js BFF
4. quote history is returned
5. the PDF report downloads successfully

To validate admin access and automatically clean up the temporary smoke user and quote, provide an
admin account:

```bash
uv run --group test python scripts/smoke_web_app.py \
  --base-url "https://nova-web-xxxxx-ew.a.run.app" \
  --admin-email "admin@nova-assurances.fr" \
  --admin-password "replace-me" \
  --admin-register-if-missing
```

If you run the script without admin credentials, it leaves a smoke user and a smoke quote in the
database. That is acceptable for a first validation, but for production it is cleaner to use the
admin cleanup mode.

## Go-live checklist

Before opening the site to real customers, validate these points:

1. Neon connection string is configured in `INSURANCE_PRICING_DATABASE_URL`
2. `INSURANCE_PRICING_RUN_ID` points to the production model bundle
3. `INSURANCE_PRICING_ADMIN_EMAILS` includes at least one real admin address
4. The Cloud Run deployment workflow succeeds end to end
5. The public web URL passes `scripts/smoke_web_app.py`
6. The admin can sign in and access `/admin`
7. The PDF report can be downloaded from a real customer account
8. Domain and DNS mapping are configured if you do not want to expose the default `run.app` URL

## Result

At the end of the workflow:

1. The web service is public and exposes the customer-facing site
2. The API service stays private
3. The browser only talks to the public web URL
4. The web service forwards quote requests to the private API through server-side route handlers

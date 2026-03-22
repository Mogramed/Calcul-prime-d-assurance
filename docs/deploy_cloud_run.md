# Deploy on Cloud Run with Neon PostgreSQL

This repository includes a dedicated GitHub Actions workflow at `.github/workflows/deploy-cloud-run.yml` for deploying the FastAPI service to Google Cloud Run while using Neon as the PostgreSQL database.

This version intentionally avoids:

1. Cloud SQL
2. Google Secret Manager

Instead, the deployment uses:

1. Google Cloud Run for the API and the Alembic migration job
2. Neon for PostgreSQL
3. GitHub Actions variables for non-sensitive deployment config
4. One GitHub Actions secret for the database URL

## Deployment model

1. GitHub Actions authenticates to Google Cloud through Workload Identity Federation.
2. The workflow builds the container image and pushes it to Artifact Registry.
3. The workflow deploys a Cloud Run Job that runs `alembic upgrade head` against Neon.
4. The workflow executes the migration job and waits for completion.
5. The workflow deploys the Cloud Run service with the same image, the pinned model `run_id`, and the Neon database URL injected as an environment variable.

The workflow supports two modes:

1. Manual deployment with `workflow_dispatch`
2. Automatic deployment after the `CI and Docker Publish` workflow succeeds on `main`, if the repository variable `ENABLE_CLOUD_RUN_DEPLOY` is set to `true`

## Google Cloud resources to create

Create these resources once before the first deployment:

1. One Artifact Registry Docker repository
2. One runtime service account for Cloud Run and Cloud Run Jobs
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
  --description "Insurance Pricing API images"
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

The runtime service account does not need `cloudsql.client` or `secretmanager.secretAccessor` in this Neon-based setup.

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
CLOUD_RUN_SERVICE=insurance-pricing-api
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
CLOUD_RUN_IMAGE_NAME=insurance-pricing-api
CLOUD_RUN_CPU=1
CLOUD_RUN_MEMORY=512Mi
CLOUD_RUN_TIMEOUT=300
CLOUD_RUN_CONCURRENCY=10
CLOUD_RUN_MIN_INSTANCES=0
CLOUD_RUN_MAX_INSTANCES=1
CLOUD_RUN_JOB_CPU=1
CLOUD_RUN_JOB_MEMORY=512Mi
CLOUD_RUN_JOB_TIMEOUT=600s
```

## Deployments

Manual deployment:

1. Open GitHub Actions
2. Run `Deploy to Cloud Run`
3. Wait for the migration job and the Cloud Run service rollout to finish

Automatic deployment:

1. Set `ENABLE_CLOUD_RUN_DEPLOY=true` in repository variables
2. Push to `main`
3. The deployment workflow will trigger automatically after the `CI and Docker Publish` workflow succeeds

## Notes

1. The Cloud Run workflow does not toggle public access. If you want a public service, manage the Cloud Run IAM policy separately.
2. The runtime still expects `INSURANCE_PRICING_RUN_ID` and `INSURANCE_PRICING_DATABASE_URL`; in this setup both are injected through GitHub Actions at deploy time.
3. The workflow defaults to `min instances = 0` and `max instances = 1` to reduce cost exposure during the Google Cloud Free Trial period.
4. This setup is designed to minimize paid Google services, but it does not guarantee zero billing if you upgrade your Google billing account or exceed the free program limits.

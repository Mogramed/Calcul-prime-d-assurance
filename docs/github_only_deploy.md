# Deploiement Cloud Run avec configuration GitHub uniquement

Cette checklist part du principe que toute la configuration applicative passe par les
`Repository variables` et `Repository secrets` GitHub.

Concretement:

1. pas de `.env` de production dans le repo
2. pas de saisie manuelle de variables sur Cloud Run
3. le workflow GitHub Actions pousse les valeurs vers Cloud Run au deploiement

La seule limite incompressible est la suivante:

1. les ressources Google Cloud doivent exister une premiere fois: projet GCP, Cloud Run, Artifact Registry, comptes de service, Workload Identity Federation
2. la base Neon doit exister

Une fois ce bootstrap fait, tu peux piloter le deploiement uniquement depuis GitHub.

## Ou saisir les valeurs

Dans GitHub:

1. `Settings`
2. `Secrets and variables`
3. `Actions`

Tu vas ensuite remplir:

1. `Variables`
2. `Secrets`

## Repository variables obligatoires

Ajoute exactement ces variables:

```text
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
INSURANCE_PRICING_ADMIN_EMAILS=admin@nova-assurances.fr
INSURANCE_PRICING_SESSION_TTL_HOURS=720
```

## Repository secrets obligatoires

Ajoute exactement ces secrets:

```text
INSURANCE_PRICING_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require
```

## Variables optionnelles utiles

Tu peux ensuite ajouter ces variables si tu veux affiner le comportement:

```text
ENABLE_CLOUD_RUN_DEPLOY=true
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
```

## Variables et secrets optionnels pour le smoke test post-deploiement

Si tu veux que GitHub teste automatiquement le site juste apres le deploiement:

Variables:

```text
ENABLE_CLOUD_RUN_SMOKE_TEST=true
CLOUD_RUN_SMOKE_TEST_ADMIN_EMAIL=admin@nova-assurances.fr
```

Secret:

```text
CLOUD_RUN_SMOKE_TEST_ADMIN_PASSWORD=your-admin-password
```

L'email admin utilise ici doit deja etre present dans `INSURANCE_PRICING_ADMIN_EMAILS`.

## Ordre recommande

1. remplir toutes les `Repository variables`
2. remplir les `Repository secrets`
3. verifier que `INSURANCE_PRICING_ADMIN_EMAILS` contient ton vrai email admin
4. lancer le workflow `Deploy to Cloud Run`
5. ouvrir l'URL publique retournee par le workflow
6. creer le compte admin avec l'email liste dans `INSURANCE_PRICING_ADMIN_EMAILS`
7. verifier l'acces a `/admin`

## Ce que tu peux ignorer

Pour la prod Cloud Run:

1. tu peux ignorer `.env.example`
2. tu peux ignorer `web/.env.example`
3. tu n'as rien a saisir a la main dans les variables Cloud Run si tu passes par le workflow GitHub

## Ce qui reste hors GitHub

Il reste uniquement ces prerequis hors GitHub:

1. creer une fois le projet GCP et Artifact Registry
2. creer une fois le compte de service runtime
3. creer une fois le compte de service deployer
4. creer une fois le provider Workload Identity Federation
5. creer une fois le projet Neon et recuperer l'URL PostgreSQL

Apres cela, ton pilotage quotidien peut rester centre sur GitHub.

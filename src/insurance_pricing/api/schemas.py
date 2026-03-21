from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

type NumericValue = StrictInt | StrictFloat
type PostalCodeValue = StrictInt | StrictStr
type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | dict[str, JsonValue] | list[JsonValue]

SINGLE_PREDICTION_EXAMPLE: dict[str, JsonValue] = {
    "index": 50000,
    "bonus": 0.58,
    "type_contrat": "Maxi",
    "duree_contrat": 1,
    "anciennete_info": 1,
    "freq_paiement": "Yearly",
    "paiement": "No",
    "utilisation": "Retired",
    "code_postal": 28388,
    "conducteur2": "No",
    "age_conducteur1": 66,
    "age_conducteur2": 0,
    "sex_conducteur1": "F",
    "sex_conducteur2": "",
    "anciennete_permis1": 34,
    "anciennete_permis2": 0,
    "anciennete_vehicule": 16.0,
    "cylindre_vehicule": 1239,
    "din_vehicule": 55,
    "essence_vehicule": "Gasoline",
    "marque_vehicule": "RENAULT",
    "modele_vehicule": "CLIO",
    "debut_vente_vehicule": 16,
    "fin_vente_vehicule": 15,
    "vitesse_vehicule": 150,
    "type_vehicule": "Tourism",
    "prix_vehicule": 10321,
    "poids_vehicule": 830,
}

BATCH_PREDICTION_EXAMPLE: dict[str, JsonValue] = {
    "records": [
        SINGLE_PREDICTION_EXAMPLE,
        {
            **SINGLE_PREDICTION_EXAMPLE,
            "index": 50001,
            "bonus": 0.63,
            "code_postal": 75015,
            "marque_vehicule": "PEUGEOT",
            "modele_vehicule": "208",
            "prix_vehicule": 14500,
            "poids_vehicule": 980,
        },
    ]
}

FREQUENCY_PREDICTION_EXAMPLE: dict[str, JsonValue] = {
    "index": 50000,
    "frequency_prediction": 0.0835,
}

SEVERITY_PREDICTION_EXAMPLE: dict[str, JsonValue] = {
    "index": 50000,
    "severity_prediction": 2410.24,
}

PRIME_PREDICTION_EXAMPLE: dict[str, JsonValue] = {
    "index": 50000,
    "frequency_prediction": 0.0835,
    "severity_prediction": 2410.24,
    "prime_prediction": 201.25504,
}

FREQUENCY_BATCH_PREDICTION_EXAMPLE: dict[str, JsonValue] = {
    "run_id": "base_v2_catboost_two_part_tweedie_1.3_train_smoke_42_classic_none_none",
    "count": 2,
    "predictions": [
        FREQUENCY_PREDICTION_EXAMPLE,
        {
            "index": 50001,
            "frequency_prediction": 0.0612,
        },
    ],
}

SEVERITY_BATCH_PREDICTION_EXAMPLE: dict[str, JsonValue] = {
    "run_id": "base_v2_catboost_two_part_tweedie_1.3_train_smoke_42_classic_none_none",
    "count": 2,
    "predictions": [
        SEVERITY_PREDICTION_EXAMPLE,
        {
            "index": 50001,
            "severity_prediction": 1795.86,
        },
    ],
}

PRIME_BATCH_PREDICTION_EXAMPLE: dict[str, JsonValue] = {
    "run_id": "base_v2_catboost_two_part_tweedie_1.3_train_smoke_42_classic_none_none",
    "count": 2,
    "predictions": [
        PRIME_PREDICTION_EXAMPLE,
        {
            "index": 50001,
            "frequency_prediction": 0.0612,
            "severity_prediction": 1795.86,
            "prime_prediction": 109.106232,
        },
    ],
}

HEALTH_RESPONSE_EXAMPLE: dict[str, JsonValue] = {
    "status": "ok",
    "run_id": "base_v2_catboost_two_part_tweedie_1.3_train_smoke_42_classic_none_none",
    "model_loaded": True,
}

API_INDEX_EXAMPLE: dict[str, JsonValue] = {
    "name": "Insurance Pricing API",
    "api_version": "0.1.0",
    "docs_url": "http://127.0.0.1:8000/docs",
    "redoc_url": "http://127.0.0.1:8000/redoc",
    "openapi_url": "http://127.0.0.1:8000/openapi.json",
    "health_url": "http://127.0.0.1:8000/health",
    "ready_url": "http://127.0.0.1:8000/ready",
    "version_url": "http://127.0.0.1:8000/version",
    "current_model_url": "http://127.0.0.1:8000/models/current",
    "prediction_schema_url": "http://127.0.0.1:8000/predict/schema",
}

VERSION_RESPONSE_EXAMPLE: dict[str, JsonValue] = {
    "api_version": "0.1.0",
    "model_run_id": "base_v2_catboost_two_part_tweedie_1.3_train_smoke_42_classic_none_none",
}

MODEL_METADATA_EXAMPLE: dict[str, JsonValue] = {
    "run_id": "base_v2_catboost_two_part_tweedie_1.3_train_smoke_42_classic_none_none",
    "created_at_utc": "2026-03-20T10:15:00+00:00",
    "notes": None,
    "model_files": {
        "freq": "artifacts/models/.../model_freq.pkl",
        "sev": "artifacts/models/.../model_sev.pkl",
        "prime": "artifacts/models/.../model_prime.pkl",
    },
    "metrics": {
        "rmse_prime": 29.18,
        "q99_ratio_pos": 1.04,
    },
    "feature_schema": {
        "feature_count": 48,
        "categorical_feature_count": 8,
        "numerical_feature_count": 40,
        "feature_columns": ["bonus", "type_contrat"],
        "categorical_columns": ["type_contrat", "freq_paiement"],
        "numerical_columns": ["bonus", "duree_contrat"],
    },
    "config": {
        "feature_set": "base_v2",
        "drop_identifiers": True,
        "frequency_engine": "catboost",
        "frequency_calibration": "none",
        "severity_engine": "catboost",
        "severity_family": "two_part_tweedie",
        "severity_mode": "classic",
        "tweedie_power": 1.3,
        "tail_mapper_enabled": False,
        "non_negative": True,
    },
}

PREDICTION_SCHEMA_EXAMPLE: dict[str, JsonValue] = {
    "record_model": "InsurancePricingRecord",
    "batch_model": "InsurancePricingBatchRequest",
    "supports_batch": True,
    "required_fields": [
        "bonus",
        "type_contrat",
        "duree_contrat",
    ],
    "optional_fields": ["index"],
    "fields": [
        {
            "name": "index",
            "type": "integer | null",
            "required": False,
            "description": "Optional correlation identifier echoed back in prediction responses.",
        },
        {
            "name": "bonus",
            "type": "integer | number",
            "required": True,
            "description": "Driver bonus-malus coefficient.",
        },
    ],
}


class PredictionInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "title": "InsurancePricingRecord",
            "examples": [SINGLE_PREDICTION_EXAMPLE],
        },
    )

    index: StrictInt | None = Field(
        default=None,
        description="Optional correlation identifier echoed back in prediction responses.",
    )
    bonus: NumericValue = Field(description="Driver bonus-malus coefficient.")
    type_contrat: StrictStr = Field(description="Insurance contract type.")
    duree_contrat: StrictInt = Field(description="Contract duration.")
    anciennete_info: StrictInt = Field(description="Years of customer history available.")
    freq_paiement: StrictStr = Field(description="Premium payment frequency.")
    paiement: StrictStr = Field(description="Whether the premium is currently paid.")
    utilisation: StrictStr = Field(description="Vehicle usage profile.")
    code_postal: PostalCodeValue = Field(description="Postal code of the insured risk.")
    conducteur2: StrictStr = Field(description="Whether a secondary driver is declared.")
    age_conducteur1: StrictInt = Field(description="Age of the primary driver.")
    age_conducteur2: StrictInt = Field(description="Age of the secondary driver, if any.")
    sex_conducteur1: StrictStr = Field(description="Sex of the primary driver.")
    sex_conducteur2: StrictStr = Field(description="Sex of the secondary driver.")
    anciennete_permis1: StrictInt = Field(description="Driving licence seniority for driver 1.")
    anciennete_permis2: StrictInt = Field(description="Driving licence seniority for driver 2.")
    anciennete_vehicule: NumericValue = Field(description="Vehicle age.")
    cylindre_vehicule: StrictInt = Field(description="Vehicle engine displacement.")
    din_vehicule: StrictInt = Field(description="Vehicle power in DIN.")
    essence_vehicule: StrictStr = Field(description="Vehicle fuel type.")
    marque_vehicule: StrictStr = Field(description="Vehicle brand.")
    modele_vehicule: StrictStr = Field(description="Vehicle model.")
    debut_vente_vehicule: StrictInt = Field(description="Vehicle sale start indicator.")
    fin_vente_vehicule: StrictInt = Field(description="Vehicle sale end indicator.")
    vitesse_vehicule: StrictInt = Field(description="Vehicle top speed.")
    type_vehicule: StrictStr = Field(description="Vehicle category.")
    prix_vehicule: StrictInt = Field(description="Vehicle price.")
    poids_vehicule: StrictInt = Field(description="Vehicle weight.")


class PredictionBatchInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "title": "InsurancePricingBatchRequest",
            "examples": [BATCH_PREDICTION_EXAMPLE],
        },
    )

    records: list[PredictionInput] = Field(
        min_length=1,
        description="Records scored in one request. Input order is preserved in the response.",
    )


class FrequencyPredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "FrequencyPredictionResponse",
            "examples": [FREQUENCY_PREDICTION_EXAMPLE],
        }
    )

    index: StrictInt | None = Field(
        default=None,
        description="Input correlation identifier, returned when it was provided by the caller.",
    )
    frequency_prediction: float = Field(
        description="Predicted probability of observing at least one claim.",
    )


class SeverityPredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "SeverityPredictionResponse",
            "examples": [SEVERITY_PREDICTION_EXAMPLE],
        }
    )

    index: StrictInt | None = Field(
        default=None,
        description="Input correlation identifier, returned when it was provided by the caller.",
    )
    severity_prediction: float = Field(
        description="Predicted expected severity conditional on a claim.",
    )


class PrimePredictionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "PrimePredictionResponse",
            "examples": [PRIME_PREDICTION_EXAMPLE],
        }
    )

    index: StrictInt | None = Field(
        default=None,
        description="Input correlation identifier, returned when it was provided by the caller.",
    )
    frequency_prediction: float = Field(description="Predicted claim frequency component.")
    severity_prediction: float = Field(description="Predicted claim severity component.")
    prime_prediction: float = Field(
        description="Final premium prediction computed as calibrated frequency x severity.",
    )


class FrequencyPredictionBatchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "FrequencyPredictionBatchResponse",
            "examples": [FREQUENCY_BATCH_PREDICTION_EXAMPLE],
        }
    )

    run_id: str = Field(description="Identifier of the model bundle currently serving the request.")
    count: int = Field(description="Number of returned predictions.")
    predictions: list[FrequencyPredictionResponse] = Field(
        description="Predictions returned in the same order as the input records.",
    )


class SeverityPredictionBatchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "SeverityPredictionBatchResponse",
            "examples": [SEVERITY_BATCH_PREDICTION_EXAMPLE],
        }
    )

    run_id: str = Field(description="Identifier of the model bundle currently serving the request.")
    count: int = Field(description="Number of returned predictions.")
    predictions: list[SeverityPredictionResponse] = Field(
        description="Predictions returned in the same order as the input records.",
    )


class PrimePredictionBatchResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "PrimePredictionBatchResponse",
            "examples": [PRIME_BATCH_PREDICTION_EXAMPLE],
        }
    )

    run_id: str = Field(description="Identifier of the model bundle currently serving the request.")
    count: int = Field(description="Number of returned predictions.")
    predictions: list[PrimePredictionResponse] = Field(
        description="Predictions returned in the same order as the input records.",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "HealthResponse",
            "examples": [HEALTH_RESPONSE_EXAMPLE],
        }
    )

    status: Literal["ok"] = Field(description="Health status of the API process.")
    run_id: str = Field(description="Configured model bundle identifier loaded at startup.")
    model_loaded: bool = Field(description="Whether the model bundle is loaded in memory.")


class FeatureSchemaSummary(BaseModel):
    feature_count: int = Field(
        description="Total number of engineered features used at inference time."
    )
    categorical_feature_count: int = Field(description="Number of categorical engineered features.")
    numerical_feature_count: int = Field(description="Number of numerical engineered features.")
    feature_columns: list[str] = Field(
        description="Ordered engineered feature list expected by the models."
    )
    categorical_columns: list[str] = Field(
        description="Categorical subset of the engineered feature list."
    )
    numerical_columns: list[str] = Field(
        description="Numerical subset of the engineered feature list."
    )


class ModelConfigSummary(BaseModel):
    feature_set: str | None = Field(
        default=None, description="Feature-set identifier used during training."
    )
    drop_identifiers: bool | None = Field(
        default=None, description="Whether identifiers were excluded from training features."
    )
    frequency_engine: str | None = Field(
        default=None, description="Training engine used for the frequency model."
    )
    frequency_calibration: str | None = Field(
        default=None, description="Calibration method applied to the raw frequency scores."
    )
    severity_engine: str | None = Field(
        default=None, description="Training engine used for the severity model."
    )
    severity_family: str | None = Field(default=None, description="Severity modeling family.")
    severity_mode: str | None = Field(default=None, description="Severity training mode.")
    tweedie_power: float | None = Field(
        default=None, description="Tweedie variance power when applicable."
    )
    tail_mapper_enabled: bool | None = Field(
        default=None, description="Whether a tail correction mapper is enabled."
    )
    non_negative: bool | None = Field(
        default=None, description="Whether final outputs are clipped to non-negative values."
    )


class ModelMetadataResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "ModelMetadataResponse",
            "examples": [MODEL_METADATA_EXAMPLE],
        }
    )

    run_id: str = Field(description="Identifier of the model bundle currently served by the API.")
    created_at_utc: str | None = Field(
        default=None, description="UTC timestamp at which the bundle manifest was created."
    )
    notes: str | None = Field(
        default=None, description="Optional free-form notes attached to the bundle."
    )
    model_files: dict[str, str] = Field(
        description="Filesystem paths of the loaded model artifacts."
    )
    metrics: dict[str, Any] = Field(
        description="Training or evaluation metrics captured with the model bundle."
    )
    feature_schema: FeatureSchemaSummary = Field(
        description="Summary of the engineered feature schema used for inference."
    )
    config: ModelConfigSummary = Field(
        description="Training configuration summary extracted from the bundle manifest."
    )


class VersionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "VersionResponse",
            "examples": [VERSION_RESPONSE_EXAMPLE],
        }
    )

    api_version: str = Field(description="Version of the installed `insurance_pricing` package.")
    model_run_id: str = Field(
        description="Identifier of the model bundle configured for this API instance."
    )


class ApiIndexResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "ApiIndexResponse",
            "examples": [API_INDEX_EXAMPLE],
        }
    )

    name: str = Field(description="Human-readable API name.")
    api_version: str = Field(description="Version of the installed `insurance_pricing` package.")
    docs_url: str = Field(description="Absolute URL of the interactive Swagger UI.")
    redoc_url: str = Field(description="Absolute URL of the ReDoc documentation.")
    openapi_url: str = Field(description="Absolute URL of the OpenAPI JSON schema.")
    health_url: str = Field(description="Absolute URL of the health endpoint.")
    ready_url: str = Field(description="Absolute URL of the readiness endpoint.")
    version_url: str = Field(description="Absolute URL of the version endpoint.")
    current_model_url: str = Field(
        description="Absolute URL of the current model metadata endpoint."
    )
    prediction_schema_url: str = Field(
        description="Absolute URL of the prediction input contract endpoint."
    )


class PredictionFieldDescriptor(BaseModel):
    name: str = Field(description="Field name expected by the prediction API.")
    type: str = Field(description="JSON/OpenAPI type summary for this field.")
    required: bool = Field(description="Whether the field must be provided by the client.")
    description: str | None = Field(
        default=None, description="Human-readable description of the field."
    )


class PredictionSchemaResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "PredictionSchemaResponse",
            "examples": [PREDICTION_SCHEMA_EXAMPLE],
        }
    )

    record_model: str = Field(description="Name of the single-record request model.")
    batch_model: str = Field(description="Name of the batch request model.")
    supports_batch: bool = Field(description="Whether batch scoring is supported by the API.")
    required_fields: list[str] = Field(description="Names of the required raw input fields.")
    optional_fields: list[str] = Field(description="Names of the optional raw input fields.")
    fields: list[PredictionFieldDescriptor] = Field(
        description="Ordered description of every raw field accepted by the prediction contract.",
    )

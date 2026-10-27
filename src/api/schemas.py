"""
SINYAL Phase 3 — Structured Pydantic v2 Schemas for FastAPI Endpoints.

Implements strict request/response validation for:
- GET /api/health
- GET /api/metadata
- POST /api/predict
- POST /api/batch (max 50 messages)
- GET /api/evaluation
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_API_MESSAGE_LENGTH = 10000
MAX_API_BATCH_SIZE = 50


class HealthResponse(BaseModel):
    """Health check response for the FastAPI server and ML model artifact."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(..., description="Overall service status ('ok' or 'degraded').")
    api_available: bool = Field(..., description="Whether the FastAPI routing layer is responsive.")
    model_available: bool = Field(..., description="Whether the serialized scikit-learn model bundle is loaded.")
    selected_model_id: str = Field(..., description="Selected Phase 2 model identifier.")
    model_version: str = Field(..., description="Serialized model version string.")
    application_mode: str = Field(
        default="research_demo",
        description="Operational mode indicator ('research_demo').",
    )


class MetadataResponse(BaseModel):
    """Public metadata about the frozen model, decision thresholds, and safety boundaries."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_version: str
    model_family: str
    model_description: str
    application_mode: str = "research_demo"
    supported_language: str
    target_labels: list[str]
    topic_labels: list[str]
    decision_thresholds: dict[str, float]
    default_thresholds: dict[str, float]
    uncertainty_half_width: float
    uncertainty_bands: dict[str, dict[str, float]]
    probability_calibration_status: str
    score_display_label: str = "Model-estimated score"
    dataset_split_counts: dict[str, int]
    safety_principles: list[str]
    limitation_summary: list[str]


class PredictRequest(BaseModel):
    """Single-message inference request payload."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        ...,
        description="English or English-translated disaster-related message to classify.",
    )

    @field_validator("message")
    @classmethod
    def validate_message_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Message must be a valid string.")
        if len(value.strip()) == 0:
            raise ValueError("Message cannot be empty or contain only whitespace.")
        if len(value) > MAX_API_MESSAGE_LENGTH:
            raise ValueError(
                f"Message length ({len(value)} characters) exceeds maximum allowed limit ({MAX_API_MESSAGE_LENGTH} characters)."
            )
        return value


class CategoryResult(BaseModel):
    """Detailed prediction and threshold comparison for a single target label."""

    model_config = ConfigDict(extra="forbid")

    label: str
    display_name: str
    is_request_intent: bool
    predicted: int
    score: float
    threshold: float
    in_uncertainty_band: bool
    lower_uncertainty_bound: float
    upper_uncertainty_bound: float

    @field_validator("score", "threshold", "lower_uncertainty_bound", "upper_uncertainty_bound")
    @classmethod
    def validate_probability_bounds(cls, value: float) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("Score and threshold values must be finite numbers.")
        val = float(value)
        if val < 0.0 or val > 1.0:
            raise ValueError(f"Probability/threshold value {val} must be in [0.0, 1.0].")
        return val


class PredictResponse(BaseModel):
    """Structured inference result for a single message."""

    model_config = ConfigDict(extra="forbid")

    message: str
    request_prediction: int
    request_score: float
    request_threshold: float
    detected_need_categories: list[str]
    need_topic_predictions: dict[str, int]
    need_topic_scores: dict[str, float]
    predictions: dict[str, int]
    scores: dict[str, float]
    thresholds: dict[str, float]
    categories: list[CategoryResult]
    human_review_recommended: bool
    uncertainty_indicated: bool
    review_status_label: str
    review_reason_code: str
    review_reasons_list: list[str]
    review_reason_human_readable: list[str]
    model_id: str
    model_version: str
    calibration_status: str
    score_display_label: str = "Model-estimated score"
    supported_language_disclaimer: str


class BatchPredictRequest(BaseModel):
    """Batch inference request payload (maximum 50 messages per request)."""

    model_config = ConfigDict(extra="forbid")

    messages: list[str] = Field(
        ...,
        description="List of messages to classify in batch order (1 to 50 messages).",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_batch(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("`messages` must be a list of strings.")
        if len(value) == 0:
            raise ValueError("`messages` list cannot be empty.")
        if len(value) > MAX_API_BATCH_SIZE:
            raise ValueError(
                f"Batch size ({len(value)}) exceeds maximum allowed limit of {MAX_API_BATCH_SIZE} messages per request."
            )
        for idx, msg in enumerate(value):
            if not isinstance(msg, str):
                raise ValueError(f"Message at index {idx} must be a string.")
            if len(msg.strip()) == 0:
                raise ValueError(f"Message at index {idx} cannot be empty or whitespace-only.")
            if len(msg) > MAX_API_MESSAGE_LENGTH:
                raise ValueError(
                    f"Message at index {idx} length ({len(msg)} characters) exceeds maximum allowed limit ({MAX_API_MESSAGE_LENGTH} characters)."
                )
        return value


class BatchPredictResponse(BaseModel):
    """Structured batch inference response preserving message order and retaining all messages."""

    model_config = ConfigDict(extra="forbid")

    total_messages: int
    flagged_for_review_count: int
    routine_monitoring_count: int
    uncertainty_indicated_count: int
    results: list[PredictResponse]
    retention_policy_note: str
    supported_language_disclaimer: str


class EvaluationResponse(BaseModel):
    """Validated frozen Phase 2 offline evaluation payload."""

    model_config = ConfigDict(extra="forbid")

    evaluation_timestamp_utc: str
    locked: bool
    dataset_description: str
    offline_benchmark_disclaimer: str
    selected_model_id: str
    calibration_status: str
    test_dataset_rows: int
    frozen_thresholds: dict[str, float]
    held_out_test_metrics_frozen_thresholds: dict[str, Any]
    held_out_test_metrics_default_0_50_reference: dict[str, Any]
    human_review_policy_test: dict[str, Any]
    exploratory_genre_subgroups_test: dict[str, Any]
    exploratory_ranking_utility_test: dict[str, Any]
    validation_model_comparison: list[dict[str, Any]]


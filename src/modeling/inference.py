"""
SINYAL Phase 2 — Reusable Offline Inference Module for Phase 3 FastAPI Integration.

Implements Section 20 of the Phase 2 Master Prompt:
- Provides `predict_messages(messages: list[str]) -> list[dict]`.
- Loads only the verified local project artifacts:
  * `models/sinyal_model.joblib`
  * `models/model_manifest.json`
  * `models/decision_policy.json`
- Validates against empty inputs, whitespace-only inputs, non-string types,
  excessively long messages, and excessive batch sizes.
- Does not save user-submitted messages by default or call external services.
- Note: Trained on English-language disaster messages (or English translations);
  Indonesian-language inputs are not validated by this benchmark model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src import TARGET_LABELS
from src.modeling.metrics import extract_positive_probabilities
from src.modeling.review_policy import TOPIC_LABELS, evaluate_single_message_policy

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "sinyal_model.joblib"
DEFAULT_MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
DEFAULT_POLICY_PATH = MODELS_DIR / "decision_policy.json"

MAX_BATCH_SIZE = 500
MAX_MESSAGE_LENGTH = 10000

_CACHED_BUNDLE: dict[str, Any] | None = None


def load_inference_bundle(
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
    force_reload: bool = False,
) -> dict[str, Any]:
    """Load and verify the serialized model pipeline, manifest, and decision policy."""
    global _CACHED_BUNDLE
    if (
        not force_reload
        and _CACHED_BUNDLE is not None
        and _CACHED_BUNDLE.get("model_path") == str(model_path)
    ):
        return _CACHED_BUNDLE

    if not model_path.exists():
        raise FileNotFoundError(f"Serialized model not found at {model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model manifest not found at {manifest_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"Decision policy not found at {policy_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    if manifest.get("target_labels") != TARGET_LABELS:
        raise ValueError(
            f"Manifest target_labels {manifest.get('target_labels')} do not match {TARGET_LABELS}"
        )
    if policy.get("target_labels") != TARGET_LABELS:
        raise ValueError(
            f"Policy target_labels {policy.get('target_labels')} do not match {TARGET_LABELS}"
        )

    model = joblib.load(model_path)
    bundle = {
        "model": model,
        "manifest": manifest,
        "policy": policy,
        "model_path": str(model_path),
    }
    _CACHED_BUNDLE = bundle
    return bundle


def validate_messages_input(
    messages: Any,
    max_batch_size: int = MAX_BATCH_SIZE,
    max_message_length: int = MAX_MESSAGE_LENGTH,
) -> list[str]:
    """
    Validate input batch for type safety, non-emptiness, whitespace, batch limits, and length limits.
    """
    if not isinstance(messages, list):
        raise TypeError(f"`messages` must be a list of strings, received {type(messages).__name__}")
    if len(messages) == 0:
        raise ValueError("`messages` list cannot be empty.")
    if len(messages) > max_batch_size:
        raise ValueError(
            f"Batch size {len(messages)} exceeds maximum allowed batch size ({max_batch_size})."
        )

    cleaned: list[str] = []
    for idx, item in enumerate(messages):
        if not isinstance(item, str):
            raise TypeError(
                f"Message at index {idx} must be a string, received {type(item).__name__}."
            )
        stripped = item.strip()
        if len(stripped) == 0:
            raise ValueError(f"Message at index {idx} is empty or contains only whitespace.")
        if len(item) > max_message_length:
            raise ValueError(
                f"Message at index {idx} length ({len(item)} chars) exceeds maximum allowed length ({max_message_length} chars)."
            )
        cleaned.append(item)

    return cleaned


def predict_messages(
    messages: list[str],
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> list[dict[str, Any]]:
    """
    Run deterministic multilabel inference and human-review policy evaluation on a batch of messages.

    For each message in `messages`, returns a dictionary containing:
    - `request_prediction`: int (0 or 1)
    - `request_score`: float in [0.0, 1.0]
    - `need_topic_predictions`: dict[str, int] for water, food, shelter, medical_help
    - `need_topic_scores`: dict[str, float] for water, food, shelter, medical_help
    - `predictions`: dict[str, int] across all 5 labels in canonical order
    - `scores`: dict[str, float] across all 5 labels in canonical order
    - `thresholds`: dict[str, float] per-label decision thresholds applied
    - `uncertainty_indicated`: bool
    - `human_review_recommended`: bool
    - `review_reason`: str
    - `model_version`: str
    - `calibration_status`: str
    - `language_scope_note`: str
    """
    valid_messages = validate_messages_input(messages)
    bundle = load_inference_bundle(
        model_path=model_path,
        manifest_path=manifest_path,
        policy_path=policy_path,
    )
    model = bundle["model"]
    manifest = bundle["manifest"]
    policy = bundle["policy"]

    prob_matrix = extract_positive_probabilities(model, valid_messages)
    thresholds = {lbl: float(policy["thresholds"][lbl]) for lbl in TARGET_LABELS}
    model_version = str(manifest.get("model_version", "2.0.0"))
    calibration_status = str(manifest.get("calibration_status", "uncalibrated"))

    outputs: list[dict[str, Any]] = []
    for i in range(len(valid_messages)):
        row_scores = {lbl: round(float(prob_matrix[i, j]), 6) for j, lbl in enumerate(TARGET_LABELS)}
        policy_eval = evaluate_single_message_policy(row_scores, policy)

        all_preds = {
            "request": policy_eval["request_prediction"],
            **policy_eval["need_topic_predictions"],
        }

        outputs.append(
            {
                "request_prediction": policy_eval["request_prediction"],
                "request_score": policy_eval["request_score"],
                "need_topic_predictions": policy_eval["need_topic_predictions"],
                "need_topic_scores": policy_eval["need_topic_scores"],
                "positive_need_topics": policy_eval["positive_need_topics"],
                "predictions": all_preds,
                "scores": row_scores,
                "thresholds": thresholds,
                "uncertainty_indicated": policy_eval["uncertainty_indicated"],
                "human_review_recommended": policy_eval["review_recommended"],
                "review_reason": policy_eval["review_reason"],
                "review_reasons_list": policy_eval["review_reasons_list"],
                "model_version": model_version,
                "selected_model_id": str(manifest.get("selected_model_id", "unknown")),
                "calibration_status": calibration_status,
                "language_scope_note": (
                    "Trained on English disaster-response messages and English translations; "
                    "not validated on untranslated Indonesian text."
                ),
            }
        )

    return outputs


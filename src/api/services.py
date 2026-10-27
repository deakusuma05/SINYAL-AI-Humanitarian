"""
SINYAL Phase 3 — Backend Service Layer.

Bridges FastAPI endpoints to the Phase 2 inference module (`src/modeling/inference.py`)
and frozen evaluation artifacts (`reports/phase2/`).
Enforces:
- No silent retraining or threshold modification.
- Strict finite probability bounds [0.0, 1.0].
- No exposure of local absolute filesystem paths in API responses.
- Clean human-readable review reason translation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from src import TARGET_LABELS
from src.api.schemas import (
    BatchPredictResponse,
    CategoryResult,
    EvaluationResponse,
    HealthResponse,
    MetadataResponse,
    PredictResponse,
)
from src.modeling.inference import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_POLICY_PATH,
    PROJECT_ROOT,
    load_inference_bundle,
    predict_messages,
)
from src.modeling.review_policy import TOPIC_LABELS

REPORTS_PHASE2_DIR = PROJECT_ROOT / "reports" / "phase2"
FINAL_TEST_METRICS_PATH = REPORTS_PHASE2_DIR / "final_test_metrics.json"
FINAL_EVAL_LOCK_PATH = REPORTS_PHASE2_DIR / "final_evaluation_lock.json"
MODEL_COMPARISON_CSV_PATH = REPORTS_PHASE2_DIR / "model_comparison_validation.csv"

LABEL_DISPLAY_NAMES: dict[str, str] = {
    "request": "Request (Aid Intent)",
    "water": "Water",
    "food": "Food",
    "shelter": "Shelter",
    "medical_help": "Medical Help",
}

REASON_HUMAN_READABLE_MAP: dict[str, str] = {
    "predicted_aid_request_verification": (
        "Predicted aid request — human verification required before any action."
    ),
    "borderline_positive_request_score": (
        "Borderline positive request score within ±0.08 uncertainty band."
    ),
    "borderline_request_score_below_threshold": (
        "Near-threshold request score within ±0.08 uncertainty band — manual check recommended."
    ),
    "topic_detected_with_moderate_request_score": (
        "Humanitarian need topic detected alongside moderate request score — verify whether aid is requested."
    ),
    "borderline_need_topic_score": (
        "Borderline need-topic score within ±0.08 uncertainty band."
    ),
    "routine_low_score_stream_retained_not_discarded": (
        "Routine monitoring stream — retained in queue and never automatically discarded."
    ),
}

LIMITATION_SUMMARY_ITEMS: list[str] = [
    "Trained and evaluated on English-language disaster-response messages and English translations only.",
    "Indonesian-language performance is not validated by this benchmark dataset.",
    "Derived from historical 2010–2012 disaster messages (Munro dataset); domain shift may occur in modern crises.",
    "Medical Help has lower held-out test F1 (0.4917) due to broad clinical and public-health vocabulary overlap.",
    "Direct messages achieve higher request F2 (0.8492) than third-person news reports (0.1880).",
    "False negatives remain possible: 56 of 478 true held-out test requests were not flagged by the review policy.",
    "All flagged messages require human verification; unflagged messages remain in routine monitoring and are never deleted.",
    "Research decision-support demonstration only — not an emergency dispatch system or clinical triage tool.",
]


class ModelUnavailableError(RuntimeError):
    """Raised when the serialized model bundle or policy artifact cannot be loaded."""


class InvalidModelOutputError(ValueError):
    """Raised when model inference returns non-finite or out-of-bounds scores."""


def _validate_finite_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidModelOutputError(f"{name} must be a numeric float, got {type(value).__name__}.")
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0 or fval > 1.0:
        raise InvalidModelOutputError(f"{name} value ({fval}) must be finite and within [0.0, 1.0].")
    return fval


def check_health(
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> HealthResponse:
    """Return service health status without leaking local file paths."""
    try:
        bundle = load_inference_bundle(
            model_path=model_path,
            manifest_path=manifest_path,
            policy_path=policy_path,
        )
        manifest = bundle["manifest"]
        return HealthResponse(
            status="ok",
            api_available=True,
            model_available=True,
            selected_model_id=str(manifest.get("selected_model_id", "M1_logreg_C2.0_cwNone")),
            model_version=str(manifest.get("model_version", "2.0.0")),
            application_mode="research_demo",
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            api_available=True,
            model_available=False,
            selected_model_id="unavailable",
            model_version="unavailable",
            application_mode="research_demo",
        )


def get_metadata(
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> MetadataResponse:
    """Return sanitized model metadata, thresholds, and safety boundaries."""
    try:
        bundle = load_inference_bundle(
            model_path=model_path,
            manifest_path=manifest_path,
            policy_path=policy_path,
        )
    except Exception as exc:
        raise ModelUnavailableError(
            "SINYAL model artifact or manifest is currently unavailable."
        ) from exc

    manifest = bundle["manifest"]
    policy = bundle["policy"]

    checksums = manifest.get("dataset_checksums", {})
    dataset_split_counts = {
        "train": int(checksums.get("train", {}).get("rows", 19032)),
        "validation": int(checksums.get("validation", {}).get("rows", 2329)),
        "test": int(checksums.get("test", {}).get("rows", 2387)),
    }

    return MetadataResponse(
        model_id=str(manifest.get("selected_model_id", "M1_logreg_C2.0_cwNone")),
        model_version=str(manifest.get("model_version", "2.0.0")),
        model_family=str(manifest.get("model_family", "M1")),
        model_description=str(
            manifest.get(
                "description",
                "TF-IDF + LogisticRegression (C=2.0, class_weight=None)",
            )
        ),
        application_mode="research_demo",
        supported_language="English (and English-translated disaster messages)",
        target_labels=list(TARGET_LABELS),
        topic_labels=list(TOPIC_LABELS),
        decision_thresholds={
            lbl: float(policy["thresholds"][lbl]) for lbl in TARGET_LABELS
        },
        default_thresholds={
            lbl: float(manifest.get("default_thresholds", {}).get(lbl, 0.5))
            for lbl in TARGET_LABELS
        },
        uncertainty_half_width=float(policy.get("uncertainty_half_width", 0.08)),
        uncertainty_bands=policy.get("uncertainty_bands", {}),
        probability_calibration_status=str(
            manifest.get("calibration_status", "uncalibrated_raw_probabilities")
        ),
        score_display_label="Model-estimated score",
        dataset_split_counts=dataset_split_counts,
        safety_principles=list(policy.get("safety_principles", [])),
        limitation_summary=list(LIMITATION_SUMMARY_ITEMS),
    )


def _format_single_prediction(
    message: str,
    raw_pred: dict[str, Any],
    policy: dict[str, Any],
) -> PredictResponse:
    """Transform a raw `predict_messages` item into a validated `PredictResponse`."""
    scores = raw_pred.get("scores", {})
    thresholds = raw_pred.get("thresholds", {})
    predictions = raw_pred.get("predictions", {})
    half_width = float(policy.get("uncertainty_half_width", 0.08))

    categories: list[CategoryResult] = []
    ordered_scores: dict[str, float] = {}
    ordered_thresholds: dict[str, float] = {}
    ordered_predictions: dict[str, int] = {}

    for lbl in TARGET_LABELS:
        score_val = _validate_finite_probability(scores.get(lbl), f"score[{lbl}]")
        thresh_val = _validate_finite_probability(thresholds.get(lbl), f"threshold[{lbl}]")
        pred_val = int(predictions.get(lbl, 0))
        if pred_val not in (0, 1):
            raise InvalidModelOutputError(f"Prediction for {lbl} must be 0 or 1, got {pred_val}.")

        lower_bound = round(max(0.0, thresh_val - half_width), 4)
        upper_bound = round(min(1.0, thresh_val + half_width), 4)
        in_band = abs(score_val - thresh_val) <= half_width + 1e-9

        ordered_scores[lbl] = score_val
        ordered_thresholds[lbl] = thresh_val
        ordered_predictions[lbl] = pred_val

        categories.append(
            CategoryResult(
                label=lbl,
                display_name=LABEL_DISPLAY_NAMES.get(lbl, lbl),
                is_request_intent=(lbl == "request"),
                predicted=pred_val,
                score=score_val,
                threshold=thresh_val,
                in_uncertainty_band=bool(in_band),
                lower_uncertainty_bound=lower_bound,
                upper_uncertainty_bound=upper_bound,
            )
        )

    request_pred = ordered_predictions["request"]
    request_score = ordered_scores["request"]
    request_threshold = ordered_thresholds["request"]

    human_review = bool(raw_pred.get("human_review_recommended", False))
    uncertainty = bool(raw_pred.get("uncertainty_indicated", False))

    if request_pred == 1:
        review_status_label = "Human verification recommended"
    elif human_review:
        review_status_label = "Borderline prediction — manual review"
    else:
        review_status_label = "Routine monitoring"

    reasons_list = list(raw_pred.get("review_reasons_list", []))
    if not reasons_list:
        reasons_list = ["routine_low_score_stream_retained_not_discarded"]

    human_readable_reasons = [
        REASON_HUMAN_READABLE_MAP.get(code, code.replace("_", " ").capitalize())
        for code in reasons_list
    ]

    detected_needs = [
        lbl for lbl in TOPIC_LABELS if ordered_predictions.get(lbl, 0) == 1
    ]

    return PredictResponse(
        message=message,
        request_prediction=request_pred,
        request_score=request_score,
        request_threshold=request_threshold,
        detected_need_categories=detected_needs,
        need_topic_predictions={lbl: ordered_predictions[lbl] for lbl in TOPIC_LABELS},
        need_topic_scores={lbl: ordered_scores[lbl] for lbl in TOPIC_LABELS},
        predictions=ordered_predictions,
        scores=ordered_scores,
        thresholds=ordered_thresholds,
        categories=categories,
        human_review_recommended=human_review,
        uncertainty_indicated=uncertainty,
        review_status_label=review_status_label,
        review_reason_code=str(raw_pred.get("review_reason", "")),
        review_reasons_list=reasons_list,
        review_reason_human_readable=human_readable_reasons,
        model_id=str(raw_pred.get("selected_model_id", "M1_logreg_C2.0_cwNone")),
        model_version=str(raw_pred.get("model_version", "2.0.0")),
        calibration_status=str(
            raw_pred.get("calibration_status", "uncalibrated_raw_probabilities")
        ),
        score_display_label="Model-estimated score",
        supported_language_disclaimer=str(
            raw_pred.get(
                "language_scope_note",
                "Trained on English disaster-response messages and English translations; not validated on untranslated Indonesian text.",
            )
        ),
    )


def run_single_prediction(
    message: str,
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> PredictResponse:
    """Run real Phase 2 model inference for a single message."""
    try:
        bundle = load_inference_bundle(
            model_path=model_path,
            manifest_path=manifest_path,
            policy_path=policy_path,
        )
        raw_results = predict_messages(
            [message],
            model_path=model_path,
            manifest_path=manifest_path,
            policy_path=policy_path,
        )
    except FileNotFoundError as exc:
        raise ModelUnavailableError(
            "Serialized SINYAL model artifact is missing or inaccessible."
        ) from exc
    except (ValueError, TypeError) as exc:
        raise exc
    except Exception as exc:
        raise ModelUnavailableError(
            "Model inference failed due to an internal model loading error."
        ) from exc

    if not raw_results or len(raw_results) != 1:
        raise InvalidModelOutputError("Model inference did not return a single prediction record.")

    return _format_single_prediction(message, raw_results[0], bundle["policy"])


def run_batch_prediction(
    messages: list[str],
    model_path: Path = DEFAULT_MODEL_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> BatchPredictResponse:
    """Run real Phase 2 model inference on a batch of up to 50 messages."""
    try:
        bundle = load_inference_bundle(
            model_path=model_path,
            manifest_path=manifest_path,
            policy_path=policy_path,
        )
        raw_results = predict_messages(
            messages,
            model_path=model_path,
            manifest_path=manifest_path,
            policy_path=policy_path,
        )
    except FileNotFoundError as exc:
        raise ModelUnavailableError(
            "Serialized SINYAL model artifact is missing or inaccessible."
        ) from exc
    except (ValueError, TypeError) as exc:
        raise exc
    except Exception as exc:
        raise ModelUnavailableError(
            "Batch model inference failed due to an internal model error."
        ) from exc

    if len(raw_results) != len(messages):
        raise InvalidModelOutputError(
            f"Batch output count ({len(raw_results)}) does not match input count ({len(messages)})."
        )

    formatted_results = [
        _format_single_prediction(msg, raw_item, bundle["policy"])
        for msg, raw_item in zip(messages, raw_results)
    ]

    flagged_count = sum(1 for item in formatted_results if item.human_review_recommended)
    routine_count = len(formatted_results) - flagged_count
    uncertainty_count = sum(1 for item in formatted_results if item.uncertainty_indicated)

    return BatchPredictResponse(
        total_messages=len(formatted_results),
        flagged_for_review_count=flagged_count,
        routine_monitoring_count=routine_count,
        uncertainty_indicated_count=uncertainty_count,
        results=formatted_results,
        retention_policy_note=(
            "All messages are retained in the triage stream. Unflagged messages remain in routine monitoring and are never automatically deleted."
        ),
        supported_language_disclaimer=(
            "Trained on English disaster-response messages and English translations; not validated on untranslated Indonesian text."
        ),
    )


def get_frozen_evaluation(
    metrics_path: Path = FINAL_TEST_METRICS_PATH,
    lock_path: Path = FINAL_EVAL_LOCK_PATH,
    comparison_csv_path: Path = MODEL_COMPARISON_CSV_PATH,
) -> EvaluationResponse:
    """Load the frozen Phase 2 held-out test metrics and validation model comparison."""
    if not metrics_path.exists() or not lock_path.exists():
        raise FileNotFoundError("Frozen Phase 2 evaluation report files are not available.")

    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))

    validation_rows: list[dict[str, Any]] = []
    if comparison_csv_path.exists():
        with comparison_csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row: dict[str, Any] = {}
                for k, v in row.items():
                    if k in ("model_id", "family", "threshold_regime"):
                        parsed_row[k] = v
                    else:
                        try:
                            parsed_row[k] = float(v) if v is not None else 0.0
                        except ValueError:
                            parsed_row[k] = v
                validation_rows.append(parsed_row)

    return EvaluationResponse(
        evaluation_timestamp_utc=str(metrics_data.get("evaluation_timestamp_utc", "")),
        locked=bool(lock_data.get("locked", True)),
        dataset_description=str(metrics_data.get("dataset_description", "")),
        offline_benchmark_disclaimer=str(metrics_data.get("offline_benchmark_disclaimer", "")),
        selected_model_id=str(metrics_data.get("selected_model_id", "M1_logreg_C2.0_cwNone")),
        calibration_status=str(
            metrics_data.get("calibration_status", "uncalibrated_raw_probabilities")
        ),
        test_dataset_rows=int(metrics_data.get("test_dataset_rows", 2387)),
        frozen_thresholds={
            lbl: float(metrics_data["frozen_thresholds"][lbl]) for lbl in TARGET_LABELS
        },
        held_out_test_metrics_frozen_thresholds=metrics_data.get(
            "held_out_test_metrics_frozen_thresholds", {}
        ),
        held_out_test_metrics_default_0_50_reference=metrics_data.get(
            "held_out_test_metrics_default_0.50_reference", {}
        ),
        human_review_policy_test=metrics_data.get("human_review_policy_test", {}),
        exploratory_genre_subgroups_test=metrics_data.get(
            "exploratory_genre_subgroups_test", {}
        ),
        exploratory_ranking_utility_test=metrics_data.get(
            "exploratory_ranking_utility_test", {}
        ),
        validation_model_comparison=validation_rows,
    )


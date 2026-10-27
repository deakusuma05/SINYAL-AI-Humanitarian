"""
SINYAL Phase 3 — Automated Backend API Tests (`tests/test_api.py`).

Covers Mandatory Backend Tests 1–12 from Section K of the Phase 3 Master Prompt:
1. Health endpoint returns 200 and model ready status.
2. Metadata endpoint returns valid schema and hides local filesystem paths.
3. Predict endpoint returns valid prediction response.
4. Batch endpoint returns valid predictions for multiple messages & enforces 50-message cap.
5. Empty/whitespace message is rejected cleanly.
6. Oversized message (> 10,000 chars) is rejected cleanly.
7. Invalid JSON / wrong field types are rejected cleanly.
8. Output label ordering matches exact project contract.
9. Frozen thresholds are preserved.
10. Review policy output is consistent with thresholds and uncertainty rules.
11. Simulated model failure returns structured 503 error without stack trace.
12. Evaluation endpoint loads the frozen evaluation payload cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import TARGET_LABELS
from src.api.main import app
from src.api import services

client = TestClient(app)

EXPECTED_FROZEN_THRESHOLDS = {
    "request": 0.19,
    "water": 0.22,
    "food": 0.26,
    "shelter": 0.32,
    "medical_help": 0.20,
}


def test_01_health_endpoint_returns_200_and_ready() -> None:
    """Test 1: Health endpoint returns 200 and model ready status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["api_available"] is True
    assert data["model_available"] is True
    assert data["selected_model_id"] == "M1_logreg_C2.0_cwNone"
    assert data["model_version"] == "2.0.0"
    assert data["application_mode"] == "research_demo"


def test_02_metadata_endpoint_valid_schema_and_no_local_paths() -> None:
    """Test 2: Metadata endpoint returns valid schema and never exposes local file paths."""
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "M1_logreg_C2.0_cwNone"
    assert data["application_mode"] == "research_demo"
    assert data["target_labels"] == TARGET_LABELS
    assert data["decision_thresholds"] == EXPECTED_FROZEN_THRESHOLDS
    assert data["uncertainty_half_width"] == pytest.approx(0.08)
    assert data["score_display_label"] == "Model-estimated score"
    assert data["dataset_split_counts"] == {
        "train": 19032,
        "validation": 2329,
        "test": 2387,
    }
    assert len(data["limitation_summary"]) >= 8

    # Ensure no local absolute paths leak in metadata JSON
    serialized = json.dumps(data)
    assert "D:\\" not in serialized
    assert "d:\\" not in serialized
    assert "C:\\" not in serialized
    assert "SEMESTER 7" not in serialized


def test_03_predict_endpoint_returns_valid_response() -> None:
    """Test 3: Predict endpoint returns valid prediction response for a direct aid request."""
    msg = "Please help us urgently, we have no clean drinking water and no food for our children."
    response = client.post("/api/predict", json={"message": msg})
    assert response.status_code == 200
    data = response.json()

    assert data["message"] == msg
    assert data["request_prediction"] == 1
    assert 0.0 <= data["request_score"] <= 1.0
    assert data["request_threshold"] == pytest.approx(0.19)
    assert "water" in data["detected_need_categories"]
    assert "food" in data["detected_need_categories"]
    assert data["human_review_recommended"] is True
    assert data["review_status_label"] == "Human verification recommended"
    assert data["score_display_label"] == "Model-estimated score"
    assert len(data["categories"]) == 5


def test_04_batch_endpoint_valid_and_enforces_max_50_limit() -> None:
    """Test 4: Batch endpoint returns valid predictions for multiple messages and rejects >50."""
    messages = [
        "We need clean water and food urgently at the shelter.",
        "Weather officials report calm winds across the coastal highway.",
        "My mother is injured and needs medical help and a tent.",
    ]
    response = client.post("/api/batch", json={"messages": messages})
    assert response.status_code == 200
    data = response.json()
    assert data["total_messages"] == 3
    assert len(data["results"]) == 3
    assert data["flagged_for_review_count"] + data["routine_monitoring_count"] == 3

    # Test batch > 50 rejection
    oversized_batch = [f"Test disaster message number {i}" for i in range(51)]
    resp_over = client.post("/api/batch", json={"messages": oversized_batch})
    assert resp_over.status_code == 422
    err_data = resp_over.json()
    assert err_data["error"] == "validation_error"
    assert "50" in err_data["message"]


def test_05_empty_and_whitespace_messages_rejected_cleanly() -> None:
    """Test 5: Empty or whitespace-only messages are rejected cleanly with 422."""
    for invalid_msg in ["", "   ", "\n\t  "]:
        resp = client.post("/api/predict", json={"message": invalid_msg})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"] == "validation_error"

    resp_batch = client.post("/api/batch", json={"messages": ["Valid message", "   "]})
    assert resp_batch.status_code == 422


def test_06_oversized_message_rejected_cleanly() -> None:
    """Test 6: Oversized message (> 10,000 chars) is rejected cleanly with 422."""
    long_msg = "water " * 2000  # 12,000 chars
    resp = client.post("/api/predict", json={"message": long_msg})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error"
    assert "10000" in body["message"]


def test_07_invalid_json_and_wrong_field_types_rejected() -> None:
    """Test 7: Invalid JSON and wrong field types are rejected cleanly with 422."""
    # Wrong field type (integer instead of string)
    resp_type = client.post("/api/predict", json={"message": 12345})
    assert resp_type.status_code == 422

    # Missing required field
    resp_missing = client.post("/api/predict", json={})
    assert resp_missing.status_code == 422

    # Unexpected extra field (extra='forbid')
    resp_extra = client.post(
        "/api/predict", json={"message": "Need water", "hack_field": True}
    )
    assert resp_extra.status_code == 422

    # Malformed raw JSON
    resp_raw = client.post(
        "/api/predict",
        content="{bad json",
        headers={"Content-Type": "application/json"},
    )
    assert resp_raw.status_code == 422


def test_08_output_label_ordering_matches_contract() -> None:
    """Test 8: Output label ordering matches exact canonical project contract."""
    resp = client.post(
        "/api/predict",
        json={"message": "We need drinking water, food, shelter, and medical help."},
    )
    assert resp.status_code == 200
    data = resp.json()

    cat_labels = [c["label"] for c in data["categories"]]
    assert cat_labels == ["request", "water", "food", "shelter", "medical_help"]
    assert list(data["predictions"].keys()) == TARGET_LABELS
    assert list(data["scores"].keys()) == TARGET_LABELS
    assert list(data["thresholds"].keys()) == TARGET_LABELS
    assert list(data["need_topic_predictions"].keys()) == [
        "water",
        "food",
        "shelter",
        "medical_help",
    ]


def test_09_frozen_thresholds_are_preserved() -> None:
    """Test 9: Frozen validation-selected thresholds are preserved across endpoints."""
    resp_pred = client.post("/api/predict", json={"message": "Need water immediately."})
    assert resp_pred.status_code == 200
    pred_thresholds = resp_pred.json()["thresholds"]
    for lbl, expected_tau in EXPECTED_FROZEN_THRESHOLDS.items():
        assert pred_thresholds[lbl] == pytest.approx(expected_tau)

    resp_eval = client.get("/api/evaluation")
    assert resp_eval.status_code == 200
    eval_thresholds = resp_eval.json()["frozen_thresholds"]
    for lbl, expected_tau in EXPECTED_FROZEN_THRESHOLDS.items():
        assert eval_thresholds[lbl] == pytest.approx(expected_tau)


def test_10_review_policy_consistency() -> None:
    """Test 10: Review policy output is consistent with thresholds and uncertainty band rules."""
    # High-probability request -> must recommend human review and have 'predicted_aid_request_verification'
    resp_high = client.post(
        "/api/predict",
        json={
            "message": "Please help us! We are stranded without water or food and need rescue shelter now."
        },
    )
    assert resp_high.status_code == 200
    high_data = resp_high.json()
    assert high_data["request_prediction"] == 1
    assert high_data["human_review_recommended"] is True
    assert "predicted_aid_request_verification" in high_data["review_reasons_list"]

    # Clear non-request news update -> routine monitoring if outside uncertainty bands
    resp_low = client.post(
        "/api/predict",
        json={
            "message": "The national symphony orchestra announced its autumn concert schedule in the capital."
        },
    )
    assert resp_low.status_code == 200
    low_data = resp_low.json()
    assert low_data["request_prediction"] == 0
    if not low_data["human_review_recommended"]:
        assert low_data["review_status_label"] == "Routine monitoring"
        assert (
            "routine_low_score_stream_retained_not_discarded"
            in low_data["review_reasons_list"]
        )


def test_11_simulated_model_failure_returns_structured_503_without_stack_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 11: Simulated model failure returns structured 503 error without stack trace."""
    missing_path = Path("non_existent_directory/missing_model.joblib")

    def mock_load_broken(*args, **kwargs):
        raise FileNotFoundError(f"Missing model at {missing_path}")

    monkeypatch.setattr(services, "load_inference_bundle", mock_load_broken)

    resp_predict = client.post("/api/predict", json={"message": "We need water."})
    assert resp_predict.status_code == 503
    err = resp_predict.json()
    assert err["error"] == "model_unavailable"
    assert "Traceback" not in json.dumps(err)
    assert "non_existent_directory" not in json.dumps(err)

    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200
    health_data = resp_health.json()
    assert health_data["status"] == "degraded"
    assert health_data["model_available"] is False


def test_12_evaluation_endpoint_loads_frozen_payload_cleanly() -> None:
    """Test 12: Evaluation endpoint loads the frozen Phase 2 evaluation payload cleanly."""
    resp = client.get("/api/evaluation")
    assert resp.status_code == 200
    data = resp.json()

    assert data["locked"] is True
    assert data["selected_model_id"] == "M1_logreg_C2.0_cwNone"
    assert data["test_dataset_rows"] == 2387

    frozen = data["held_out_test_metrics_frozen_thresholds"]
    assert frozen["macro_f1"] == pytest.approx(0.716627, abs=1e-5)
    assert frozen["micro_f1"] == pytest.approx(0.726061, abs=1e-5)
    assert frozen["request_f2"] == pytest.approx(0.778092, abs=1e-5)
    assert frozen["per_label"]["request"]["f1"] == pytest.approx(0.740385, abs=1e-5)
    assert frozen["per_label"]["water"]["f1"] == pytest.approx(0.810390, abs=1e-5)
    assert frozen["per_label"]["food"]["f1"] == pytest.approx(0.812925, abs=1e-5)
    assert frozen["per_label"]["shelter"]["f1"] == pytest.approx(0.727711, abs=1e-5)
    assert frozen["per_label"]["medical_help"]["f1"] == pytest.approx(0.491726, abs=1e-5)

    policy = data["human_review_policy_test"]
    assert policy["flagged_for_review_count"] == 810
    assert policy["true_requests_captured_in_review_queue"] == 422
    assert policy["unflagged_subset_missed_requests_count"] == 56


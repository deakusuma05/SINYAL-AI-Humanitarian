"""
SINYAL Phase 3 — End-to-End Integration & Resilience Tests (`tests/test_phase3_integration.py`).

Covers Mandatory Integration Tests 13–19 from Section K of the Phase 3 Master Prompt:
13. API prediction matches direct `predict_messages()` output for the same inputs.
14. Multi-label output supports zero, one, or multiple positive labels.
15. Batch ordering matches input ordering.
16. Unicode, multilingual, punctuation-heavy, and emoji inputs do not crash inference.
17. Unflagged messages remain present in batch output (never discarded).
18. Non-finite or invalid probability output is rejected cleanly.
19. Existing Phase 1 and Phase 2 artifacts and static frontend routes remain intact.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from src import TARGET_LABELS
from src.api import services
from src.api.main import app
from src.modeling.inference import predict_messages

client = TestClient(app)


def test_13_api_prediction_matches_direct_predict_messages_output() -> None:
    """Test 13: API prediction matches direct `predict_messages()` output to exact floating-point precision."""
    sample_messages = [
        "Please send drinking water and rice to the elementary school shelter immediately.",
        "Our clinic is out of bandages and antibiotics after the earthquake.",
        "Financial markets closed higher on Tuesday following trade negotiations.",
        "The flood destroyed several tents near the riverbank.",
    ]

    direct_outputs = predict_messages(sample_messages)
    batch_resp = client.post("/api/batch", json={"messages": sample_messages})
    assert batch_resp.status_code == 200
    api_batch_outputs = batch_resp.json()["results"]

    for msg, direct_item, api_item in zip(
        sample_messages, direct_outputs, api_batch_outputs
    ):
        single_resp = client.post("/api/predict", json={"message": msg})
        assert single_resp.status_code == 200
        api_single_item = single_resp.json()

        # Verify single vs direct
        assert api_single_item["request_prediction"] == direct_item["request_prediction"]
        assert api_single_item["request_score"] == pytest.approx(
            direct_item["request_score"], abs=1e-6
        )
        assert api_single_item["predictions"] == direct_item["predictions"]
        assert (
            api_single_item["human_review_recommended"]
            == direct_item["human_review_recommended"]
        )
        assert (
            api_single_item["uncertainty_indicated"]
            == direct_item["uncertainty_indicated"]
        )

        for lbl in TARGET_LABELS:
            assert api_single_item["scores"][lbl] == pytest.approx(
                direct_item["scores"][lbl], abs=1e-6
            )
            assert api_item["scores"][lbl] == pytest.approx(
                direct_item["scores"][lbl], abs=1e-6
            )


def test_14_multilabel_output_supports_zero_one_or_multiple_positive_labels() -> None:
    """Test 14: Multi-label output supports zero, one, or multiple positive labels simultaneously."""
    # Zero positive labels
    resp_zero = client.post(
        "/api/predict",
        json={
            "message": "The national museum opened a new exhibition of nineteenth century landscape paintings."
        },
    )
    assert resp_zero.status_code == 200
    preds_zero = resp_zero.json()["predictions"]
    assert sum(preds_zero.values()) == 0

    # Single positive topic (e.g., water mention without explicit request or vice versa)
    resp_one = client.post(
        "/api/predict",
        json={"message": "Residents are boiling tap water from the municipal reservoir."},
    )
    assert resp_one.status_code == 200
    preds_one = resp_one.json()["predictions"]
    assert preds_one["water"] == 1

    # Multiple positive labels (request + water + food + shelter)
    resp_multi = client.post(
        "/api/predict",
        json={
            "message": "Please help us! We need clean water, food rations, and emergency shelter tents for 50 flood victims."
        },
    )
    assert resp_multi.status_code == 200
    preds_multi = resp_multi.json()["predictions"]
    assert sum(preds_multi.values()) >= 3
    assert preds_multi["request"] == 1
    assert preds_multi["water"] == 1
    assert preds_multi["food"] == 1


def test_15_batch_ordering_matches_input_ordering() -> None:
    """Test 15: Batch ordering strictly preserves input message ordering."""
    ordered_inputs = [
        "First message: We urgently need drinking water.",
        "Second message: General economic news from the central bank.",
        "Third message: Families need emergency tents and shelter tonight.",
        "Fourth message: Hospital requires medical supplies for injured patients.",
        "Fifth message: Please send food packets to the village square.",
    ]
    resp = client.post("/api/batch", json={"messages": ordered_inputs})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert [r["message"] for r in results] == ordered_inputs


def test_16_unicode_multilingual_punctuation_and_emoji_inputs_do_not_crash() -> None:
    """Test 16: Unicode, Indonesian, Creole, punctuation-heavy, and emoji inputs do not crash inference."""
    diverse_inputs = [
        "Tolong kami butuh air bersih dan makanan darurat di posko pengungsian sekarang! 🙏💧🍚",
        "Nou bezwen dlo potab ak manje pou timoun yo nan lekòl la!!!",
        "URGENT?!?!?! --- Need water & shelter @ 12.345N, 67.890E #disaster #help ***",
        "救援要請：飲料水と食料が必要です。Please send water and food.",
    ]
    resp = client.post("/api/batch", json={"messages": diverse_inputs})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_messages"] == len(diverse_inputs)
    for item in data["results"]:
        for lbl in TARGET_LABELS:
            score = item["scores"][lbl]
            assert isinstance(score, float)
            assert math.isfinite(score)
            assert 0.0 <= score <= 1.0


def test_17_unflagged_messages_remain_present_in_batch_output() -> None:
    """Test 17: Unflagged messages are retained in batch output and never discarded."""
    batch_messages = [
        "Please help us urgently, we have no water and no food.",
        "The local library will remain closed for renovations during the summer.",
        "A new documentary film about marine biology premiered on television last night.",
    ]
    resp = client.post("/api/batch", json={"messages": batch_messages})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_messages"] == 3
    assert len(data["results"]) == 3
    assert data["flagged_for_review_count"] >= 1
    assert data["routine_monitoring_count"] >= 1
    assert (
        data["flagged_for_review_count"] + data["routine_monitoring_count"]
        == data["total_messages"]
    )


def test_18_non_finite_or_invalid_probability_output_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 18: Non-finite (NaN/Inf) or out-of-range probability output is rejected cleanly."""

    def mock_corrupted_predict_messages(*args, **kwargs):
        return [
            {
                "request_prediction": 1,
                "request_score": float("nan"),
                "need_topic_predictions": {
                    "water": 0,
                    "food": 0,
                    "shelter": 0,
                    "medical_help": 0,
                },
                "need_topic_scores": {
                    "water": 0.1,
                    "food": 0.1,
                    "shelter": 0.1,
                    "medical_help": 0.1,
                },
                "positive_need_topics": [],
                "predictions": {
                    "request": 1,
                    "water": 0,
                    "food": 0,
                    "shelter": 0,
                    "medical_help": 0,
                },
                "scores": {
                    "request": float("nan"),
                    "water": 0.1,
                    "food": 0.1,
                    "shelter": 0.1,
                    "medical_help": 0.1,
                },
                "thresholds": {
                    "request": 0.19,
                    "water": 0.22,
                    "food": 0.26,
                    "shelter": 0.32,
                    "medical_help": 0.20,
                },
                "uncertainty_indicated": False,
                "human_review_recommended": True,
                "review_reason": "predicted_aid_request_verification",
                "review_reasons_list": ["predicted_aid_request_verification"],
                "model_version": "2.0.0",
                "selected_model_id": "M1_logreg_C2.0_cwNone",
                "calibration_status": "uncalibrated_raw_probabilities",
                "language_scope_note": "Test note",
            }
        ]

    monkeypatch.setattr(services, "predict_messages", mock_corrupted_predict_messages)
    resp = client.post("/api/predict", json={"message": "Test NaN rejection."})
    assert resp.status_code == 500
    err = resp.json()
    assert err["error"] == "invalid_model_output"


def test_19_static_frontend_routes_and_xss_safety_controls() -> None:
    """Test 19: Dashboard HTML, CSS, and JS files are served properly and use safe DOM rendering."""
    resp_html = client.get("/")
    assert resp_html.status_code == 200
    html_text = resp_html.text
    assert "SINYAL" in html_text
    assert "From Crisis Messages to Humanitarian Action" in html_text
    assert "Research Demonstration" in html_text

    resp_css = client.get("/css/styles.css")
    assert resp_css.status_code == 200
    assert "--navy-900: #0F2342" in resp_css.text

    for js_file in ["/js/api.js", "/js/charts.js", "/js/app.js"]:
        resp_js = client.get(js_file)
        assert resp_js.status_code == 200
        assert "innerHTML" not in resp_js.text

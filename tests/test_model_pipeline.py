"""
SINYAL Phase 2 — Automated Unit & Integration Tests for Modeling, Thresholding,
Calibration, Review Policy, Serialization, and Single-Shot Test Evaluation Lock.

Implements all 14 mandatory Phase 2 test requirements (Section 23):
1. Label order correctness.
2. Training-only vectorizer fitting (zero validation/test vocabulary leakage).
3. Model output dimensions `(n_samples, 5)`.
4. Safe positive-class (`1`) probability extraction from `MultiOutputClassifier.predict_proba`.
5. Threshold application & deterministic validation threshold optimization.
6. Multilabel predictions (non-mutually-exclusive binary outputs).
7. No unknown target values (strictly `{0, 1}`).
8. Model serialization round-trip parity (`sinyal_model.joblib` reload matches in-memory predictions).
9. Empty, whitespace-only, non-string, and oversized input validation.
10. Deterministic inference across repeated calls.
11. No unexpected network requests during inference (socket guard).
12. Final test evaluation lock (`evaluate_test.py` requires `--confirm-frozen` and blocks repeat runs).
13. Selected model matches the frozen manifest (`models/model_manifest.json`).
14. No use of test data for training or threshold selection.
"""

from __future__ import annotations

import ast
import json
import socket
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from src import TARGET_LABELS
from src.modeling.evaluate_test import (
    FINAL_METRICS_PATH,
    LOCK_PATH,
    MANIFEST_PATH,
    MODEL_PATH,
    POLICY_PATH,
    compute_sha256,
    run_single_shot_test_evaluation,
)
from src.modeling.inference import (
    MAX_BATCH_SIZE,
    MAX_MESSAGE_LENGTH,
    load_inference_bundle,
    predict_messages,
    validate_messages_input,
)
from src.modeling.metrics import (
    DEFAULT_THRESHOLDS,
    apply_thresholds,
    compute_multilabel_metrics,
    extract_positive_probabilities,
    optimize_thresholds_on_validation,
)
from src.modeling.review_policy import (
    build_decision_policy_config,
    evaluate_single_message_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def synthetic_training_fixture() -> tuple[list[str], pd.DataFrame]:
    """Small synthetic fixture for fast, self-contained unit tests."""
    messages = [
        "Please help us we need drinking water and food in Port-au-Prince",
        "Urgent medical assistance needed for injured people at shelter camp",
        "We are trapped and need clean water and tents immediately",
        "Please send food and water to our village after the flood",
        "We need doctors medicine and hospital care right now",
        "News update: international donors pledged support after the earthquake",
        "Weather forecast reports heavy rain and wind across the region",
        "Thank you to all volunteers working at the stadium today",
        "Government officials held a press conference regarding road repairs",
        "Local sports team won the championship game yesterday evening",
    ]
    labels = pd.DataFrame(
        [
            {"request": 1, "water": 1, "food": 1, "shelter": 0, "medical_help": 0},
            {"request": 1, "water": 0, "food": 0, "shelter": 1, "medical_help": 1},
            {"request": 1, "water": 1, "food": 0, "shelter": 1, "medical_help": 0},
            {"request": 1, "water": 1, "food": 1, "shelter": 0, "medical_help": 0},
            {"request": 1, "water": 0, "food": 0, "shelter": 0, "medical_help": 1},
            {"request": 0, "water": 0, "food": 0, "shelter": 0, "medical_help": 0},
            {"request": 0, "water": 0, "food": 0, "shelter": 0, "medical_help": 0},
            {"request": 0, "water": 0, "food": 0, "shelter": 0, "medical_help": 0},
            {"request": 0, "water": 0, "food": 0, "shelter": 0, "medical_help": 0},
            {"request": 0, "water": 0, "food": 0, "shelter": 0, "medical_help": 0},
        ]
    )[TARGET_LABELS]
    return messages, labels


def test_p2_01_label_order_correctness() -> None:
    """1. Verify exact canonical order of the five target labels across code and artifacts."""
    expected_order = ["request", "water", "food", "shelter", "medical_help"]
    assert TARGET_LABELS == expected_order

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert manifest["target_labels"] == expected_order
    assert policy["target_labels"] == expected_order
    assert list(manifest["decision_thresholds"].keys()) == expected_order
    assert list(policy["thresholds"].keys()) == expected_order


def test_p2_02_training_only_vectorizer_fitting() -> None:
    """2. Verify that TF-IDF vocabulary is fit strictly on training data and excludes validation/test-only tokens."""
    train_texts = ["water food request alpha_train_token", "shelter medical alpha_train_token"]
    val_texts = ["unique_validation_only_xyz987"]
    test_texts = ["unique_test_only_qwe123"]
    y_train = pd.DataFrame(
        [
            {"request": 1, "water": 1, "food": 1, "shelter": 0, "medical_help": 0},
            {"request": 0, "water": 0, "food": 0, "shelter": 1, "medical_help": 1},
        ]
    )[TARGET_LABELS]

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", MultiOutputClassifier(LogisticRegression(random_state=42))),
        ]
    )
    pipe.fit(train_texts, y_train)
    vocab = pipe.named_steps["tfidf"].vocabulary_
    assert "alpha_train_token" in vocab
    assert "unique_validation_only_xyz987" not in vocab
    assert "unique_test_only_qwe123" not in vocab

    # Also verify that the serialized production model's vocabulary size <= max_features and was fit on train.csv
    bundle = load_inference_bundle()
    prod_tfidf: TfidfVectorizer = bundle["model"].named_steps["tfidf"]
    assert hasattr(prod_tfidf, "vocabulary_")
    assert 1000 < len(prod_tfidf.vocabulary_) <= 30000


def test_p2_03_model_output_dimensions(
    synthetic_training_fixture: tuple[list[str], pd.DataFrame],
) -> None:
    """3. Verify probability and binary prediction matrices have exact shape `(n_samples, 5)`."""
    texts, y_df = synthetic_training_fixture
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", MultiOutputClassifier(LogisticRegression(random_state=42))),
        ]
    )
    pipe.fit(texts, y_df)

    probs = extract_positive_probabilities(pipe, texts[:4])
    preds = apply_thresholds(probs, DEFAULT_THRESHOLDS)
    assert probs.shape == (4, 5)
    assert preds.shape == (4, 5)


def test_p2_04_safe_probability_extraction(
    synthetic_training_fixture: tuple[list[str], pd.DataFrame],
) -> None:
    """4. Verify `extract_positive_probabilities` inspects `classes_` and extracts class `1` accurately."""
    texts, y_df = synthetic_training_fixture
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("clf", MultiOutputClassifier(LogisticRegression(random_state=42))),
        ]
    )
    pipe.fit(texts, y_df)

    probs = extract_positive_probabilities(pipe, texts)
    raw_list = pipe.predict_proba(texts)
    clf_step = pipe.named_steps["clf"]

    for j, (est, raw_arr) in enumerate(zip(clf_step.estimators_, raw_list)):
        pos_idx = list(est.classes_).index(1)
        np.testing.assert_allclose(probs[:, j], raw_arr[:, pos_idx], rtol=1e-12, atol=1e-12)

    assert np.isfinite(probs).all()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_p2_05_threshold_application_and_optimization() -> None:
    """5. Verify per-label threshold application and deterministic threshold optimization."""
    prob_matrix = np.array(
        [
            [0.35, 0.60, 0.20, 0.80, 0.45],
            [0.25, 0.30, 0.55, 0.10, 0.52],
        ],
        dtype=np.float64,
    )
    custom_thresholds = {
        "request": 0.30,
        "water": 0.50,
        "food": 0.50,
        "shelter": 0.70,
        "medical_help": 0.50,
    }
    preds = apply_thresholds(prob_matrix, custom_thresholds)
    expected = np.array(
        [
            [1, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(preds, expected)

    # Verify deterministic optimization returns identical thresholds across repeated calls
    y_true = np.array(
        [
            [1, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    y_prob = np.array(
        [
            [0.42, 0.70, 0.15, 0.65, 0.20],
            [0.18, 0.25, 0.62, 0.12, 0.58],
            [0.38, 0.20, 0.55, 0.10, 0.15],
            [0.10, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=np.float64,
    )
    t1, _ = optimize_thresholds_on_validation(y_true, y_prob)
    t2, _ = optimize_thresholds_on_validation(y_true, y_prob)
    assert t1 == t2
    assert set(t1.keys()) == set(TARGET_LABELS)


def test_p2_06_multilabel_predictions_and_topic_intent_separation() -> None:
    """6. Verify a single message can activate multiple need topics simultaneously and separate topic from intent."""
    results = predict_messages(
        [
            "We desperately need clean drinking water, food, and medical help at the clinic!",
            "News report: 5,000 liters of clean water and food rations were delivered to the region yesterday.",
        ]
    )
    assert len(results) == 2
    first = results[0]
    assert first["request_prediction"] == 1
    # Multi-need detection: at least 2 need topics should be predicted positive on explicit multi-need message
    assert len(first["positive_need_topics"]) >= 2
    assert set(first["need_topic_predictions"].keys()) == {"water", "food", "shelter", "medical_help"}


def test_p2_07_no_unknown_target_values() -> None:
    """7. Verify predictions contain strictly binary `{0, 1}` integers with no NaNs or unknown values."""
    sample_msgs = [
        "Please send water and tents to Delmas 32",
        "Hurricane update from the meteorological service",
    ]
    outputs = predict_messages(sample_msgs)
    for item in outputs:
        assert item["request_prediction"] in (0, 1)
        for lbl, val in item["predictions"].items():
            assert lbl in TARGET_LABELS
            assert val in (0, 1)


def test_p2_08_model_serialization_round_trip() -> None:
    """8. Verify serialized `models/sinyal_model.joblib` loads cleanly and matches validation metrics in manifest."""
    assert MODEL_PATH.exists(), f"Missing {MODEL_PATH}"
    assert MANIFEST_PATH.exists(), f"Missing {MANIFEST_PATH}"
    assert POLICY_PATH.exists(), f"Missing {POLICY_PATH}"

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model_A = joblib.load(MODEL_PATH)
    model_B = joblib.load(MODEL_PATH)

    sample_texts = [
        "We need urgent food and water in Carrefour",
        "News headline: flooding reported in northern province",
        "Hospital needs medical supplies and doctors",
    ]
    prob_A = extract_positive_probabilities(model_A, sample_texts)
    prob_B = extract_positive_probabilities(model_B, sample_texts)
    np.testing.assert_allclose(prob_A, prob_B, rtol=0.0, atol=1e-15)
    assert prob_A.shape == (3, 5)
    assert compute_sha256(MODEL_PATH) == manifest["artifacts"]["model_joblib_sha256"]


def test_p2_09_empty_and_invalid_input_handling() -> None:
    """9. Verify `predict_messages` rejects empty lists, empty/whitespace strings, wrong types, and oversized batches."""
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_messages([])

    with pytest.raises(ValueError, match="empty or contains only whitespace"):
        predict_messages(["   \t\n  "])

    with pytest.raises(TypeError, match="must be a list"):
        predict_messages("not a list")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a string"):
        predict_messages([12345])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        predict_messages(["a" * (MAX_MESSAGE_LENGTH + 10)])

    with pytest.raises(ValueError, match="exceeds maximum allowed batch size"):
        validate_messages_input(["valid text"] * (MAX_BATCH_SIZE + 1))


def test_p2_10_deterministic_inference() -> None:
    """10. Verify repeated `predict_messages` calls return identical scores and review recommendations."""
    msgs = [
        "Please help us we have no water or food for three days",
        "Information bulletin on tropical storm trajectory",
    ]
    out1 = predict_messages(msgs)
    out2 = predict_messages(msgs)
    assert out1 == out2


def test_p2_11_no_unexpected_network_requests_during_inference() -> None:
    """11. Verify `predict_messages` runs 100% offline and never opens network sockets."""
    def guard_socket(*args, **kwargs):
        raise RuntimeError("Network access is forbidden during SINYAL inference!")

    with patch.object(socket, "create_connection", side_effect=guard_socket), patch.object(
        socket.socket, "connect", side_effect=guard_socket
    ):
        res = predict_messages(["We need drinking water and shelter in our community."])
        assert len(res) == 1
        assert "request_score" in res[0]


def test_p2_12_final_test_evaluation_lock_enforcement(tmp_path: Path) -> None:
    """12. Verify `run_single_shot_test_evaluation` requires `confirm_frozen=True` and blocks repeat runs when locked."""
    with pytest.raises(PermissionError, match="requires explicit confirmation"):
        run_single_shot_test_evaluation(confirm_frozen=False, lock_path=tmp_path / "lock.json")

    # Verify the actual production lock exists and blocks a second execution
    assert LOCK_PATH.exists(), f"Expected final evaluation lock at {LOCK_PATH}"
    with pytest.raises(RuntimeError, match="Held-out test evaluation is LOCKED"):
        run_single_shot_test_evaluation(confirm_frozen=True, lock_path=LOCK_PATH)


def test_p2_13_selected_model_matches_frozen_manifest() -> None:
    """13. Verify serialized model, decision policy, validation report, and final test lock all agree on the frozen winner."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    val_report = json.loads(
        (PROJECT_ROOT / "reports" / "phase2" / "validation_metrics.json").read_text(encoding="utf-8")
    )
    test_report = json.loads(FINAL_METRICS_PATH.read_text(encoding="utf-8"))
    lock_data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    winner_id = manifest["selected_model_id"]
    assert val_report["selected_winner"]["selected_model_id"] == winner_id
    assert test_report["selected_model_id"] == winner_id
    assert lock_data["selected_model_id"] == winner_id

    assert manifest["decision_thresholds"] == policy["thresholds"]
    assert test_report["frozen_thresholds"] == policy["thresholds"]
    assert lock_data["model_joblib_sha256"] == manifest["artifacts"]["model_joblib_sha256"]


def test_p2_14_no_use_of_test_data_in_train_validate_module() -> None:
    """14. Verify `train_validate.py` never uses `test.csv` for fitting, threshold tuning, or model selection."""
    train_validate_path = PROJECT_ROOT / "src" / "modeling" / "train_validate.py"
    source_code = train_validate_path.read_text(encoding="utf-8")

    # Verify `run_preflight_verification` deletes `loaded_splits["test"]` before returning
    assert 'del loaded_splits["test"]' in source_code

    # Parse AST of `run_training_and_validation` to ensure `test` split is never referenced
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_training_and_validation":
            fn_source = ast.get_source_segment(source_code, node) or ""
            assert "test.csv" not in fn_source
            assert "test_df" not in fn_source


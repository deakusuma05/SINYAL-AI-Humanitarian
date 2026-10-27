"""
SINYAL Phase 2 — Human-Review Decision & Uncertainty Policy Module.

Implements Section 13 of the Phase 2 Master Prompt:
- Evaluates request score, four need-topic scores, five validation-selected decision thresholds,
  and validation-defined uncertainty margins.
- Produces structured review recommendations and human-readable reasons for the volunteer dashboard.
- Never permanently discards or automatically rejects messages.
- Never assigns clinical urgency or treats uncertain predictions as verified emergencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score

from src import TARGET_LABELS

TOPIC_LABELS = ["water", "food", "shelter", "medical_help"]
DEFAULT_UNCERTAINTY_HALF_WIDTH = 0.08


def build_decision_policy_config(
    thresholds: dict[str, float],
    uncertainty_half_width: float = DEFAULT_UNCERTAINTY_HALF_WIDTH,
    implicit_request_ratio: float = 0.50,
) -> dict[str, Any]:
    """
    Construct the serializable decision policy dictionary (`models/decision_policy.json`).
    """
    bands: dict[str, dict[str, float]] = {}
    for lbl in TARGET_LABELS:
        tau = float(thresholds[lbl])
        low = round(max(0.02, tau - uncertainty_half_width), 4)
        high = round(min(0.98, tau + uncertainty_half_width), 4)
        bands[lbl] = {
            "threshold": round(tau, 4),
            "uncertainty_half_width": round(float(uncertainty_half_width), 4),
            "lower_bound": low,
            "upper_bound": high,
        }

    return {
        "policy_version": "1.0.0",
        "target_labels": TARGET_LABELS,
        "topic_labels": TOPIC_LABELS,
        "thresholds": {k: round(float(thresholds[k]), 4) for k in TARGET_LABELS},
        "uncertainty_half_width": round(float(uncertainty_half_width), 4),
        "implicit_request_ratio": round(float(implicit_request_ratio), 4),
        "uncertainty_bands": bands,
        "safety_principles": [
            "Decision-support prototype only: never automatically rejects or deletes messages.",
            "All predicted aid requests are routed to human verification.",
            "Borderline request scores and topic-positive messages with near-threshold request scores are flagged for human review.",
            "Scores are model-estimated probabilities/scores, not clinical urgency or verified authenticity.",
        ],
    }


def evaluate_single_message_policy(
    scores: dict[str, float],
    policy_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply the human-review decision policy to a single message's 5 model scores.
    Returns predicted binary flags, uncertainty indicators, review recommendation, and reasons.
    """
    thresholds = policy_config["thresholds"]
    bands = policy_config["uncertainty_bands"]
    implicit_ratio = float(policy_config.get("implicit_request_ratio", 0.50))

    req_score = float(scores["request"])
    req_tau = float(thresholds["request"])
    req_band = bands["request"]
    req_pred = int(req_score >= req_tau)
    req_in_band = bool(req_band["lower_bound"] <= req_score <= req_band["upper_bound"])

    topic_predictions: dict[str, int] = {}
    topic_scores: dict[str, float] = {}
    borderline_topics: list[str] = []
    positive_topics: list[str] = []

    for t_lbl in TOPIC_LABELS:
        sc = float(scores[t_lbl])
        tau = float(thresholds[t_lbl])
        bnd = bands[t_lbl]
        pred = int(sc >= tau)
        topic_predictions[t_lbl] = pred
        topic_scores[t_lbl] = round(sc, 6)
        if pred == 1:
            positive_topics.append(t_lbl)
        if bnd["lower_bound"] <= sc <= bnd["upper_bound"]:
            borderline_topics.append(t_lbl)

    reasons: list[str] = []
    uncertainty_flag = False

    # Rule 1: Predicted aid request -> route to human verification queue
    if req_pred == 1:
        reasons.append("predicted_aid_request_verification")
        if req_in_band:
            uncertainty_flag = True
            reasons.append("borderline_positive_request_score")

    # Rule 2: Borderline negative request score within lower uncertainty band
    elif req_score >= req_band["lower_bound"]:
        uncertainty_flag = True
        reasons.append("borderline_request_score_below_threshold")

    # Rule 3: Positive need topic(s) detected with moderate/near-threshold request score
    if req_pred == 0 and len(positive_topics) > 0 and req_score >= (implicit_ratio * req_tau):
        uncertainty_flag = True
        if "topic_detected_with_moderate_request_score" not in reasons:
            reasons.append("topic_detected_with_moderate_request_score")

    # Rule 4: Borderline need topic(s) when request is positive or near-threshold
    if len(borderline_topics) > 0 and (req_pred == 1 or req_score >= (implicit_ratio * req_tau)):
        uncertainty_flag = True
        reasons.append(f"borderline_need_topic_score({','.join(borderline_topics)})")

    review_recommended = len(reasons) > 0
    if not reasons:
        primary_reason = "routine_low_score_stream_retained_not_discarded"
    else:
        primary_reason = "; ".join(reasons)

    return {
        "request_prediction": req_pred,
        "request_score": round(req_score, 6),
        "need_topic_predictions": topic_predictions,
        "need_topic_scores": topic_scores,
        "positive_need_topics": positive_topics,
        "uncertainty_indicated": bool(uncertainty_flag),
        "review_recommended": bool(review_recommended),
        "review_reason": primary_reason,
        "review_reasons_list": reasons,
    }


def evaluate_review_policy_on_split(
    y_true: np.ndarray | pd.DataFrame,
    y_prob: np.ndarray,
    policy_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Simulate the human-review policy across a dataset split (validation or held-out test)
    and compute coverage, review-flag rate, request recall among flagged messages,
    and metrics on the unflagged (confident non-request) subset.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true[TARGET_LABELS].to_numpy(dtype=np.int64)
    else:
        y_true_arr = np.asarray(y_true, dtype=np.int64)

    n_samples = y_true_arr.shape[0]
    thresholds = policy_config["thresholds"]
    y_pred_arr = np.column_stack(
        [(y_prob[:, i] >= float(thresholds[lbl])).astype(np.int64) for i, lbl in enumerate(TARGET_LABELS)]
    )

    flagged_mask = np.zeros(n_samples, dtype=bool)
    uncertain_mask = np.zeros(n_samples, dtype=bool)
    reason_counts: dict[str, int] = {}

    for i in range(n_samples):
        row_scores = {lbl: float(y_prob[i, j]) for j, lbl in enumerate(TARGET_LABELS)}
        dec = evaluate_single_message_policy(row_scores, policy_config)
        flagged_mask[i] = dec["review_recommended"]
        uncertain_mask[i] = dec["uncertainty_indicated"]
        for r in dec["review_reasons_list"]:
            base_r = r.split("(")[0]
            reason_counts[base_r] = reason_counts.get(base_r, 0) + 1

    flagged_count = int(flagged_mask.sum())
    unflagged_count = int((~flagged_mask).sum())
    review_rate = float(flagged_count / n_samples) if n_samples > 0 else 0.0
    unflagged_coverage = float(unflagged_count / n_samples) if n_samples > 0 else 0.0

    # True requests captured inside the human review queue
    req_true = y_true_arr[:, 0]
    req_pred = y_pred_arr[:, 0]
    total_true_requests = int(req_true.sum())
    true_requests_in_review_queue = int(req_true[flagged_mask].sum())
    request_queue_capture_recall = (
        float(true_requests_in_review_queue / total_true_requests) if total_true_requests > 0 else 0.0
    )

    # Compare against standard binary prediction recall (without borderline/uncertainty expansion)
    direct_pred_request_recall = float(recall_score(req_true, req_pred, zero_division=0))

    # Metrics on the unflagged (remaining) subset
    if unflagged_count > 0:
        unflagged_req_true = req_true[~flagged_mask]
        unflagged_req_pred = req_pred[~flagged_mask]
        unflagged_false_negatives = int(unflagged_req_true.sum())
        unflagged_req_accuracy = float(accuracy_score(unflagged_req_true, unflagged_req_pred))
        unflagged_macro_f1 = float(
            f1_score(y_true_arr[~flagged_mask], y_pred_arr[~flagged_mask], average="macro", zero_division=0)
        )
        unflagged_micro_f1 = float(
            f1_score(y_true_arr[~flagged_mask], y_pred_arr[~flagged_mask], average="micro", zero_division=0)
        )
    else:
        unflagged_false_negatives = 0
        unflagged_req_accuracy = 0.0
        unflagged_macro_f1 = 0.0
        unflagged_micro_f1 = 0.0

    # Sensitivity curve across candidate uncertainty half-widths [0.00, 0.04, 0.08, 0.12, 0.16]
    sensitivity_curve: list[dict[str, float]] = []
    for hw in [0.00, 0.04, 0.08, 0.12, 0.16]:
        temp_cfg = build_decision_policy_config(thresholds, uncertainty_half_width=hw)
        f_mask = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            row_scores = {lbl: float(y_prob[i, j]) for j, lbl in enumerate(TARGET_LABELS)}
            f_mask[i] = evaluate_single_message_policy(row_scores, temp_cfg)["review_recommended"]
        rev_r = float(f_mask.mean())
        cap_rec = float(req_true[f_mask].sum() / total_true_requests) if total_true_requests > 0 else 0.0
        rem_cov = float((~f_mask).mean())
        rem_acc = float(accuracy_score(req_true[~f_mask], req_pred[~f_mask])) if (~f_mask).any() else 1.0
        sensitivity_curve.append(
            {
                "uncertainty_half_width": hw,
                "review_flag_rate": round(rev_r, 6),
                "remaining_subset_coverage": round(rem_cov, 6),
                "request_queue_capture_recall": round(cap_rec, 6),
                "remaining_subset_request_accuracy": round(rem_acc, 6),
            }
        )

    return {
        "total_messages": n_samples,
        "flagged_for_review_count": flagged_count,
        "flagged_for_review_rate": round(review_rate, 6),
        "uncertainty_indicated_count": int(uncertain_mask.sum()),
        "uncertainty_indicated_rate": round(float(uncertain_mask.mean()), 6),
        "remaining_unflagged_count": unflagged_count,
        "remaining_unflagged_coverage": round(unflagged_coverage, 6),
        "total_true_requests": total_true_requests,
        "true_requests_captured_in_review_queue": true_requests_in_review_queue,
        "request_recall_among_flagged_queue": round(request_queue_capture_recall, 6),
        "direct_binary_request_recall_without_uncertainty_rules": round(direct_pred_request_recall, 6),
        "additional_true_requests_rescued_by_uncertainty_policy": int(
            true_requests_in_review_queue - int(req_true[req_pred == 1].sum())
        ),
        "unflagged_subset_missed_requests_count": unflagged_false_negatives,
        "unflagged_subset_request_negative_predictive_accuracy": round(unflagged_req_accuracy, 6),
        "unflagged_subset_macro_f1": round(unflagged_macro_f1, 6),
        "unflagged_subset_micro_f1": round(unflagged_micro_f1, 6),
        "reason_trigger_counts": reason_counts,
        "coverage_vs_review_sensitivity": sensitivity_curve,
    }


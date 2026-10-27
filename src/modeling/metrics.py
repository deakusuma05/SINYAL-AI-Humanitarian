"""
SINYAL Phase 2 — Multilabel Probability Extraction, Threshold Optimization, and Evaluation Metrics.

Implements Sections 8, 9, 10, 12, 14, and 16 of the Phase 2 Master Prompt:
- Safe positive-class probability extraction from scikit-learn MultiOutputClassifier / Pipeline.
- Primary metrics: Macro F1, Micro F1, per-label Precision, Recall, F1, and `request` F2.
- Supporting metrics: Average Precision (AP), confusion matrices, support, Brier score,
  secondary accuracy, balanced accuracy, and Hamming loss.
- Deterministic validation threshold tuning over [0.05, 0.95] (F2 for `request`, F1 for need topics).
- Exploratory ranking utility analysis (Recall@10%, Recall@20%, Recall@30% vs. seeded random baseline).
- Exploratory subgroup metrics by `genre` (`direct` vs. `news`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    hamming_loss,
    precision_score,
    recall_score,
)

from src import TARGET_LABELS

DEFAULT_THRESHOLDS: dict[str, float] = {lbl: 0.50 for lbl in TARGET_LABELS}


def extract_positive_probabilities(model: Any, messages: list[str] | pd.Series) -> np.ndarray:
    """
    Safely extract positive-class (class == 1) probabilities from a fitted multilabel pipeline
    or MultiOutputClassifier, returning a 2D float64 array of shape (n_samples, 5).

    Verifies:
    - Exact number of outputs matches len(TARGET_LABELS) == 5.
    - Each binary sub-estimator has class `1` in its `classes_` attribute.
    - Output shape is (n_samples, 5), finite, and within [0.0, 1.0].
    """
    n_samples = len(messages)
    if n_samples == 0:
        return np.empty((0, len(TARGET_LABELS)), dtype=np.float64)

    raw_probas = model.predict_proba(messages)
    if not isinstance(raw_probas, list):
        raise TypeError(
            f"Expected MultiOutputClassifier.predict_proba to return a list of arrays, got {type(raw_probas)}"
        )
    if len(raw_probas) != len(TARGET_LABELS):
        raise ValueError(
            f"Expected {len(TARGET_LABELS)} probability outputs, got {len(raw_probas)}"
        )

    # Locate the MultiOutputClassifier step if wrapped in a Pipeline
    clf_step = model.named_steps["clf"] if hasattr(model, "named_steps") and "clf" in model.named_steps else model
    estimators = getattr(clf_step, "estimators_", None)
    if estimators is None or len(estimators) != len(TARGET_LABELS):
        raise ValueError("Fitted MultiOutputClassifier is missing valid `estimators_` list.")

    prob_matrix = np.zeros((n_samples, len(TARGET_LABELS)), dtype=np.float64)

    for idx, (lbl, est, prob_arr) in enumerate(zip(TARGET_LABELS, estimators, raw_probas)):
        classes = list(getattr(est, "classes_", []))
        if 1 not in classes:
            raise ValueError(
                f"Sub-estimator for target '{lbl}' (index {idx}) does not contain positive class `1` in classes_={classes}"
            )
        pos_col_idx = classes.index(1)
        if prob_arr.ndim != 2 or prob_arr.shape[0] != n_samples or pos_col_idx >= prob_arr.shape[1]:
            raise ValueError(
                f"Unexpected probability array shape {prob_arr.shape} for target '{lbl}'"
            )
        prob_matrix[:, idx] = prob_arr[:, pos_col_idx]

    if not np.isfinite(prob_matrix).all():
        raise ValueError("Extracted probability matrix contains NaN or infinite values.")
    if (prob_matrix < -1e-6).any() or (prob_matrix > 1.0 + 1e-6).any():
        raise ValueError(
            f"Extracted probabilities lie outside [0, 1]: min={prob_matrix.min()}, max={prob_matrix.max()}"
        )

    return np.clip(prob_matrix, 0.0, 1.0)


def apply_thresholds(
    prob_matrix: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Convert a (n_samples, 5) probability matrix into a binary {0, 1} decision matrix
    using per-label thresholds in exact TARGET_LABELS order.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    if prob_matrix.ndim != 2 or prob_matrix.shape[1] != len(TARGET_LABELS):
        raise ValueError(
            f"Expected prob_matrix of shape (n_samples, {len(TARGET_LABELS)}), got {prob_matrix.shape}"
        )

    thresh_vec = np.array([float(thresholds[lbl]) for lbl in TARGET_LABELS], dtype=np.float64)
    if (thresh_vec <= 0.0).any() or (thresh_vec >= 1.0).any():
        raise ValueError(f"All decision thresholds must lie strictly within (0, 1): {thresholds}")

    return (prob_matrix >= thresh_vec).astype(np.int64)


def compute_multilabel_metrics(
    y_true: np.ndarray | pd.DataFrame,
    y_prob: np.ndarray,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Compute all pre-registered primary and supporting metrics for the 5 binary labels.
    Sets zero_division=0 for undefined precision/F-score cases as documented.
    """
    if isinstance(y_true, pd.DataFrame):
        y_true_arr = y_true[TARGET_LABELS].to_numpy(dtype=np.int64)
    else:
        y_true_arr = np.asarray(y_true, dtype=np.int64)

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    y_pred = apply_thresholds(y_prob, thresholds)
    n_samples = y_true_arr.shape[0]

    macro_f1 = float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true_arr, y_pred, average="micro", zero_division=0))
    hamming = float(hamming_loss(y_true_arr, y_pred))
    exact_match_subset_acc = float(accuracy_score(y_true_arr, y_pred))

    per_label: dict[str, dict[str, Any]] = {}
    brier_scores: list[float] = []
    ap_scores: list[float] = []

    for idx, lbl in enumerate(TARGET_LABELS):
        yt = y_true_arr[:, idx]
        yp = y_pred[:, idx]
        ypr = y_prob[:, idx]

        prec = float(precision_score(yt, yp, zero_division=0))
        rec = float(recall_score(yt, yp, zero_division=0))
        f1_val = float(f1_score(yt, yp, zero_division=0))
        f2_val = float(fbeta_score(yt, yp, beta=2.0, zero_division=0))
        ap_val = float(average_precision_score(yt, ypr))
        brier_val = float(brier_score_loss(yt, ypr))
        acc_secondary = float(accuracy_score(yt, yp))
        bal_acc = float(balanced_accuracy_score(yt, yp))

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        pos_support = int(yt.sum())
        neg_support = int(n_samples - pos_support)
        pred_pos_count = int(yp.sum())
        pred_pos_rate = float(pred_pos_count / n_samples) if n_samples > 0 else 0.0

        brier_scores.append(brier_val)
        ap_scores.append(ap_val)

        per_label[lbl] = {
            "label": lbl,
            "threshold": float(thresholds[lbl]),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1_val, 6),
            "f2": round(f2_val, 6),
            "average_precision": round(ap_val, 6),
            "brier_score": round(brier_val, 6),
            "accuracy_secondary": round(acc_secondary, 6),
            "balanced_accuracy": round(bal_acc, 6),
            "positive_support": pos_support,
            "negative_support": neg_support,
            "predicted_positive_count": pred_pos_count,
            "predicted_positive_rate": round(pred_pos_rate, 6),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    request_f2 = per_label["request"]["f2"]
    request_f1 = per_label["request"]["f1"]
    mean_ap = float(np.mean(ap_scores))
    mean_brier = float(np.mean(brier_scores))

    return {
        "n_samples": int(n_samples),
        "thresholds": {k: round(float(v), 4) for k, v in thresholds.items()},
        "macro_f1": round(macro_f1, 6),
        "micro_f1": round(micro_f1, 6),
        "request_f2": round(request_f2, 6),
        "request_f1": round(request_f1, 6),
        "mean_average_precision": round(mean_ap, 6),
        "mean_brier_score": round(mean_brier, 6),
        "hamming_loss": round(hamming, 6),
        "exact_match_subset_accuracy_secondary": round(exact_match_subset_acc, 6),
        "per_label": per_label,
    }


def optimize_thresholds_on_validation(
    y_val_true: np.ndarray | pd.DataFrame,
    y_val_prob: np.ndarray,
    min_thresh: float = 0.05,
    max_thresh: float = 0.95,
    step: float = 0.01,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """
    Perform deterministic validation threshold search over [0.05, 0.95] with step 0.01.
    - Objective for `request`: F2 score (beta=2.0).
    - Objective for `water`, `food`, `shelter`, `medical_help`: F1 score (beta=1.0).
    - Deterministic tie-breaking: closest to 0.50, then lower threshold (favoring recall).
    """
    if isinstance(y_val_true, pd.DataFrame):
        y_true_arr = y_val_true[TARGET_LABELS].to_numpy(dtype=np.int64)
    else:
        y_true_arr = np.asarray(y_val_true, dtype=np.int64)

    grid = np.round(np.arange(min_thresh, max_thresh + 1e-9, step), 2)
    selected_thresholds: dict[str, float] = {}
    tuning_details: dict[str, dict[str, Any]] = {}

    for idx, lbl in enumerate(TARGET_LABELS):
        yt = y_true_arr[:, idx]
        ypr = y_val_prob[:, idx]
        objective_name = "f2" if lbl == "request" else "f1"
        beta_val = 2.0 if lbl == "request" else 1.0

        # Evaluate default 0.50 first
        yp_def = (ypr >= 0.50).astype(np.int64)
        default_obj = float(fbeta_score(yt, yp_def, beta=beta_val, zero_division=0))
        default_prec = float(precision_score(yt, yp_def, zero_division=0))
        default_rec = float(recall_score(yt, yp_def, zero_division=0))
        default_f1 = float(f1_score(yt, yp_def, zero_division=0))
        default_f2 = float(fbeta_score(yt, yp_def, beta=2.0, zero_division=0))

        best_tau = 0.50
        best_score = -1.0
        curve_records: list[dict[str, float]] = []

        for tau in grid:
            tau_f = float(tau)
            yp = (ypr >= tau_f).astype(np.int64)
            score = round(float(fbeta_score(yt, yp, beta=beta_val, zero_division=0)), 8)
            prec = float(precision_score(yt, yp, zero_division=0))
            rec = float(recall_score(yt, yp, zero_division=0))
            f1_v = float(f1_score(yt, yp, zero_division=0))
            f2_v = float(fbeta_score(yt, yp, beta=2.0, zero_division=0))

            curve_records.append(
                {
                    "threshold": tau_f,
                    "objective_score": score,
                    "precision": round(prec, 6),
                    "recall": round(rec, 6),
                    "f1": round(f1_v, 6),
                    "f2": round(f2_v, 6),
                    "predicted_positive_rate": round(float(yp.mean()), 6),
                }
            )

            if score > best_score:
                best_score = score
                best_tau = tau_f
            elif score == best_score:
                # Deterministic tie-breaker: closer to 0.50, then smaller tau
                if abs(tau_f - 0.50) < abs(best_tau - 0.50) - 1e-9:
                    best_tau = tau_f
                elif abs( abs(tau_f - 0.50) - abs(best_tau - 0.50) ) <= 1e-9 and tau_f < best_tau:
                    best_tau = tau_f

        yp_best = (ypr >= best_tau).astype(np.int64)
        tuned_prec = float(precision_score(yt, yp_best, zero_division=0))
        tuned_rec = float(recall_score(yt, yp_best, zero_division=0))
        tuned_f1 = float(f1_score(yt, yp_best, zero_division=0))
        tuned_f2 = float(fbeta_score(yt, yp_best, beta=2.0, zero_division=0))

        selected_thresholds[lbl] = round(best_tau, 2)
        tuning_details[lbl] = {
            "label": lbl,
            "objective_metric": objective_name,
            "default_threshold": 0.50,
            "selected_threshold": round(best_tau, 2),
            "metric_before_tuning": round(default_obj, 6),
            "metric_after_tuning": round(float(best_score), 6),
            "default_precision": round(default_prec, 6),
            "default_recall": round(default_rec, 6),
            "default_f1": round(default_f1, 6),
            "default_f2": round(default_f2, 6),
            "default_positive_prediction_rate": round(float(yp_def.mean()), 6),
            "tuned_precision": round(tuned_prec, 6),
            "tuned_recall": round(tuned_rec, 6),
            "tuned_f1": round(tuned_f1, 6),
            "tuned_f2": round(tuned_f2, 6),
            "tuned_positive_prediction_rate": round(float(yp_best.mean()), 6),
            "grid_curve": curve_records,
        }

    return selected_thresholds, tuning_details


def compute_ranking_utility(
    y_request_true: np.ndarray | pd.Series,
    y_request_prob: np.ndarray,
    fractions: tuple[float, ...] = (0.10, 0.20, 0.30),
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Exploratory triage utility analysis (Section 14):
    Evaluate Recall@10%, Recall@20%, Recall@30% of messages reviewed when ranked by
    predicted `request` score descending, compared against a reproducible random-order baseline.
    """
    yt = np.asarray(y_request_true, dtype=np.int64)
    ypr = np.asarray(y_request_prob, dtype=np.float64)
    n_total = len(yt)
    total_positives = int(yt.sum())

    # Stable sort descending by request score
    order_model = np.argsort(-ypr, kind="mergesort")
    yt_sorted_model = yt[order_model]

    # Reproducible random order baseline (averaged over 20 seeded permutations for stability)
    rng = np.random.default_rng(random_seed)
    results_by_k: dict[str, Any] = {}

    for frac in fractions:
        k_count = max(1, int(round(frac * n_total)))
        captured_model = int(yt_sorted_model[:k_count].sum())
        recall_at_k = float(captured_model / total_positives) if total_positives > 0 else 0.0
        precision_at_k = float(captured_model / k_count)

        rand_recalls = []
        for _ in range(20):
            perm = rng.permutation(n_total)
            rand_captured = int(yt[perm[:k_count]].sum())
            rand_recalls.append(rand_captured / total_positives if total_positives > 0 else 0.0)
        random_recall_at_k = float(np.mean(rand_recalls))

        pct_label = f"top_{int(round(frac * 100))}pct"
        results_by_k[pct_label] = {
            "fraction_reviewed": frac,
            "messages_reviewed": k_count,
            "total_messages": n_total,
            "total_true_requests": total_positives,
            "requests_captured_by_model_ranking": captured_model,
            "model_recall_at_k": round(recall_at_k, 6),
            "model_precision_at_k": round(precision_at_k, 6),
            "random_baseline_recall_at_k": round(random_recall_at_k, 6),
            "lift_over_random": round(recall_at_k / random_recall_at_k, 4) if random_recall_at_k > 0 else 0.0,
        }

    return {
        "note": "Exploratory queue-ordering utility metric on historical messages; does not measure clinical urgency or field response time.",
        "metrics": results_by_k,
    }


def compute_genre_subgroup_metrics(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """
    Exploratory subgroup performance analysis across existing `genre` values (`direct`, `news`).
    Never invents non-existent subgroups.
    """
    y_true_arr = df[TARGET_LABELS].to_numpy(dtype=np.int64)
    y_pred_arr = apply_thresholds(y_prob, thresholds)
    genres = df["genre"].astype(str)

    subgroups: dict[str, Any] = {}
    for g_val in sorted(genres.unique()):
        mask = (genres == g_val).to_numpy()
        n_grp = int(mask.sum())
        if n_grp == 0:
            continue

        yt_g = y_true_arr[mask]
        yp_g = y_pred_arr[mask]
        ypr_g = y_prob[mask]

        req_yt = yt_g[:, 0]
        req_yp = yp_g[:, 0]
        req_pos = int(req_yt.sum())

        req_prec = float(precision_score(req_yt, req_yp, zero_division=0))
        req_rec = float(recall_score(req_yt, req_yp, zero_division=0))
        req_f1 = float(f1_score(req_yt, req_yp, zero_division=0))
        req_f2 = float(fbeta_score(req_yt, req_yp, beta=2.0, zero_division=0))
        macro_f1_g = float(f1_score(yt_g, yp_g, average="macro", zero_division=0))
        micro_f1_g = float(f1_score(yt_g, yp_g, average="micro", zero_division=0))

        per_lbl_g: dict[str, Any] = {}
        for idx, lbl in enumerate(TARGET_LABELS):
            per_lbl_g[lbl] = {
                "positive_support": int(yt_g[:, idx].sum()),
                "precision": round(float(precision_score(yt_g[:, idx], yp_g[:, idx], zero_division=0)), 6),
                "recall": round(float(recall_score(yt_g[:, idx], yp_g[:, idx], zero_division=0)), 6),
                "f1": round(float(f1_score(yt_g[:, idx], yp_g[:, idx], zero_division=0)), 6),
                "average_precision": round(
                    float(average_precision_score(yt_g[:, idx], ypr_g[:, idx])) if yt_g[:, idx].sum() > 0 else 0.0,
                    6,
                ),
            }

        subgroups[g_val] = {
            "genre": g_val,
            "record_count": n_grp,
            "request_positive_count": req_pos,
            "request_positive_rate": round(req_pos / n_grp, 6),
            "request_precision": round(req_prec, 6),
            "request_recall": round(req_rec, 6),
            "request_f1": round(req_f1, 6),
            "request_f2": round(req_f2, 6),
            "macro_f1": round(macro_f1_g, 6),
            "micro_f1": round(micro_f1_g, 6),
            "per_label": per_lbl_g,
        }

    return subgroups


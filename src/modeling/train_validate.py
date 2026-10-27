"""
SINYAL Phase 2 — Model Training, Validation Evaluation, Threshold Tuning, Calibration,
Interpretability, Error Analysis, and Winner Serialization (`src/modeling/train_validate.py`).

STRICT DATA USAGE CONTRACT:
- Fits TF-IDF vocabulary, IDF weights, classifiers, and optional CV calibration EXCLUSIVELY on `data/processed/train.csv`.
- Evaluates models, selects thresholds, assesses calibration, and conducts error analysis EXCLUSIVELY on `data/processed/validation.csv`.
- NEVER reads `data/processed/test.csv` for training, selection, or tuning (only verifies its file SHA-256 checksum during pre-flight).
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src import TARGET_LABELS
from src.modeling.metrics import (
    DEFAULT_THRESHOLDS,
    apply_thresholds,
    compute_genre_subgroup_metrics,
    compute_multilabel_metrics,
    compute_ranking_utility,
    extract_positive_probabilities,
    optimize_thresholds_on_validation,
)
from src.modeling.review_policy import (
    DEFAULT_UNCERTAINTY_HALF_WIDTH,
    build_decision_policy_config,
    evaluate_review_policy_on_split,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PHASE2_REPORTS_DIR = PROJECT_ROOT / "reports" / "phase2"
PHASE2_FIGURES_DIR = PHASE2_REPORTS_DIR / "figures"

EXPECTED_PHASE1_HASHES = {
    "train": {
        "path": PROCESSED_DIR / "train.csv",
        "expected_rows": 19032,
        "expected_sha256": "13d0c0336f5ed8bc89c19ee3529075e5638f7aaf1976f1aaeefd0a3a75ecb02b",
    },
    "validation": {
        "path": PROCESSED_DIR / "validation.csv",
        "expected_rows": 2329,
        "expected_sha256": "e09d9842d17414e08946739d93853db1687a873a08a25210ea65bb5aec7b722e",
    },
    "test": {
        "path": PROCESSED_DIR / "test.csv",
        "expected_rows": 2387,
        "expected_sha256": "d0f7265d67753a27a40a0fcc6caab8f88849b352427a71929ac692b37b06d75a",
    },
}

RANDOM_SEED = 42

TFIDF_PARAMS: dict[str, Any] = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.98,
    "max_features": 30000,
    "sublinear_tf": True,
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_preflight_verification() -> dict[str, Any]:
    """
    Verify Phase 1 processed files, SHA-256 hashes, row counts, target columns,
    binary values, and split separation before training.
    Note: Test split is verified for integrity/checksum only and NOT returned for modeling.
    """
    verification_info: dict[str, Any] = {}
    loaded_splits: dict[str, pd.DataFrame] = {}

    for split_name, spec in EXPECTED_PHASE1_HASHES.items():
        path: Path = spec["path"]
        if not path.exists():
            raise FileNotFoundError(f"Pre-flight check failed: missing {path}")
        actual_sha = compute_sha256(path)
        if actual_sha != spec["expected_sha256"]:
            raise ValueError(
                f"Pre-flight SHA-256 mismatch for {split_name}: "
                f"expected {spec['expected_sha256']}, got {actual_sha}"
            )
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        if len(df) != spec["expected_rows"]:
            raise ValueError(
                f"Pre-flight row count mismatch for {split_name}: "
                f"expected {spec['expected_rows']}, got {len(df)}"
            )
        for lbl in TARGET_LABELS:
            if lbl not in df.columns:
                raise ValueError(f"Missing target column '{lbl}' in {split_name}")
            vals = set(df[lbl].dropna().unique())
            if not vals.issubset({0, 1}):
                raise ValueError(f"Non-binary target values in {split_name}.{lbl}: {vals}")
            if split_name == "train" and vals != {0, 1}:
                raise ValueError(f"Training target '{lbl}' does not contain both classes 0 and 1!")

        if df["message"].isna().any() or (df["message"].astype(str).str.strip().str.len() == 0).any():
            raise ValueError(f"Unexpected empty or null message in {split_name}")

        verification_info[split_name] = {
            "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "rows": len(df),
            "sha256": actual_sha,
            "verified_match": True,
        }
        loaded_splits[split_name] = df

    # Verify zero ID or normalized_message overlap across splits
    t_ids, v_ids, te_ids = (
        set(loaded_splits["train"]["id"]),
        set(loaded_splits["validation"]["id"]),
        set(loaded_splits["test"]["id"]),
    )
    if (t_ids & v_ids) or (t_ids & te_ids) or (v_ids & te_ids):
        raise ValueError("Pre-flight check failed: duplicate IDs detected across splits!")

    print("[PRE-FLIGHT] All 3 processed dataset splits verified (exact SHA-256 & row count match).")
    # Discard test dataframe from memory immediately so training/validation cannot touch it
    del loaded_splits["test"]
    return {
        "verification": verification_info,
        "train_df": loaded_splits["train"],
        "val_df": loaded_splits["validation"],
    }


def build_candidate_models() -> list[dict[str, Any]]:
    """
    Construct the three model families (B0, B1, M1) with deterministic seeds.
    """
    candidates: list[dict[str, Any]] = [
        {
            "model_id": "B0_most_frequent",
            "family": "B0",
            "description": "Dummy Baseline (most_frequent)",
            "simplicity_rank": 1,
            "hyperparameters": {"strategy": "most_frequent", "random_state": RANDOM_SEED},
            "pipeline": Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(**TFIDF_PARAMS),
                    ),
                    (
                        "clf",
                        MultiOutputClassifier(
                            DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
                        ),
                    ),
                ]
            ),
        },
        {
            "model_id": "B0_stratified",
            "family": "B0",
            "description": "Dummy Baseline (stratified)",
            "simplicity_rank": 2,
            "hyperparameters": {"strategy": "stratified", "random_state": RANDOM_SEED},
            "pipeline": Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(**TFIDF_PARAMS),
                    ),
                    (
                        "clf",
                        MultiOutputClassifier(
                            DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
                        ),
                    ),
                ]
            ),
        },
    ]

    # B1 — Multinomial Naive Bayes grid
    for idx, alpha_val in enumerate([0.5, 1.0, 2.0], start=1):
        candidates.append(
            {
                "model_id": f"B1_nb_alpha_{alpha_val}",
                "family": "B1",
                "description": f"TF-IDF + MultinomialNB (alpha={alpha_val})",
                "simplicity_rank": 10 + idx,
                "hyperparameters": {
                    "tfidf": TFIDF_PARAMS,
                    "alpha": alpha_val,
                },
                "pipeline": Pipeline(
                    [
                        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
                        ("clf", MultiOutputClassifier(MultinomialNB(alpha=alpha_val))),
                    ]
                ),
            }
        )

    # M1 — Logistic Regression grid
    simplicity_counter = 20
    for cw_label, cw_val in [("cwNone", None), ("cwBalanced", "balanced")]:
        for c_val in [0.5, 1.0, 2.0]:
            simplicity_counter += 1
            candidates.append(
                {
                    "model_id": f"M1_logreg_C{c_val}_{cw_label}",
                    "family": "M1",
                    "description": f"TF-IDF + LogisticRegression (C={c_val}, class_weight={cw_val})",
                    "simplicity_rank": simplicity_counter,
                    "hyperparameters": {
                        "tfidf": TFIDF_PARAMS,
                        "C": c_val,
                        "class_weight": cw_val,
                        "solver": "lbfgs",
                        "max_iter": 1000,
                        "random_state": RANDOM_SEED,
                    },
                    "pipeline": Pipeline(
                        [
                            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
                            (
                                "clf",
                                MultiOutputClassifier(
                                    LogisticRegression(
                                        C=c_val,
                                        class_weight=cw_val,
                                        solver="lbfgs",
                                        max_iter=1000,
                                        random_state=RANDOM_SEED,
                                    )
                                ),
                            ),
                        ]
                    ),
                }
            )

    return candidates


def select_winning_model(evaluated_models: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Apply the pre-registered selection rule from `docs/phase2_selection_policy.md`:
    1. Highest validation Macro F1 using validation-tuned thresholds (within 0.0001 treated as tie).
    2. Highest validation `request` F2 using validation-tuned thresholds.
    3. Lower simplicity_rank (simpler model).
    """
    def sort_key(item: dict[str, Any]) -> tuple[float, float, int]:
        tuned = item["validation_tuned_metrics"]
        # Round macro_f1 to 4 decimals (0.0001 tolerance) for primary comparison
        macro_f1_rounded = round(float(tuned["macro_f1"]), 4)
        req_f2_rounded = round(float(tuned["request_f2"]), 6)
        return (macro_f1_rounded, req_f2_rounded, -int(item["simplicity_rank"]))

    ranked = sorted(evaluated_models, key=sort_key, reverse=True)
    return ranked[0]


def evaluate_calibration_and_optional_sigmoid(
    winner_entry: dict[str, Any],
    X_train: pd.Series,
    y_train: pd.DataFrame,
    X_val: pd.Series,
    y_val: pd.DataFrame,
) -> dict[str, Any]:
    """
    Assess probability calibration on validation (Brier score per label + reliability curves)
    and train a leakage-safe 3-fold CV sigmoid-calibrated version on `train.csv` to compare.
    """
    uncal_prob = winner_entry["val_prob_matrix"]
    uncal_tuned_metrics = winner_entry["validation_tuned_metrics"]
    uncal_brier_per_label = {
        lbl: uncal_tuned_metrics["per_label"][lbl]["brier_score"] for lbl in TARGET_LABELS
    }
    uncal_mean_brier = uncal_tuned_metrics["mean_brier_score"]

    # Fit a leakage-safe Sigmoid CalibratedClassifierCV(cv=3) on training data only
    hp = winner_entry["hyperparameters"]
    t0 = time.perf_counter()
    if winner_entry["family"] == "M1":
        base_est = LogisticRegression(
            C=hp["C"],
            class_weight=hp["class_weight"],
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_SEED,
        )
    elif winner_entry["family"] == "B1":
        base_est = MultinomialNB(alpha=hp["alpha"])
    else:
        base_est = DummyClassifier(strategy=hp["strategy"], random_state=RANDOM_SEED)

    cal_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            (
                "clf",
                MultiOutputClassifier(
                    CalibratedClassifierCV(estimator=base_est, method="sigmoid", cv=3)
                ),
            ),
        ]
    )
    cal_pipeline.fit(X_train, y_train[TARGET_LABELS])
    cal_train_sec = round(time.perf_counter() - t0, 3)

    cal_val_prob = extract_positive_probabilities(cal_pipeline, X_val)
    cal_default_metrics = compute_multilabel_metrics(y_val, cal_val_prob, DEFAULT_THRESHOLDS)
    cal_thresholds, cal_tuning_details = optimize_thresholds_on_validation(y_val, cal_val_prob)
    cal_tuned_metrics = compute_multilabel_metrics(y_val, cal_val_prob, cal_thresholds)

    cal_brier_per_label = {
        lbl: cal_tuned_metrics["per_label"][lbl]["brier_score"] for lbl in TARGET_LABELS
    }
    cal_mean_brier = cal_tuned_metrics["mean_brier_score"]

    # Check selection criterion from `docs/phase2_selection_policy.md` Section 5:
    # Select calibrated version if it improves mean Brier score AND maintains tuned Macro F1 and request F2 within 0.005
    brier_improved = cal_mean_brier < uncal_mean_brier
    macro_f1_diff = cal_tuned_metrics["macro_f1"] - uncal_tuned_metrics["macro_f1"]
    req_f2_diff = cal_tuned_metrics["request_f2"] - uncal_tuned_metrics["request_f2"]

    use_calibrated = bool(
        brier_improved and (macro_f1_diff >= -0.005) and (req_f2_diff >= -0.005)
    )

    # Compute reliability curve bins (10 bins) for `request` and `food` / `water` on validation
    reliability_curves: dict[str, Any] = {}
    y_val_arr = y_val[TARGET_LABELS].to_numpy(dtype=np.int64)
    for idx, lbl in enumerate(TARGET_LABELS):
        frac_pos_u, mean_pred_u = calibration_curve(
            y_val_arr[:, idx], uncal_prob[:, idx], n_bins=8, strategy="uniform"
        )
        frac_pos_c, mean_pred_c = calibration_curve(
            y_val_arr[:, idx], cal_val_prob[:, idx], n_bins=8, strategy="uniform"
        )
        reliability_curves[lbl] = {
            "uncalibrated": {
                "mean_predicted_value": [round(float(x), 6) for x in mean_pred_u],
                "fraction_of_positives": [round(float(x), 6) for x in frac_pos_u],
            },
            "sigmoid_calibrated": {
                "mean_predicted_value": [round(float(x), 6) for x in mean_pred_c],
                "fraction_of_positives": [round(float(x), 6) for x in frac_pos_c],
            },
        }

    return {
        "uncalibrated_mean_brier": uncal_mean_brier,
        "uncalibrated_brier_per_label": uncal_brier_per_label,
        "sigmoid_calibrated_mean_brier": cal_mean_brier,
        "sigmoid_calibrated_brier_per_label": cal_brier_per_label,
        "sigmoid_training_duration_sec": cal_train_sec,
        "sigmoid_default_metrics": cal_default_metrics,
        "sigmoid_tuned_metrics": cal_tuned_metrics,
        "sigmoid_selected_thresholds": cal_thresholds,
        "sigmoid_tuning_details": cal_tuning_details,
        "macro_f1_difference_cal_minus_uncal": round(macro_f1_diff, 6),
        "request_f2_difference_cal_minus_uncal": round(req_f2_diff, 6),
        "use_calibrated_pipeline": use_calibrated,
        "calibration_decision_explanation": (
            "Selected 3-fold CV sigmoid-calibrated pipeline because it improved mean validation Brier score "
            f"({uncal_mean_brier:.6f} -> {cal_mean_brier:.6f}) while keeping tuned Macro F1 "
            f"(diff={macro_f1_diff:+.4f}) and request F2 (diff={req_f2_diff:+.4f}) within the 0.005 tolerance."
            if use_calibrated
            else f"Retained uncalibrated pipeline because sigmoid calibration changed Macro F1 by {macro_f1_diff:+.4f} "
            f"and request F2 by {req_f2_diff:+.4f} (or did not improve Brier score {uncal_mean_brier:.6f} vs {cal_mean_brier:.6f})."
        ),
        "calibrated_pipeline_object": cal_pipeline,
        "calibrated_val_prob_matrix": cal_val_prob,
        "reliability_curves": reliability_curves,
    }


def extract_interpretability_features(
    fitted_pipeline: Pipeline,
    top_k: int = 15,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Extract top positive and negative lexical features per binary target label.
    Supports both direct LogisticRegression/MultinomialNB sub-estimators and
    CalibratedClassifierCV-wrapped LogisticRegression sub-estimators (by averaging
    coefficients across the CV fold base estimators).
    """
    tfidf: TfidfVectorizer = fitted_pipeline.named_steps["tfidf"]
    feature_names = np.asarray(tfidf.get_feature_names_out())
    clf_multi: MultiOutputClassifier = fitted_pipeline.named_steps["clf"]

    interpretability: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for idx, (lbl, est) in enumerate(zip(TARGET_LABELS, clf_multi.estimators_)):
        coef_vec: np.ndarray | None = None
        weight_type = "linear_coefficient"

        if hasattr(est, "coef_"):
            coef_vec = np.asarray(est.coef_).ravel()
        elif isinstance(est, CalibratedClassifierCV) and hasattr(est, "calibrated_classifiers_"):
            fold_coefs = []
            for cal_clf in est.calibrated_classifiers_:
                base_e = getattr(cal_clf, "estimator", None)
                if base_e is not None and hasattr(base_e, "coef_"):
                    fold_coefs.append(np.asarray(base_e.coef_).ravel())
                elif base_e is not None and hasattr(base_e, "feature_log_prob_"):
                    diff = base_e.feature_log_prob_[1] - base_e.feature_log_prob_[0]
                    fold_coefs.append(np.asarray(diff).ravel())
                    weight_type = "log_prob_difference"
            if fold_coefs:
                coef_vec = np.mean(np.vstack(fold_coefs), axis=0)
        elif hasattr(est, "feature_log_prob_"):
            coef_vec = np.asarray(est.feature_log_prob_[1] - est.feature_log_prob_[0]).ravel()
            weight_type = "log_prob_difference"

        if coef_vec is None or len(coef_vec) != len(feature_names):
            interpretability[lbl] = {"top_positive": [], "top_negative": []}
            continue

        top_pos_idx = np.argsort(coef_vec)[::-1][:top_k]
        top_neg_idx = np.argsort(coef_vec)[:top_k]

        interpretability[lbl] = {
            "weight_type": weight_type,
            "top_positive": [
                {"term": str(feature_names[i]), "weight": round(float(coef_vec[i]), 5)}
                for i in top_pos_idx
            ],
            "top_negative": [
                {"term": str(feature_names[i]), "weight": round(float(coef_vec[i]), 5)}
                for i in top_neg_idx
            ],
        }

    return interpretability


def redact_snippet(text: str, max_len: int = 110) -> str:
    """
    Create a safe, privacy-preserving snippet for error analysis reports.
    Redacts any digit sequences >= 4 digits, email patterns, or capitalized proper names after 'my name is'.
    """
    s = re.sub(r"\s+", " ", str(text)).strip()
    s = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", s)
    s = re.sub(r"\d{4,}", "[REDACTED_NUM]", s)
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "..."
    return s


def run_validation_error_analysis(
    val_df: pd.DataFrame,
    y_val_prob: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """
    Analyze validation errors across the 6 mandatory error categories (Section 15):
    1. False negative aid requests (`request` true=1, pred=0).
    2. False positive informational messages (`request` true=0, pred=1).
    3. Multi-label prediction failures (messages with >=2 true needs where >=1 need is missed).
    4. Short messages (<= 45 characters or <= 7 words).
    5. Translation-related ambiguity (`original` is non-null vs. English-only).
    6. Confusion between `direct` and `news` genres.
    """
    y_true = val_df[TARGET_LABELS].to_numpy(dtype=np.int64)
    y_pred = apply_thresholds(y_val_prob, thresholds)

    req_true = y_true[:, 0]
    req_pred = y_pred[:, 0]
    req_prob = y_val_prob[:, 0]

    # 1. False negative aid requests
    fn_mask = (req_true == 1) & (req_pred == 0)
    fn_df = val_df.loc[fn_mask].copy()
    fn_df["request_score"] = req_prob[fn_mask]
    fn_df = fn_df.sort_values("request_score", ascending=True)
    fn_examples = []
    for _, row in fn_df.head(6).iterrows():
        fn_examples.append(
            {
                "id": int(row["id"]),
                "genre": str(row["genre"]),
                "has_translation_original": bool(pd.notna(row["original"])),
                "request_score": round(float(row["request_score"]), 4),
                "true_topics": [lbl for lbl in TARGET_LABELS[1:] if int(row[lbl]) == 1],
                "redacted_excerpt": redact_snippet(row["message"]),
            }
        )

    # 2. False positive informational messages
    fp_mask = (req_true == 0) & (req_pred == 1)
    fp_df = val_df.loc[fp_mask].copy()
    fp_df["request_score"] = req_prob[fp_mask]
    fp_df = fp_df.sort_values("request_score", ascending=False)
    fp_examples = []
    for _, row in fp_df.head(6).iterrows():
        fp_examples.append(
            {
                "id": int(row["id"]),
                "genre": str(row["genre"]),
                "has_translation_original": bool(pd.notna(row["original"])),
                "request_score": round(float(row["request_score"]), 4),
                "true_topics": [lbl for lbl in TARGET_LABELS[1:] if int(row[lbl]) == 1],
                "redacted_excerpt": redact_snippet(row["message"]),
            }
        )

    # 3. Multi-label need prediction failures (>=2 true need topics, at least 1 missed)
    topic_true = y_true[:, 1:]
    topic_pred = y_pred[:, 1:]
    multi_need_mask = topic_true.sum(axis=1) >= 2
    multi_need_partial_miss_mask = multi_need_mask & ((topic_true & (topic_pred == 0)).sum(axis=1) >= 1)
    multi_need_Exact_match_count = int((multi_need_mask & ((topic_true == topic_pred).all(axis=1))).sum())

    multi_examples = []
    for idx_pos in np.where(multi_need_partial_miss_mask)[0][:6]:
        row = val_df.iloc[idx_pos]
        t_true_list = [TARGET_LABELS[1 + j] for j in range(4) if topic_true[idx_pos, j] == 1]
        t_pred_list = [TARGET_LABELS[1 + j] for j in range(4) if topic_pred[idx_pos, j] == 1]
        multi_examples.append(
            {
                "id": int(row["id"]),
                "genre": str(row["genre"]),
                "true_topics": t_true_list,
                "predicted_topics": t_pred_list,
                "redacted_excerpt": redact_snippet(row["message"]),
            }
        )

    # 4. Short messages analysis (<= 8 words)
    word_counts = val_df["message"].astype(str).str.split().str.len().to_numpy()
    short_mask = word_counts <= 8
    short_req_errors = int((short_mask & (req_true != req_pred)).sum())
    short_total = int(short_mask.sum())

    # 5. Translated (`original` non-null) vs Untranslated (`original` null) error rates
    trans_mask = val_df["original"].notna().to_numpy()
    trans_Metrics = compute_multilabel_metrics(y_true[trans_mask], y_val_prob[trans_mask], thresholds)
    untrans_Metrics = compute_multilabel_metrics(y_true[~trans_mask], y_val_prob[~trans_mask], thresholds)

    # 6. Genre subgroup comparison (`direct` vs `news`)
    genre_subgroups = compute_genre_subgroup_metrics(val_df, y_val_prob, thresholds)

    return {
        "false_negative_requests": {
            "count": int(fn_mask.sum()),
            "total_true_requests": int(req_true.sum()),
            "false_negative_rate": round(float(fn_mask.sum() / max(1, req_true.sum())), 6),
            "by_genre": val_df.loc[fn_mask, "genre"].value_counts().to_dict(),
            "representative_examples": fn_examples,
        },
        "false_positive_requests": {
            "count": int(fp_mask.sum()),
            "total_true_non_requests": int((req_true == 0).sum()),
            "false_positive_rate": round(float(fp_mask.sum() / max(1, (req_true == 0).sum())), 6),
            "by_genre": val_df.loc[fp_mask, "genre"].value_counts().to_dict(),
            "representative_examples": fp_examples,
        },
        "multilabel_cooccurrence_errors": {
            "messages_with_2plus_true_needs": int(multi_need_mask.sum()),
            "messages_with_at_least_one_missed_need": int(multi_need_partial_miss_mask.sum()),
            "messages_with_exact_need_match": multi_need_Exact_match_count,
            "representative_examples": multi_examples,
        },
        "short_messages_8_words_or_fewer": {
            "total_short_messages": short_total,
            "request_misclassifications": short_req_errors,
            "error_rate": round(float(short_req_errors / max(1, short_total)), 6),
        },
        "translation_comparison": {
            "translated_messages_count": int(trans_mask.sum()),
            "translated_macro_f1": trans_Metrics["macro_f1"],
            "translated_request_f2": trans_Metrics["request_f2"],
            "english_only_messages_count": int((~trans_mask).sum()),
            "english_only_macro_f1": untrans_Metrics["macro_f1"],
            "english_only_request_f2": untrans_Metrics["request_f2"],
        },
        "genre_subgroups": genre_subgroups,
    }


def generate_validation_figures(
    evaluated_models: list[dict[str, Any]],
    final_winner_info: dict[str, Any],
    y_val_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> list[str]:
    """
    Generate publication-ready validation & analysis figures in `reports/phase2/figures/`.
    """
    PHASE2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    NAVY = "#1B365D"
    TEAL = "#008080"
    ORANGE = "#E06D53"
    SLATE = "#4A607A"
    GOLD = "#D99B26"

    saved_files: list[str] = []

    # Pick best representative from each family B0, B1, M1 for clean charts + full 11-model chart
    best_b0 = max([m for m in evaluated_models if m["family"] == "B0"], key=lambda x: x["validation_tuned_metrics"]["macro_f1"])
    best_b1 = max([m for m in evaluated_models if m["family"] == "B1"], key=lambda x: x["validation_tuned_metrics"]["macro_f1"])
    best_m1 = max([m for m in evaluated_models if m["family"] == "M1"], key=lambda x: x["validation_tuned_metrics"]["macro_f1"])

    # Figure 1: Validation comparison of B0, B1, and M1 (Macro F1, Micro F1, Request F2)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    families = [
        f"{best_b0['model_id']}\n(Dummy)",
        f"{best_b1['model_id']}\n(Naive Bayes)",
        f"{best_m1['model_id']}\n(Logistic Reg.)",
        f"Selected Winner\n({final_winner_info['selected_model_id']})",
    ]
    macro_vals = [
        best_b0["validation_tuned_metrics"]["macro_f1"],
        best_b1["validation_tuned_metrics"]["macro_f1"],
        best_m1["validation_tuned_metrics"]["macro_f1"],
        final_winner_info["final_validation_metrics"]["macro_f1"],
    ]
    micro_vals = [
        best_b0["validation_tuned_metrics"]["micro_f1"],
        best_b1["validation_tuned_metrics"]["micro_f1"],
        best_m1["validation_tuned_metrics"]["micro_f1"],
        final_winner_info["final_validation_metrics"]["micro_f1"],
    ]
    req_f2_vals = [
        best_b0["validation_tuned_metrics"]["request_f2"],
        best_b1["validation_tuned_metrics"]["request_f2"],
        best_m1["validation_tuned_metrics"]["request_f2"],
        final_winner_info["final_validation_metrics"]["request_f2"],
    ]

    x = np.arange(len(families))
    w = 0.25
    b1_bars = ax.bar(x - w, macro_vals, width=w, label="Macro F1 (5 Labels)", color=NAVY)
    b2_bars = ax.bar(x, micro_vals, width=w, label="Micro F1 (5 Labels)", color=TEAL)
    b3_bars = ax.bar(x + w, req_f2_vals, width=w, label="Request F2 (Task A)", color=ORANGE)

    ax.set_title("Validation Performance Comparison Across Model Families (Tuned Thresholds)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9.5)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True)

    for bar_group in (b1_bars, b2_bars, b3_bars):
        for bar in bar_group:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
    fig.tight_layout()
    f1_path = PHASE2_FIGURES_DIR / "fig1_validation_model_comparison.png"
    fig.savefig(f1_path)
    plt.close(fig)
    saved_files.append(str(f1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Figure 2: Per-label F1 comparison across B0 (stratified), B1 (best), and Selected Winner on Validation
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=150)
    x_lbl = np.arange(len(TARGET_LABELS))
    w = 0.25
    b0_f1s = [best_b0["validation_tuned_metrics"]["per_label"][l]["f1"] for l in TARGET_LABELS]
    b1_f1s = [best_b1["validation_tuned_metrics"]["per_label"][l]["f1"] for l in TARGET_LABELS]
    win_f1s = [final_winner_info["final_validation_metrics"]["per_label"][l]["f1"] for l in TARGET_LABELS]

    bars_b0 = ax.bar(x_lbl - w, b0_f1s, width=w, label=f"B0 ({best_b0['model_id']})", color=SLATE)
    bars_b1 = ax.bar(x_lbl, b1_f1s, width=w, label=f"B1 ({best_b1['model_id']})", color=TEAL)
    bars_win = ax.bar(x_lbl + w, win_f1s, width=w, label=f"Winner ({final_winner_info['selected_model_id']})", color=NAVY)

    ax.set_title("Validation Per-Label F1 Score Comparison (B0 vs. B1 vs. Selected M1 Winner)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticks(x_lbl)
    ax.set_xticklabels(TARGET_LABELS, fontsize=10.5)
    ax.set_ylabel("Validation F1 Score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    for grp in (bars_b0, bars_b1, bars_win):
        for bar in grp:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    f2_path = PHASE2_FIGURES_DIR / "fig2_per_label_f1_comparison.png"
    fig.savefig(f2_path)
    plt.close(fig)
    saved_files.append(str(f2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Figure 3: Request Precision-Recall Curve on Validation (comparing B1 and Selected Winner)
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    y_req_true = y_val_df["request"].to_numpy(dtype=np.int64)
    win_req_prob = final_winner_info["val_prob_matrix"][:, 0]
    b1_req_prob = best_b1["val_prob_matrix"][:, 0]

    prec_w, rec_w, _ = precision_recall_curve(y_req_true, win_req_prob)
    prec_b1, rec_b1, _ = precision_recall_curve(y_req_true, b1_req_prob)
    ap_w = final_winner_info["final_validation_metrics"]["per_label"]["request"]["average_precision"]
    ap_b1 = best_b1["validation_tuned_metrics"]["per_label"]["request"]["average_precision"]

    ax.plot(rec_w, prec_w, color=NAVY, lw=2.2, label=f"Selected Winner (AP = {ap_w:.3f})")
    ax.plot(rec_b1, prec_b1, color=TEAL, lw=1.8, linestyle="--", label=f"Best B1 Naive Bayes (AP = {ap_b1:.3f})")
    base_rate = float(y_req_true.mean())
    ax.axhline(base_rate, color=SLATE, linestyle=":", label=f"Prevalence Baseline ({base_rate:.3f})")

    # Mark selected operating point
    win_req_metrics = final_winner_info["final_validation_metrics"]["per_label"]["request"]
    ax.scatter(
        [win_req_metrics["recall"]],
        [win_req_metrics["precision"]],
        color=ORANGE,
        s=90,
        zorder=5,
        label=f"Selected Threshold (τ={win_req_metrics['threshold']:.2f}, F2={win_req_metrics['f2']:.3f})",
    )
    ax.set_title("Validation Precision-Recall Curve for `request` (Aid-Request Intent)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", frameon=True, fontsize=9.5)
    fig.tight_layout()
    f3_path = PHASE2_FIGURES_DIR / "fig3_request_precision_recall_curve.png"
    fig.savefig(f3_path)
    plt.close(fig)
    saved_files.append(str(f3_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Figure 4: Five Confusion Matrices for the Selected Model on Validation
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), dpi=150)
    for idx, lbl in enumerate(TARGET_LABELS):
        ax_cm = axes[idx]
        cm_dict = final_winner_info["final_validation_metrics"]["per_label"][lbl]["confusion_matrix"]
        mat = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
        ax_cm.imshow(mat, cmap="Blues")
        ax_cm.set_title(f"{lbl}\n(τ={final_winner_info['selected_thresholds'][lbl]:.2f})", fontsize=10.5, fontweight="bold")
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
        ax_cm.set_yticklabels(["True 0", "True 1"], fontsize=9)
        labels_txt = [["TN", "FP"], ["FN", "TP"]]
        for r in range(2):
            for c in range(2):
                val_cnt = mat[r, c]
                color_txt = "white" if val_cnt > (mat.max() * 0.55) else "#0F203C"
                ax_cm.text(
                    c, r, f"{labels_txt[r][c]}\n{val_cnt:,}",
                    ha="center", va="center", color=color_txt, fontsize=9.5, fontweight="bold"
                )
    fig.suptitle("Validation Confusion Matrices Across All 5 Labels (Selected Model)", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    f4_path = PHASE2_FIGURES_DIR / "fig4_validation_confusion_matrices.png"
    fig.savefig(f4_path, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(f4_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Figure 5: Threshold vs. Request F2, F1, Precision, and Recall on Validation
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    req_curve = final_winner_info["threshold_tuning_details"]["request"]["grid_curve"]
    taus = [pt["threshold"] for pt in req_curve]
    f2s = [pt["f2"] for pt in req_curve]
    f1s = [pt["f1"] for pt in req_curve]
    precs = [pt["precision"] for pt in req_curve]
    recs = [pt["recall"] for pt in req_curve]
    best_req_tau = final_winner_info["selected_thresholds"]["request"]
    best_req_f2 = final_winner_info["final_validation_metrics"]["per_label"]["request"]["f2"]

    ax.plot(taus, f2s, color=ORANGE, lw=2.5, label="Request F2 Score (Objective)")
    ax.plot(taus, f1s, color=NAVY, lw=1.8, linestyle="--", label="Request F1 Score")
    ax.plot(taus, precs, color=TEAL, lw=1.5, linestyle=":", label="Precision")
    ax.plot(taus, recs, color=SLATE, lw=1.5, linestyle="-.", label="Recall")
    ax.axvline(best_req_tau, color=ORANGE, linestyle="--", alpha=0.8, label=f"Selected τ = {best_req_tau:.2f} (F2 = {best_req_f2:.3f})")
    ax.axvline(0.50, color="gray", linestyle=":", alpha=0.7, label="Default τ = 0.50")

    ax.set_title("Validation Threshold Optimization Curve for `request`", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold (τ)", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    fig.tight_layout()
    f5_path = PHASE2_FIGURES_DIR / "fig5_threshold_vs_request_f2.png"
    fig.savefig(f5_path)
    plt.close(fig)
    saved_files.append(str(f5_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Figure 6: Label Prevalence (%) vs. Achieved Validation Recall (%) (Default vs. Tuned)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    prevalences = [
        (final_winner_info["final_validation_metrics"]["per_label"][l]["positive_support"]
         / final_winner_info["final_validation_metrics"]["n_samples"]) * 100.0
        for l in TARGET_LABELS
    ]
    def_recalls = [
        final_winner_info["default_validation_metrics"]["per_label"][l]["recall"] * 100.0
        for l in TARGET_LABELS
    ]
    tuned_recalls = [
        final_winner_info["final_validation_metrics"]["per_label"][l]["recall"] * 100.0
        for l in TARGET_LABELS
    ]

    x_pos = np.arange(len(TARGET_LABELS))
    w = 0.26
    b_prev = ax.bar(x_pos - w, prevalences, width=w, color=SLATE, label="Positive Prevalence (%)")
    b_drec = ax.bar(x_pos, def_recalls, width=w, color=TEAL, label="Achieved Recall @ Default τ=0.50 (%)")
    b_trec = ax.bar(x_pos + w, tuned_recalls, width=w, color=ORANGE, label="Achieved Recall @ Tuned τ* (%)")

    ax.set_title("Validation Label Prevalence vs. Achieved Recall (Default vs. Tuned Thresholds)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(TARGET_LABELS, fontsize=10.5)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    for grp in (b_prev, b_drec, b_trec):
        for bar in grp:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    f6_path = PHASE2_FIGURES_DIR / "fig6_prevalence_vs_achieved_recall.png"
    fig.savefig(f6_path)
    plt.close(fig)
    saved_files.append(str(f6_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Optional Figure 7: Reliability Diagram (Uncalibrated vs. Sigmoid-Calibrated on `request` and `food`)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=150)
    rel_data = final_winner_info["calibration_analysis"]["reliability_curves"]
    for ax_r, lbl in zip(axes, ["request", "food"]):
        u_curve = rel_data[lbl]["uncalibrated"]
        c_curve = rel_data[lbl]["sigmoid_calibrated"]
        ax_r.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        ax_r.plot(
            u_curve["mean_predicted_value"],
            u_curve["fraction_of_positives"],
            "s-",
            color=ORANGE,
            label=f"Uncalibrated (Brier={final_winner_info['calibration_analysis']['uncalibrated_brier_per_label'][lbl]:.4f})",
        )
        ax_r.plot(
            c_curve["mean_predicted_value"],
            c_curve["fraction_of_positives"],
            "o-",
            color=NAVY,
            label=f"Sigmoid CV-Calibrated (Brier={final_winner_info['calibration_analysis']['sigmoid_calibrated_brier_per_label'][lbl]:.4f})",
        )
        ax_r.set_title(f"Reliability Diagram — `{lbl}` (Validation)", fontsize=11.5, fontweight="bold")
        ax_r.set_xlabel("Mean Predicted Score", fontsize=10.5)
        ax_r.set_ylabel("Empirical Fraction of Positives", fontsize=10.5)
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(0, 1)
        ax_r.grid(True, linestyle="--", alpha=0.4)
        ax_r.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    f7_path = PHASE2_FIGURES_DIR / "fig7_reliability_diagram.png"
    fig.savefig(f7_path)
    plt.close(fig)
    saved_files.append(str(f7_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    # Optional Figure 8: Genre Subgroup Performance Comparison (`direct` vs. `news`) on Validation
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    sub_g = final_winner_info["error_analysis"]["genre_subgroups"]
    g_names = ["direct", "news"]
    m_keys = ["request_precision", "request_recall", "request_f1", "request_f2", "macro_f1"]
    m_labels = ["Req Precision", "Req Recall", "Req F1", "Req F2", "Macro F1 (5 Labels)"]
    x_m = np.arange(len(m_keys))
    w = 0.34
    d_vals = [sub_g["direct"][k] for k in m_keys]
    n_vals = [sub_g["news"][k] for k in m_keys]

    b_dir = ax.bar(x_m - w / 2, d_vals, width=w, color=NAVY, label=f"direct (n={sub_g['direct']['record_count']:,})")
    b_nws = ax.bar(x_m + w / 2, n_vals, width=w, color=TEAL, label=f"news (n={sub_g['news']['record_count']:,})")
    ax.set_title("Validation Performance by Message Genre (`direct` vs. `news`)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x_m)
    ax.set_xticklabels(m_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    for grp in (b_dir, b_nws):
        for bar in grp:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    fig.tight_layout()
    f8_path = PHASE2_FIGURES_DIR / "fig8_genre_subgroup_comparison.png"
    fig.savefig(f8_path)
    plt.close(fig)
    saved_files.append(str(f8_path.relative_to(PROJECT_ROOT)).replace("\\", "/"))

    return saved_files


def run_training_and_validation() -> dict[str, Any]:
    """
    Execute Steps A through G and Step I (Model Serialization) of SINYAL Phase 2.
    """
    overall_t0 = time.perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step A: Pre-flight verification
    preflight = run_preflight_verification()
    train_df: pd.DataFrame = preflight["train_df"]
    val_df: pd.DataFrame = preflight["val_df"]

    X_train = train_df["message"].astype(str)
    y_train = train_df[TARGET_LABELS]
    X_val = val_df["message"].astype(str)
    y_val = val_df[TARGET_LABELS]

    # Step B, C, D, E: Build, train, and evaluate all B0, B1, and M1 candidate configurations on validation
    candidates = build_candidate_models()
    evaluated_models: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for cand in candidates:
        m_id = cand["model_id"]
        pipe: Pipeline = cand["pipeline"]
        print(f"[TRAIN] Fitting {m_id} ({cand['description']}) on train.csv ({len(X_train):,} rows)...")
        t_start = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_sec = round(time.perf_counter() - t_start, 3)

        val_prob = extract_positive_probabilities(pipe, X_val)
        default_metrics = compute_multilabel_metrics(y_val, val_prob, DEFAULT_THRESHOLDS)
        tuned_thresholds, tuning_details = optimize_thresholds_on_validation(y_val, val_prob)
        tuned_metrics = compute_multilabel_metrics(y_val, val_prob, tuned_thresholds)

        entry = {
            **cand,
            "training_duration_sec": train_sec,
            "val_prob_matrix": val_prob,
            "validation_default_metrics": default_metrics,
            "selected_thresholds": tuned_thresholds,
            "threshold_tuning_details": tuning_details,
            "validation_tuned_metrics": tuned_metrics,
        }
        evaluated_models.append(entry)

        # Record both Default (0.50) and Validation-Tuned rows for transparent side-by-side comparison
        for regime_name, m_obj in [("default_0.50", default_metrics), ("validation_tuned", tuned_metrics)]:
            row_dict: dict[str, Any] = {
                "model_id": m_id,
                "family": cand["family"],
                "threshold_regime": regime_name,
                "training_duration_sec": train_sec,
                "macro_f1": m_obj["macro_f1"],
                "micro_f1": m_obj["micro_f1"],
                "request_f2": m_obj["request_f2"],
                "request_f1": m_obj["per_label"]["request"]["f1"],
                "water_f1": m_obj["per_label"]["water"]["f1"],
                "food_f1": m_obj["per_label"]["food"]["f1"],
                "shelter_f1": m_obj["per_label"]["shelter"]["f1"],
                "medical_help_f1": m_obj["per_label"]["medical_help"]["f1"],
                "mean_average_precision": m_obj["mean_average_precision"],
                "mean_brier_score": m_obj["mean_brier_score"],
                "hamming_loss": m_obj["hamming_loss"],
                "request_threshold": m_obj["thresholds"]["request"],
                "water_threshold": m_obj["thresholds"]["water"],
                "food_threshold": m_obj["thresholds"]["food"],
                "shelter_threshold": m_obj["thresholds"]["shelter"],
                "medical_help_threshold": m_obj["thresholds"]["medical_help"],
            }
            comparison_rows.append(row_dict)

        print(
            f"  -> {m_id} ({train_sec:.2f}s) | Default Macro F1={default_metrics['macro_f1']:.4f}, "
            f"Req F2={default_metrics['request_f2']:.4f} | Tuned Macro F1={tuned_metrics['macro_f1']:.4f}, "
            f"Req F2={tuned_metrics['request_f2']:.4f}"
        )

    # Step F: Apply written selection policy to pick base winning family/hyperparameters
    base_winner = select_winning_model(evaluated_models)
    print(f"[WINNER BASE] Selected {base_winner['model_id']} via pre-registered validation selection policy.")

    # Step G: Assess calibration and evaluate leakage-safe 3-fold CV sigmoid calibration on train.csv
    cal_eval = evaluate_calibration_and_optional_sigmoid(
        base_winner, X_train, y_train, X_val, y_val
    )
    print(f"[CALIBRATION] {cal_eval['calibration_decision_explanation']}")

    # Also add the sigmoid-calibrated candidate to the validation comparison table for full transparency
    cal_model_id = f"{base_winner['model_id']}_sigmoid_cv3"
    for regime_name, m_obj in [
        ("default_0.50", cal_eval["sigmoid_default_metrics"]),
        ("validation_tuned", cal_eval["sigmoid_tuned_metrics"]),
    ]:
        comparison_rows.append(
            {
                "model_id": cal_model_id,
                "family": f"{base_winner['family']}_Calibrated",
                "threshold_regime": regime_name,
                "training_duration_sec": cal_eval["sigmoid_training_duration_sec"],
                "macro_f1": m_obj["macro_f1"],
                "micro_f1": m_obj["micro_f1"],
                "request_f2": m_obj["request_f2"],
                "request_f1": m_obj["per_label"]["request"]["f1"],
                "water_f1": m_obj["per_label"]["water"]["f1"],
                "food_f1": m_obj["per_label"]["food"]["f1"],
                "shelter_f1": m_obj["per_label"]["shelter"]["f1"],
                "medical_help_f1": m_obj["per_label"]["medical_help"]["f1"],
                "mean_average_precision": m_obj["mean_average_precision"],
                "mean_brier_score": m_obj["mean_brier_score"],
                "hamming_loss": m_obj["hamming_loss"],
                "request_threshold": m_obj["thresholds"]["request"],
                "water_threshold": m_obj["thresholds"]["water"],
                "food_threshold": m_obj["thresholds"]["food"],
                "shelter_threshold": m_obj["thresholds"]["shelter"],
                "medical_help_threshold": m_obj["thresholds"]["medical_help"],
            }
        )

    if cal_eval["use_calibrated_pipeline"]:
        final_pipeline = cal_eval["calibrated_pipeline_object"]
        final_val_prob = cal_eval["calibrated_val_prob_matrix"]
        final_thresholds = cal_eval["sigmoid_selected_thresholds"]
        final_tuning_details = cal_eval["sigmoid_tuning_details"]
        final_default_val_metrics = cal_eval["sigmoid_default_metrics"]
        final_tuned_val_metrics = cal_eval["sigmoid_tuned_metrics"]
        selected_model_id = cal_model_id
        calibration_status = "sigmoid_cv3_calibrated_on_train"
        final_train_sec = cal_eval["sigmoid_training_duration_sec"]
    else:
        final_pipeline = base_winner["pipeline"]
        final_val_prob = base_winner["val_prob_matrix"]
        final_thresholds = base_winner["selected_thresholds"]
        final_tuning_details = base_winner["threshold_tuning_details"]
        final_default_val_metrics = base_winner["validation_default_metrics"]
        final_tuned_val_metrics = base_winner["validation_tuned_metrics"]
        selected_model_id = base_winner["model_id"]
        calibration_status = "uncalibrated_raw_probabilities"
        final_train_sec = base_winner["training_duration_sec"]

    # Build human-review policy config & evaluate on validation
    policy_config = build_decision_policy_config(
        thresholds=final_thresholds,
        uncertainty_half_width=DEFAULT_UNCERTAINTY_HALF_WIDTH,
        implicit_request_ratio=0.50,
    )
    val_review_simulation = evaluate_review_policy_on_split(y_val, final_val_prob, policy_config)

    # Exploratory triage ranking utility on validation (Recall@10%, 20%, 30%)
    val_ranking_utility = compute_ranking_utility(
        y_val["request"], final_val_prob[:, 0], random_seed=RANDOM_SEED
    )

    # Validation error analysis & subgroup robustness
    val_error_analysis = run_validation_error_analysis(val_df, final_val_prob, final_thresholds)

    # Extract lexical interpretability features
    interpretability = extract_interpretability_features(final_pipeline, top_k=15)

    # Save comparison CSV
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv_path = PHASE2_REPORTS_DIR / "model_comparison_validation.csv"
    comp_df.to_csv(comp_csv_path, index=False, encoding="utf-8")
    print(f"[SAVED] {comp_csv_path}")

    final_winner_info = {
        "selected_model_id": selected_model_id,
        "base_winner_model_id": base_winner["model_id"],
        "family": base_winner["family"],
        "description": base_winner["description"],
        "calibration_status": calibration_status,
        "training_duration_sec": final_train_sec,
        "hyperparameters": base_winner["hyperparameters"],
        "selected_thresholds": final_thresholds,
        "threshold_tuning_details": final_tuning_details,
        "default_validation_metrics": final_default_val_metrics,
        "final_validation_metrics": final_tuned_val_metrics,
        "val_prob_matrix": final_val_prob,
        "calibration_analysis": {
            k: v
            for k, v in cal_eval.items()
            if k not in ("calibrated_pipeline_object", "calibrated_val_prob_matrix")
        },
        "error_analysis": val_error_analysis,
        "review_policy_simulation": val_review_simulation,
        "ranking_utility": val_ranking_utility,
        "interpretability": interpretability,
    }

    # Generate validation figures
    saved_val_figures = generate_validation_figures(
        evaluated_models, final_winner_info, y_val, val_df
    )

    # Step I: Serialize selected model, model_manifest.json, and decision_policy.json BEFORE test evaluation
    model_path = MODELS_DIR / "sinyal_model.joblib"
    manifest_path = MODELS_DIR / "model_manifest.json"
    policy_path = MODELS_DIR / "decision_policy.json"

    joblib.dump(final_pipeline, model_path)
    model_sha256 = compute_sha256(model_path)

    # Reload immediately and verify exact inference parity on validation set
    reloaded_pipeline = joblib.load(model_path)
    reloaded_val_prob = extract_positive_probabilities(reloaded_pipeline, X_val)
    max_abs_diff = float(np.max(np.abs(final_val_prob - reloaded_val_prob)))
    if max_abs_diff > 1e-12:
        raise ValueError(f"Serialized model reload parity check failed: max_abs_diff={max_abs_diff}")
    print(f"[SERIALIZED] {model_path} (SHA-256: {model_sha256}, reload max_abs_diff={max_abs_diff:.2e})")

    policy_path.write_text(json.dumps(policy_config, indent=2), encoding="utf-8")
    policy_sha256 = compute_sha256(policy_path)

    manifest = {
        "model_version": "2.0.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_BEFORE_HELD_OUT_TEST",
        "selected_model_id": selected_model_id,
        "base_model_id": base_winner["model_id"],
        "model_family": base_winner["family"],
        "description": base_winner["description"],
        "calibration_status": calibration_status,
        "random_seed": RANDOM_SEED,
        "target_labels": TARGET_LABELS,
        "tfidf_parameters": {
            **TFIDF_PARAMS,
            "ngram_range": list(TFIDF_PARAMS["ngram_range"]),
        },
        "classifier_hyperparameters": base_winner["hyperparameters"],
        "decision_thresholds": final_thresholds,
        "default_thresholds": DEFAULT_THRESHOLDS,
        "uncertainty_half_width": DEFAULT_UNCERTAINTY_HALF_WIDTH,
        "dataset_checksums": preflight["verification"],
        "artifacts": {
            "model_joblib_path": "models/sinyal_model.joblib",
            "model_joblib_sha256": model_sha256,
            "decision_policy_path": "models/decision_policy.json",
            "decision_policy_sha256": policy_sha256,
        },
        "validation_summary_at_freeze": {
            "default_macro_f1": final_default_val_metrics["macro_f1"],
            "default_micro_f1": final_default_val_metrics["micro_f1"],
            "default_request_f2": final_default_val_metrics["request_f2"],
            "tuned_macro_f1": final_tuned_val_metrics["macro_f1"],
            "tuned_micro_f1": final_tuned_val_metrics["micro_f1"],
            "tuned_request_f2": final_tuned_val_metrics["request_f2"],
            "mean_average_precision": final_tuned_val_metrics["mean_average_precision"],
            "mean_brier_score": final_tuned_val_metrics["mean_brier_score"],
            "review_flag_rate": val_review_simulation["flagged_for_review_rate"],
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
                "matplotlib": matplotlib.__version__,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SAVED] {manifest_path} & {policy_path}")

    # Save complete validation metrics & analysis JSON
    # Strip large curve arrays from individual non-winning models to keep JSON clean and readable
    clean_models_summary = []
    for m in evaluated_models:
        clean_models_summary.append(
            {
                "model_id": m["model_id"],
                "family": m["family"],
                "description": m["description"],
                "training_duration_sec": m["training_duration_sec"],
                "hyperparameters": {
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in m["hyperparameters"].items()
                },
                "selected_thresholds": m["selected_thresholds"],
                "validation_default_metrics": m["validation_default_metrics"],
                "validation_tuned_metrics": m["validation_tuned_metrics"],
            }
        )

    validation_report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_sec": round(time.perf_counter() - overall_t0, 3),
        "preflight_verification": preflight["verification"],
        "evaluated_models": clean_models_summary,
        "selected_winner": {
            "selected_model_id": selected_model_id,
            "base_winner_model_id": base_winner["model_id"],
            "calibration_status": calibration_status,
            "selected_thresholds": final_thresholds,
            "default_validation_metrics": final_default_val_metrics,
            "tuned_validation_metrics": final_tuned_val_metrics,
            "threshold_tuning_summary": {
                lbl: {
                    k: v
                    for k, v in final_tuning_details[lbl].items()
                    if k != "grid_curve"
                }
                for lbl in TARGET_LABELS
            },
        },
        "calibration_analysis": final_winner_info["calibration_analysis"],
        "human_review_policy_validation": val_review_simulation,
        "exploratory_ranking_utility_validation": val_ranking_utility,
        "validation_error_analysis": val_error_analysis,
        "interpretability_top_features": interpretability,
        "generated_figures": saved_val_figures,
    }

    val_json_path = PHASE2_REPORTS_DIR / "validation_metrics.json"
    val_json_path.write_text(json.dumps(validation_report_payload, indent=2), encoding="utf-8")
    print(f"[SAVED] {val_json_path}")

    return validation_report_payload


if __name__ == "__main__":
    run_training_and_validation()


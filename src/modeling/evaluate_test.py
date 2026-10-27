"""
SINYAL Phase 2 — Single-Shot Held-Out Test Set Evaluation (`src/modeling/evaluate_test.py`).

STRICT DATA USAGE CONTRACT (Sections 6 & 18):
- Requires explicit CLI flag `--confirm-frozen` confirming that model family, hyperparameters,
  vectorizer, decision thresholds, calibration choice, and human-review policy are frozen.
- Checks `reports/phase2/final_evaluation_lock.json`: if a lock already exists for the current
  frozen manifest, refuses to re-run unless explicitly invoked by automated lock-verification tests.
- Evaluates the serialized frozen model (`models/sinyal_model.joblib`) on the cleaned held-out
  test set (`data/processed/test.csv`, 2,387 rows) exactly once.
- Generates:
  * `reports/phase2/final_test_metrics.json`
  * `reports/phase2/per_label_test_metrics.csv`
  * `reports/phase2/final_evaluation_lock.json`
  * Held-out test confusion matrices figure in `reports/phase2/figures/`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import TARGET_LABELS
from src.modeling.metrics import (
    DEFAULT_THRESHOLDS,
    compute_genre_subgroup_metrics,
    compute_multilabel_metrics,
    compute_ranking_utility,
    extract_positive_probabilities,
)
from src.modeling.review_policy import evaluate_review_policy_on_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PHASE2_REPORTS_DIR = PROJECT_ROOT / "reports" / "phase2"
PHASE2_FIGURES_DIR = PHASE2_REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "sinyal_model.joblib"
MANIFEST_PATH = MODELS_DIR / "model_manifest.json"
POLICY_PATH = MODELS_DIR / "decision_policy.json"
LOCK_PATH = PHASE2_REPORTS_DIR / "final_evaluation_lock.json"
FINAL_METRICS_PATH = PHASE2_REPORTS_DIR / "final_test_metrics.json"
PER_LABEL_CSV_PATH = PHASE2_REPORTS_DIR / "per_label_test_metrics.csv"

EXPECTED_TEST_SHA256 = "d0f7265d67753a27a40a0fcc6caab8f88849b352427a71929ac692b37b06d75a"
EXPECTED_TEST_ROWS = 2387


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_test_confusion_matrix_figure(
    test_metrics_tuned: dict[str, Any],
    thresholds: dict[str, float],
) -> str:
    """Save 5-panel held-out test confusion matrix chart."""
    PHASE2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), dpi=150)
    for idx, lbl in enumerate(TARGET_LABELS):
        ax_cm = axes[idx]
        cm_dict = test_metrics_tuned["per_label"][lbl]["confusion_matrix"]
        mat = np.array([[cm_dict["tn"], cm_dict["fp"]], [cm_dict["fn"], cm_dict["tp"]]])
        ax_cm.imshow(mat, cmap="Teals" if "Teals" in plt.colormaps() else "YlGnBu")
        ax_cm.set_title(f"{lbl}\n(τ={thresholds[lbl]:.2f})", fontsize=10.5, fontweight="bold")
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
    fig.suptitle(
        "Cleaned Held-Out Test Set Confusion Matrices Across All 5 Labels (Frozen Winner)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    out_path = PHASE2_FIGURES_DIR / "fig9_test_confusion_matrices.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def run_single_shot_test_evaluation(
    confirm_frozen: bool = False,
    lock_path: Path = LOCK_PATH,
) -> dict[str, Any]:
    """
    Execute the frozen winning pipeline on `data/processed/test.csv` once and write lock + metrics.
    """
    if not confirm_frozen:
        raise PermissionError(
            "Held-out test evaluation requires explicit confirmation that the experiment "
            "configuration is frozen (`confirm_frozen=True` or CLI `--confirm-frozen`)."
        )

    if lock_path.exists():
        existing_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Held-out test evaluation is LOCKED (`{lock_path}`). "
            f"Single-shot evaluation was already executed at {existing_lock.get('evaluated_at_utc')} "
            f"for model `{existing_lock.get('selected_model_id')}` "
            f"(model SHA-256: `{existing_lock.get('model_joblib_sha256')}`). "
            "Repeated test evaluations are forbidden by the Phase 2 Data Usage Contract."
        )

    if not MANIFEST_PATH.exists() or not MODEL_PATH.exists() or not POLICY_PATH.exists():
        raise FileNotFoundError(
            "Missing frozen model artifacts (`sinyal_model.joblib`, `model_manifest.json`, or `decision_policy.json`). "
            "Run `python -m src.modeling.train_validate` first."
        )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    if manifest.get("status") != "FROZEN_BEFORE_HELD_OUT_TEST":
        raise ValueError(f"Manifest status is not FROZEN_BEFORE_HELD_OUT_TEST: {manifest.get('status')}")

    actual_model_sha = compute_sha256(MODEL_PATH)
    expected_model_sha = manifest["artifacts"]["model_joblib_sha256"]
    if actual_model_sha != expected_model_sha:
        raise ValueError(
            f"Model artifact SHA-256 ({actual_model_sha}) does not match frozen manifest ({expected_model_sha})!"
        )

    if not PROCESSED_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned held-out test set: {PROCESSED_TEST_PATH}")
    actual_test_sha = compute_sha256(PROCESSED_TEST_PATH)
    if actual_test_sha != EXPECTED_TEST_SHA256:
        raise ValueError(
            f"Cleaned held-out test set SHA-256 mismatch: expected {EXPECTED_TEST_SHA256}, got {actual_test_sha}"
        )

    test_df = pd.read_csv(PROCESSED_TEST_PATH, encoding="utf-8", low_memory=False)
    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Cleaned held-out test set row count mismatch: expected {EXPECTED_TEST_ROWS}, got {len(test_df)}")

    model = joblib.load(MODEL_PATH)
    X_test = test_df["message"].astype(str)
    y_test = test_df[TARGET_LABELS]

    print(
        f"[HELD-OUT TEST] Evaluating frozen model `{manifest['selected_model_id']}` "
        f"on cleaned held-out test set ({len(test_df):,} rows)..."
    )
    test_prob = extract_positive_probabilities(model, X_test)
    frozen_thresholds = {lbl: float(manifest["decision_thresholds"][lbl]) for lbl in TARGET_LABELS}

    # Evaluate at frozen validation-tuned thresholds (primary) and default 0.50 thresholds (reference)
    test_metrics_tuned = compute_multilabel_metrics(y_test, test_prob, frozen_thresholds)
    test_metrics_default = compute_multilabel_metrics(y_test, test_prob, DEFAULT_THRESHOLDS)

    # Human-review policy evaluation on held-out test set
    test_review_policy = evaluate_review_policy_on_split(y_test, test_prob, policy)

    # Exploratory subgroup analysis by genre (`direct` vs `news`) on held-out test set
    test_genre_subgroups = compute_genre_subgroup_metrics(test_df, test_prob, frozen_thresholds)

    # Exploratory triage ranking utility (Recall@10%, 20%, 30%) on held-out test set
    test_ranking_utility = compute_ranking_utility(
        y_test["request"], test_prob[:, 0], random_seed=int(manifest.get("random_seed", 42))
    )

    # Generate held-out test confusion matrices figure
    cm_fig_relpath = generate_test_confusion_matrix_figure(test_metrics_tuned, frozen_thresholds)

    # Save per-label test metrics CSV
    per_label_rows = []
    for lbl in TARGET_LABELS:
        pl_t = test_metrics_tuned["per_label"][lbl]
        pl_d = test_metrics_default["per_label"][lbl]
        per_label_rows.append(
            {
                "label": lbl,
                "positive_support": pl_t["positive_support"],
                "negative_support": pl_t["negative_support"],
                "frozen_threshold": pl_t["threshold"],
                "precision": pl_t["precision"],
                "recall": pl_t["recall"],
                "f1": pl_t["f1"],
                "f2": pl_t["f2"],
                "average_precision": pl_t["average_precision"],
                "brier_score": pl_t["brier_score"],
                "balanced_accuracy": pl_t["balanced_accuracy"],
                "accuracy_secondary": pl_t["accuracy_secondary"],
                "predicted_positive_count": pl_t["predicted_positive_count"],
                "predicted_positive_rate": pl_t["predicted_positive_rate"],
                "tn": pl_t["confusion_matrix"]["tn"],
                "fp": pl_t["confusion_matrix"]["fp"],
                "fn": pl_t["confusion_matrix"]["fn"],
                "tp": pl_t["confusion_matrix"]["tp"],
                "default_0.50_precision": pl_d["precision"],
                "default_0.50_recall": pl_d["recall"],
                "default_0.50_f1": pl_d["f1"],
                "default_0.50_f2": pl_d["f2"],
            }
        )
    PHASE2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_label_rows).to_csv(PER_LABEL_CSV_PATH, index=False, encoding="utf-8")

    evaluated_ts = datetime.now(timezone.utc).isoformat()
    final_payload = {
        "evaluation_timestamp_utc": evaluated_ts,
        "dataset_description": (
            "Cleaned held-out test set (`data/processed/test.csv`, 2,387 rows; 10 duplicate IDs "
            "removed from raw `disaster_response_test.csv` during Phase 1 deterministic preparation)."
        ),
        "offline_benchmark_disclaimer": (
            "Offline benchmark evaluation on historical disaster-response messages only; "
            "does not constitute field validation or real-world deployment evidence."
        ),
        "selected_model_id": manifest["selected_model_id"],
        "base_model_id": manifest["base_model_id"],
        "calibration_status": manifest["calibration_status"],
        "test_dataset_sha256": actual_test_sha,
        "test_dataset_rows": len(test_df),
        "model_joblib_sha256": actual_model_sha,
        "frozen_thresholds": frozen_thresholds,
        "held_out_test_metrics_frozen_thresholds": test_metrics_tuned,
        "held_out_test_metrics_default_0.50_reference": test_metrics_default,
        "human_review_policy_test": test_review_policy,
        "exploratory_genre_subgroups_test": test_genre_subgroups,
        "exploratory_ranking_utility_test": test_ranking_utility,
        "confusion_matrix_figure": cm_fig_relpath,
    }
    FINAL_METRICS_PATH.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    final_metrics_sha = compute_sha256(FINAL_METRICS_PATH)

    # Write final evaluation lock
    lock_payload = {
        "locked": True,
        "evaluated_at_utc": evaluated_ts,
        "selected_model_id": manifest["selected_model_id"],
        "model_joblib_sha256": actual_model_sha,
        "manifest_sha256": compute_sha256(MANIFEST_PATH),
        "policy_sha256": compute_sha256(POLICY_PATH),
        "test_dataset_path": "data/processed/test.csv",
        "test_dataset_rows": len(test_df),
        "test_dataset_sha256": actual_test_sha,
        "final_test_metrics_sha256": final_metrics_sha,
        "summary": {
            "macro_f1": test_metrics_tuned["macro_f1"],
            "micro_f1": test_metrics_tuned["micro_f1"],
            "request_f2": test_metrics_tuned["request_f2"],
            "request_f1": test_metrics_tuned["request_f1"],
            "mean_average_precision": test_metrics_tuned["mean_average_precision"],
            "review_flag_rate": test_review_policy["flagged_for_review_rate"],
        },
    }
    lock_path.write_text(json.dumps(lock_payload, indent=2), encoding="utf-8")

    print(
        f"[HELD-OUT TEST COMPLETE] Macro F1={test_metrics_tuned['macro_f1']:.4f} | "
        f"Micro F1={test_metrics_tuned['micro_f1']:.4f} | Request F2={test_metrics_tuned['request_f2']:.4f} | "
        f"Review Flag Rate={test_review_policy['flagged_for_review_rate']*100:.2f}%"
    )
    print(f"[LOCKED] {lock_path}")
    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute single-shot held-out test evaluation for SINYAL Phase 2."
    )
    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="Confirm that model selection, hyperparameters, thresholds, and review policy are frozen.",
    )
    args = parser.parse_args()
    run_single_shot_test_evaluation(confirm_frozen=args.confirm_frozen)


if __name__ == "__main__":
    main()


# Phase 2 Scientific Methodology & Implementation Reference — SINYAL

## 1. Alignment with Research Frameworks

### 1.1 CRISP-DM (Stages 4 & 5)
- **Stage 4 — Modeling:**
  - Implemented three distinct model families (`B0` Dummy baselines, `B1` TF-IDF + Multinomial Naive Bayes, `M1` TF-IDF + Logistic Regression) inside [`src/modeling/train_validate.py`](../src/modeling/train_validate.py).
  - Enforced strict training-only fitting of `TfidfVectorizer` (`lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.98, max_features=30000, sublinear_tf=True`) on `data/processed/train.csv` (`19,032` rows).
- **Stage 5 — Evaluation:**
  - Evaluated all candidate models under both default ($\tau = 0.50$) and validation-tuned ($\tau_k^* \in [0.05, 0.95]$) thresholds on `data/processed/validation.csv` (`2,329` rows).
  - Assessed probability calibration via Brier score and 3-fold training cross-validation sigmoid calibration (`CalibratedClassifierCV`).
  - Simulated the human-review decision policy and exploratory triage ranking utility (`Recall@10%`, `Recall@20%`, `Recall@30%`).
  - Executed a single locked evaluation on the **cleaned held-out test set** (`data/processed/test.csv`, `2,387` rows) via [`src/modeling/evaluate_test.py`](../src/modeling/evaluate_test.py).

### 1.2 Design Science Research Methodology (DSRM — Steps 3, 4, & 5)
- **Step 3 — Design and Development:** Created the modular `src/modeling/` subsystem (`metrics.py`, `review_policy.py`, `train_validate.py`, `evaluate_test.py`, `inference.py`) and serialized the frozen pipeline (`models/sinyal_model.joblib`, `models/model_manifest.json`, `models/decision_policy.json`).
- **Step 4 — Controlled Demonstration:** Demonstrated deterministic offline batch inference and human-review routing via `predict_messages(messages: list[str]) -> list[dict]`.
- **Step 5 — Technical Evaluation:** Benchmarked candidate linear models against trivial (`B0`) and probabilistic (`B1`) baselines across Macro F1, Micro F1, per-label Precision/Recall/F1, `request` $F_2$, and Average Precision.

---

## 2. Data Usage Contract & Single-Shot Test Lock

| Split | File | Rows | SHA-256 Checksum | Permitted Phase 2 Role |
|---|---|---:|---|---|
| **Train** | `data/processed/train.csv` | `19,032` | `13d0c0336f5ed8bc89c19ee3529075e5638f7aaf1976f1aaeefd0a3a75ecb02b` | Fitting TF-IDF vocabulary, IDF weights, classifiers, and 3-fold CV sigmoid calibrators. |
| **Validation** | `data/processed/validation.csv` | `2,329` | `e09d9842d17414e08946739d93853db1687a873a08a25210ea65bb5aec7b722e` | Hyperparameter comparison, winner selection, threshold optimization, calibration & error analysis. |
| **Cleaned Held-Out Test** | `data/processed/test.csv` | `2,387` | `d0f7265d67753a27a40a0fcc6caab8f88849b352427a71929ac692b37b06d75a` | Single-shot final evaluation after configuration freeze (`--confirm-frozen`), protected by `reports/phase2/final_evaluation_lock.json`. |

> **Important Provenance Note:** `data/processed/test.csv` (`2,387` rows) is the **cleaned held-out test set** produced in Phase 1 by deterministically deduplicating `10` duplicate `id` rows from the raw `disaster_response_test.csv` (`2,397` rows) while preserving all non-duplicate test instances untouched.

---

## 3. Safe Multilabel Probability Extraction

In `scikit-learn`, `MultiOutputClassifier.predict_proba(X)` returns a Python `list` of length $K=5$, where element $k$ is a 2D array of shape `(n_samples, 2)` corresponding to that sub-estimator's `classes_`.

[`extract_positive_probabilities()`](../src/modeling/metrics.py) enforces:
1. Verification that `len(raw_probas) == 5` and matches `TARGET_LABELS = ["request", "water", "food", "shelter", "medical_help"]`.
2. Explicit lookup of `pos_col_idx = list(est.classes_).index(1)` for every sub-estimator rather than assuming column index `1` without verification.
3. Validation that the resulting `(n_samples, 5)` float64 matrix contains strictly finite values in $[0.0, 1.0]$.

---

## 4. Metric Conventions & Threshold Optimization

- **Zero-Division Convention:** All precision, recall, $F_1$, and $F_2$ calculations explicitly set `zero_division=0` so majority-negative dummy baselines (`B0_most_frequent`) that predict zero positives receive a well-defined `0.0` score rather than raising warnings or NaNs.
- **Average Precision (AP):** Computed via `sklearn.metrics.average_precision_score(y_true, y_prob)`, which summarizes the precision-recall curve as the weighted mean of precisions achieved at each threshold (without trapezoidal interpolation).
- **Threshold Optimization Grid:** Evaluated over $\tau \in \{0.05, 0.06, \dots, 0.95\}$ ($\Delta\tau = 0.01$) on `validation.csv`:
  - `request`: maximizes $F_2$ ($\beta = 2.0$), weighting recall twice as heavily as precision.
  - `water`, `food`, `shelter`, `medical_help`: maximizes per-label $F_1$ ($\beta = 1.0$).

---

## 5. Interpretability & Ethical Guardrails

- **Lexical Associations, Not Causality:** Top positive and negative TF-IDF feature weights extracted from the linear classifiers reflect statistical co-occurrence in historical training data, not causal proof or factual verification of a humanitarian need.
- **Human-in-the-Loop Decision Support:** `src/modeling/review_policy.py` and `src/modeling/inference.py` route all predicted aid requests and borderline/uncertain predictions to human review and never automatically reject or delete low-scoring messages.


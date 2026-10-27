# Phase 2 Pre-Test Model Selection, Thresholding & Review Policy

> **Pre-Test Methodological Registration Notice:**
> This document is written and frozen **before** training any machine learning models and **before** accessing `data/processed/test.csv`. Where this policy specifies concrete numerical tie-breakers, threshold grid step sizes, calibration rules, and uncertainty review margins that were only outlined conceptually in `docs/experiment_protocol.md`, those specifications are explicitly documented here as **Phase 2 Pre-Test Amendments**.

---

## 1. Authoritative Target Ordering & Output Contract

Every model and evaluation routine must preserve the exact five binary labels in the following order:

```python
TARGET_LABELS = [
    "request",
    "water",
    "food",
    "shelter",
    "medical_help",
]
```

- **Shape Contract:** Predicted probability matrices and binary decision matrices must have shape `(n_samples, 5)`.
- **Semantic Separation:**
  - `request` represents **Task A (Aid-Request Intent)**.
  - `water`, `food`, `shelter`, and `medical_help` represent **Task B (Need-Topic Identification)**.
  - Positive topic predictions do **not** automatically imply a confirmed aid request.

---

## 2. Candidate Model Families & Hyperparameter Grid

All models are trained strictly on `data/processed/train.csv` (`19,032` rows) using deterministic seed `RANDOM_SEED = 42`.

### 2.1 Shared TF-IDF Vectorizer Configuration (for B1 and M1)
Fit exclusively on `train["message"]`:
- `lowercase = True`
- `ngram_range = (1, 2)`
- `min_df = 2`
- `max_df = 0.98`
- `max_features = 30000`
- `sublinear_tf = True`
- `strip_accents = None` (preserves multilingual tokens and numbers)

### 2.2 Model Family B0 — Dummy Baselines
1. **`B0_most_frequent`:** `MultiOutputClassifier(DummyClassifier(strategy="most_frequent", random_state=42))`
   - *Expected behavior:* Predicts `0` for all five minority-positive labels, demonstrating that high raw accuracy (~82%–93%) yields `0.0000` F1 and `0.0000` request $F_2$ and is operationally useless for humanitarian request detection.
2. **`B0_stratified`:** `MultiOutputClassifier(DummyClassifier(strategy="stratified", random_state=42))`
   - *Expected behavior:* Generates random binary predictions matching training class priors, establishing the stochastic performance floor.

### 2.3 Model Family B1 — Multinomial Naive Bayes Baselines
Architecture: `Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", MultiOutputClassifier(MultinomialNB(alpha=alpha)))])`
- Evaluated configurations on validation:
  1. **`B1_nb_alpha_0.5`:** `alpha = 0.5`
  2. **`B1_nb_alpha_1.0`:** `alpha = 1.0`
  3. **`B1_nb_alpha_2.0`:** `alpha = 2.0`

### 2.4 Model Family M1 — Logistic Regression Candidates
Architecture: `Pipeline([("tfidf", TfidfVectorizer(...)), ("clf", MultiOutputClassifier(LogisticRegression(C=C, class_weight=cw, solver="lbfgs", max_iter=1000, random_state=42)))])`
- Evaluated configurations on validation (6 combinations):
  1. **`M1_logreg_C0.5_cwNone`:** `C = 0.5, class_weight = None`
  2. **`M1_logreg_C1.0_cwNone`:** `C = 1.0, class_weight = None`
  3. **`M1_logreg_C2.0_cwNone`:** `C = 2.0, class_weight = None`
  4. **`M1_logreg_C0.5_cwBalanced`:** `C = 0.5, class_weight = "balanced"`
  5. **`M1_logreg_C1.0_cwBalanced`:** `C = 1.0, class_weight = "balanced"`
  6. **`M1_logreg_C2.0_cwBalanced`:** `C = 2.0, class_weight = "balanced"`

---

## 3. Validation Threshold Optimization Protocol (Phase 2 Pre-Test Amendment)

For every trained model, we preserve two complete validation evaluations:
- **Regime A (Default Thresholds):** Fixed threshold $\tau_k = 0.50$ for all five labels $k \in \{1, \dots, 5\}$.
- **Regime B (Validation-Tuned Thresholds):** Independent threshold $\tau_k^* \in [0.05, 0.95]$ selected on `data/processed/validation.csv` over the 91-point grid:
  $$\mathcal{T} = \{0.05, 0.06, 0.07, \dots, 0.95\} \quad (\text{step size } \Delta\tau = 0.01)$$

### Per-Label Objective Functions
1. **For `request` (Task A):** Maximize validation **$F_2$ score** ($\beta = 2$), weighting Recall twice as heavily as Precision so genuine aid requests are less likely to be missed.
2. **For `water`, `food`, `shelter`, `medical_help` (Task B):** Maximize validation per-label **$F_1$ score** ($\beta = 1$).

### Deterministic Tie-Breaking for Threshold Selection
If two candidate thresholds $\tau_a, \tau_b \in \mathcal{T}$ achieve the exact same validation objective score (rounded to 8 decimal places):
- Choose the threshold closer to $0.50$ ($|\tau - 0.50|$ minimized).
- If still tied (e.g., $0.45$ vs. $0.55$), choose the lower threshold to favor higher recall.

---

## 4. Deterministic Winner-Selection Rule (Phase 2 Pre-Test Amendment)

Because `docs/experiment_protocol.md` listed multiple primary metrics without a single scalar ranking formula, we pre-register the following deterministic hierarchy on `data/processed/validation.csv` before viewing test data:

1. **Primary Criterion:** Highest validation **Macro F1** (across all 5 labels) using each model's **validation-tuned thresholds** ($\tau^*$).
   - If two models are within $0.0001$ ($0.01\%$ Macro F1), they are treated as tied on the primary criterion.
2. **First Tie-Breaker:** Highest validation **`request` $F_2$ score** using validation-tuned thresholds.
3. **Second Tie-Breaker:** Model simplicity and calibration stability (`B0` < `B1` < `M1` with `class_weight=None` < `M1` with `class_weight="balanced"`, and smaller `C` preferred over larger `C`).

*Scientific Honesty Rule:* We do **not** force Logistic Regression (`M1`) to win. If a `B1` Naive Bayes variant achieves higher validation Macro F1 under this rule, `B1` will be selected.

---

## 5. Probability Calibration Protocol (Phase 2 Pre-Test Amendment)

1. For the winning model family/configuration, we compute per-label **Brier scores** and reliability curves on `data/processed/validation.csv`.
2. Because `class_weight="balanced"` in Logistic Regression (or uncalibrated likelihood ratios in Naive Bayes) shifts raw predicted probabilities toward $0.50$, we also fit a leakage-safe **Sigmoid-Calibrated version** of the winning estimator using 3-fold cross-validation **strictly on `data/processed/train.csv`** (`CalibratedClassifierCV(estimator, method="sigmoid", cv=3)`), and tune its decision thresholds on `data/processed/validation.csv`.
3. **Selection Between Uncalibrated and Sigmoid-Calibrated Winner:**
   - If the training-CV sigmoid-calibrated version improves mean validation Brier score **and** maintains validation tuned Macro F1 and `request` $F_2$ within $0.005$ ($0.5\%$) of the uncalibrated winner (or improves them), we select the calibrated pipeline so probability scores shown to volunteers are well-calibrated.
   - Otherwise, we retain the uncalibrated winner for maximum discriminative F1/$F_2$ performance and explicitly label its outputs as *uncalibrated model scores* in `models/model_manifest.json` and the UI.

---

## 6. Human-Review Decision Policy (Phase 2 Pre-Test Amendment)

To support the human-in-the-loop screening workflow without ever automatically discarding messages or claiming clinical urgency:

Given a message's 5 model-estimated scores $p_k \in [0, 1]$ and frozen validation thresholds $\tau_k \in (0, 1)$, we define a symmetric uncertainty half-width $\delta_k = 0.08$ around each threshold (clipped to $[0.02, 0.98]$):
$$\text{Uncertain Band}_k = [\max(0.02, \tau_k - 0.08),\; \min(0.98, \tau_k + 0.08)]$$

A message is flagged with `review_recommended = True` if **any** of the following deterministic rules hold:
1. **`predicted_aid_request_verification`:** `request` is predicted positive ($p_{\text{request}} \ge \tau_{\text{request}}$). All predicted aid requests are routed to the human verification queue.
2. **`borderline_request_score`:** $p_{\text{request}} \in [\tau_{\text{request}} - 0.08,\; \tau_{\text{request}})$ (score is just below the request threshold and could be a false negative).
3. **`topic_detected_with_near_threshold_request`:** At least one need topic (`water`, `food`, `shelter`, `medical_help`) is predicted positive ($p_k \ge \tau_k$), and $p_{\text{request}} \ge 0.5 \cdot \tau_{\text{request}}$ while $p_{\text{request}} < \tau_{\text{request}}$ (potential implicit aid request mentioning a survival need).
4. **`borderline_need_topic_score`:** One or more need topics fall inside their uncertainty band $|p_k - \tau_k| \le 0.08$ while `request` is positive or borderline.

Messages not triggering Rules 1–4 have `review_recommended = False` (low-priority / routine informational stream) but are **never deleted or permanently rejected**.


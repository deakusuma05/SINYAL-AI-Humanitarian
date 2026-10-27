# Pre-Registered Experiment Protocol — SINYAL

> **Phase 1 Notice:** This document pre-registers the modeling, validation, evaluation, and error-analysis protocol prior to any model training. **No models are trained or evaluated during Phase 1.**

---

## 1. Primary Research Hypothesis & Objective

**Research Question:**
Can a lightweight supervised NLP model classify humanitarian aid requests (`request`) and relevant need categories (`water`, `food`, `shelter`, `medical_help`) more effectively than simple baseline models on held-out disaster-response messages?

**Experimental Design Principle:**
We do **not** assume a priori that the candidate model (`M1`) will outperform simpler baselines (`B0`, `B1`) across all metrics or labels. Every model must be evaluated under identical split boundaries and pre-registered metrics.

---

## 2. Prediction Task Formulation

### Output Space
Each message $x_i$ is associated with a 5-dimensional binary target vector:
$$y_i = \left(y_{i,\text{request}},\; y_{i,\text{water}},\; y_{i,\text{food}},\; y_{i,\text{shelter}},\; y_{i,\text{medical\_help}}\right) \in \{0, 1\}^5$$

```python
TARGET_LABELS = [
    "request",
    "water",
    "food",
    "shelter",
    "medical_help",
]
```

### Task Separation Rule
1. **Task A — Request Intent (`request`):** Binary classification predicting whether the message represents an aid request.
2. **Task B — Need-Topic Identification (`water`, `food`, `shelter`, `medical_help`):** Multilabel binary classification predicting which need topics are discussed in the message.
3. **Independence of Dimensions:** The five binary targets are modeled using a multilabel-compatible one-vs-rest (one independent binary estimator per label) formulation. They are **never** collapsed into a single mutually exclusive multiclass target, and topic predictions are **never** automatically equated to aid requests.

---

## 3. Model Candidates

All models operate strictly on CPU using `scikit-learn` and are serialized via `joblib` for Phase 3 FastAPI inference.

### Model B0 — Majority / Stratified Dummy Baseline
- **Architecture:** `DummyClassifier` wrapped in `MultiOutputClassifier` (or one binary dummy estimator per target label).
- **Strategy:** Evaluates both `most_frequent` (predicting all `0`s given minority positive prevalence, yielding $0.0$ F1) and `stratified` / `prior` reference baselines so minimum non-trivial performance floors are transparently documented.
- **Purpose:** Establishes the lower-bound performance floor under severe class imbalance.

### Model B1 — Naive Bayes Baseline
- **Architecture:** `Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultiOutputClassifier(MultinomialNB()))])`
- **Features:** Word/character n-gram TF-IDF representations learned strictly on `train.csv`.
- **Classifier:** One independent `MultinomialNB` binary classifier per target label.
- **Purpose:** Provides a fast generative/probabilistic lexical baseline for text classification.

### Model M1 — Candidate Linear Classifier
- **Architecture:** `Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultiOutputClassifier(LogisticRegression()))])`
- **Features:** TF-IDF vectorizer fit exclusively on `train.csv` (preserving numbers, negations, and urgency expressions).
- **Classifier:** One independent binary `LogisticRegression` estimator per target label (with validation-tuned regularization `C` and class-weight configurations such as `None` vs. `"balanced"`).
- **Purpose:** Tests whether discriminative linear classification with class-imbalance handling improves per-label F1 and `request` $F_2$ over `B0` and `B1`.

---

## 4. Strict Data Split Protocol & Leakage Prevention

1. **Training Split (`data/processed/train.csv`):**
   - Used **exclusively** to fit the TF-IDF vocabulary, IDF weights, and classifier parameters.
   - No validation or test documents may ever be included when fitting `TfidfVectorizer`.
2. **Validation Split (`data/processed/validation.csv`):**
   - Used to compare hyperparameter configurations (e.g., n-gram range, minimum document frequency, regularization strength `C`, class weighting).
   - Used to select per-label classification probability thresholds $\tau_k \in (0, 1)$ and uncertainty review bands.
   - Used to evaluate probability calibration (e.g., Brier score / reliability diagrams) before treating predicted scores as confidence indicators.
3. **Held-Out Test Split (`data/processed/test.csv`):**
   - Evaluated **only once** in Phase 2 after all model architectures, preprocessing parameters, decision thresholds, and uncertainty bands are frozen.
   - Test-set label prevalence or performance must **never** be used to tune features, hyperparameters, or thresholds.

---

## 5. Evaluation Metrics

### Primary Metrics
1. **Macro F1 (across all 5 labels):** Unweighted mean of per-label F1 scores; penalizes poor performance on rarer need categories (`water`, `medical_help`, `shelter`).
2. **Micro F1 (across all 5 labels):** Global F1 computed over pooled true positives, false positives, and false negatives across all five targets.
3. **Per-Label F1 Score:** Individual $F_1$ for `request`, `water`, `food`, `shelter`, and `medical_help`.
4. **Per-Label Precision:** $\frac{\text{TP}_k}{\text{TP}_k + \text{FP}_k}$ for each label $k$.
5. **Per-Label Recall:** $\frac{\text{TP}_k}{\text{TP}_k + \text{FN}_k}$ for each label $k$.
6. **$F_2$ Score for `request`:**
   $$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}, \quad \beta = 2$$
   In humanitarian request screening, failing to surface a genuine request (False Negative) is operationally costlier than routing a non-request to human review (False Positive). Therefore, $F_2$ on `request` is a primary operational metric.

### Supporting Metrics
1. **PR-AUC (Average Precision):** Area under the Precision-Recall curve per label, providing a threshold-independent measure suited for imbalanced binary targets.
2. **Per-Label Confusion Matrices:** Explicit counts of $\text{TN}, \text{FP}, \text{FN}, \text{TP}$ for each of the 5 labels.
3. **Per-Label Support:** Exact number of positive and negative ground-truth instances in the evaluated split.
4. **Selective Prediction Coverage & Review Rate:** Percentage of messages routed to the **Human Review Queue** versus confident predictions, along with metric improvements on the non-deferred subset.

---

## 6. Uncertainty-Aware Workflow & Calibration Protocol

To support the workflow **Message → Request intent prediction → Need-topic predictions → Uncertainty indication → Human review queue**:
1. **No Unexamined Raw Probabilities:** Raw outputs from `MultinomialNB` or class-weighted `LogisticRegression` are not assumed to be well-calibrated probabilities. Phase 2 will inspect calibration on the validation split (and apply `CalibratedClassifierCV` on validation/cross-validation if needed).
2. **Validation-Defined Uncertainty Band:** For each binary target $k$ with validation-selected decision threshold $\tau_k$, predictions whose probability $p_k(x)$ falls within a validation-defined margin $[\tau_k - \delta_k,\; \tau_k + \delta_k]$ (or where a topic is predicted positive while `request` is borderline) are flagged as **Uncertain — Needs Human Review**.
3. **No Test-Set Optimization:** Neither $\tau_k$ nor $\delta_k$ may be adjusted after viewing test-set results.

---

## 7. Pre-Registered Error Analysis & Limitations Protocol

During Phase 2 evaluation, qualitative and quantitative error analysis will inspect the following mandatory categories on validation (and summarize on held-out test without retuning):

1. **False Negatives Involving Aid Requests (`request` FN):** Direct or implicit pleas for help missed by the classifier (e.g., indirect phrasing, brief location + need statements).
2. **False Positives on Informational Messages (`request` FP):** News reports, government announcements, or donations/offers misclassified as aid requests because they mention food, water, or shelter.
3. **Multi-Need Co-occurrence Errors:** Messages expressing multiple simultaneous needs (e.g., food + water + shelter) where the model captures dominant high-frequency needs (`food`) but misses lower-prevalence needs (`water` or `medical_help`).
4. **Unusual Wording, Abbreviations, and SMS Noise:** Telegraphic SMS phrasing, typos, numeric quantities, or transliterated location names.
5. **Translation-Related Ambiguity:** Artifacts arising because `message` contains English translations of original messages (`original` column) written in Haitian Creole, Spanish, or other languages.
6. **Domain Differences Across Genres (`direct` vs. `social` vs. `news`):** Performance disparities between direct survivor texts (`direct`), social media posts (`social`), and journalistic reporting (`news`).

### Methodological Guardrails
- **Dataset Bias & Distribution Shift:** Benchmark results reflect historical crises curated by Figure Eight / Appen and cannot be extrapolated to arbitrary future disasters without domain adaptation and human oversight.
- **No Claim of Field Validation:** High test-set F1 or $F_2$ scores demonstrate offline benchmark effectiveness only and must never be reported as field-validated humanitarian impact.


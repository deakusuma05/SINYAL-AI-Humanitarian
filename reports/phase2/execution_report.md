# SINYAL PHASE 2 — EXECUTION REPORT

## A. Status
**PASS** — All mandatory Phase 2 scientific, engineering, threshold-tuning, calibration, human-review policy, single-shot held-out testing, serialization, and automated testing requirements have been completed and verified.

---

## B. Environment
- **OS:** Windows 11 (`Windows-11-10.0.26200-SP0`)
- **Python Version:** Python `3.12.10` (`.\.venv\Scripts\python.exe`)
- **Dependency Versions:**
  - `pandas == 2.3.3`
  - `numpy == 2.5.4`
  - `scikit-learn == 1.9.1` (`scipy == 1.18.1`, `joblib == 1.6.0`)
  - `matplotlib == 3.11.2`
  - `pytest == 9.1.1`
  - `requests == 2.34.2`
- **Runtime:**
  - `src.modeling.train_validate`: `131.99s` (~2.2 minutes for all 11 baseline/candidate pipelines + 3-fold CV sigmoid calibration + threshold optimization + 8 validation charts + serialization).
  - `src.modeling.evaluate_test --confirm-frozen`: `~9.5s`.

---

## C. Data Verification (Pre-Flight Audit)

| Split | File Path | Verified Rows | Expected SHA-256 | Actual SHA-256 | Match |
|---|---|---:|---|---|---|
| **Training** | `data/processed/train.csv` | `19,032` | `13d0c0336f5ed8bc89c19ee3529075e5638f7aaf1976f1aaeefd0a3a75ecb02b` | `13d0c0336f5ed8bc89c19ee3529075e5638f7aaf1976f1aaeefd0a3a75ecb02b` | **PASS** |
| **Validation** | `data/processed/validation.csv` | `2,329` | `e09d9842d17414e08946739d93853db1687a873a08a25210ea65bb5aec7b722e` | `e09d9842d17414e08946739d93853db1687a873a08a25210ea65bb5aec7b722e` | **PASS** |
| **Cleaned Held-Out Test** | `data/processed/test.csv` | `2,387` | `d0f7265d67753a27a40a0fcc6caab8f88849b352427a71929ac692b37b06d75a` | `d0f7265d67753a27a40a0fcc6caab8f88849b352427a71929ac692b37b06d75a` | **PASS** |

*Note:* `data/processed/test.csv` (`2,387` rows) is the **cleaned held-out test set** (with 10 duplicate IDs removed deterministically during Phase 1 from the 2,397-row raw test CSV).

---

## D. Trained Models & Configurations

All models were fit strictly on `data/processed/train.csv` (`19,032` rows) with `RANDOM_SEED = 42` and shared `TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.98, max_features=30000, sublinear_tf=True)`:

| Model ID | Family | Configuration | Fit Duration (s) |
|---|---|---|---:|
| `B0_most_frequent` | `B0` | `DummyClassifier(strategy="most_frequent")` | `1.840s` |
| `B0_stratified` | `B0` | `DummyClassifier(strategy="stratified", random_state=42)` | `1.742s` |
| `B1_nb_alpha_0.5` | `B1` | `MultinomialNB(alpha=0.5)` | `1.807s` |
| `B1_nb_alpha_1.0` | `B1` | `MultinomialNB(alpha=1.0)` | `1.867s` |
| `B1_nb_alpha_2.0` | `B1` | `MultinomialNB(alpha=2.0)` | `1.871s` |
| `M1_logreg_C0.5_cwNone` | `M1` | `LogisticRegression(C=0.5, class_weight=None)` | `2.831s` |
| `M1_logreg_C1.0_cwNone` | `M1` | `LogisticRegression(C=1.0, class_weight=None)` | `3.605s` |
| `M1_logreg_C2.0_cwNone` | `M1` | `LogisticRegression(C=2.0, class_weight=None)` | `3.920s` |
| `M1_logreg_C0.5_cwBalanced` | `M1` | `LogisticRegression(C=0.5, class_weight="balanced")` | `3.447s` |
| `M1_logreg_C1.0_cwBalanced` | `M1` | `LogisticRegression(C=1.0, class_weight="balanced")` | `2.736s` |
| `M1_logreg_C2.0_cwBalanced` | `M1` | `LogisticRegression(C=2.0, class_weight="balanced")` | `2.880s` |
| `M1_logreg_C2.0_cwNone_sigmoid_cv3` | `M1_Calibrated` | `CalibratedClassifierCV(M1_logreg_C2.0_cwNone, method="sigmoid", cv=3)` | `4.858s` |

---

## E. Validation Results (`data/processed/validation.csv`, 2,329 Rows)

Full side-by-side comparison across both **Default ($\tau = 0.50$)** and **Validation-Tuned ($\tau^*$)** regimes (`reports/phase2/model_comparison_validation.csv`):

| Model ID | Regime | Macro F1 | Micro F1 | `request` F2 | `request` F1 | `water` F1 | `food` F1 | `shelter` F1 | `medical_help` F1 | Mean AP | Mean Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B0_most_frequent` | Default `0.50` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.1052` | `0.1052` |
| `B0_stratified` | Default / Tuned | `0.0995` | `0.1174` | `0.1944` | `0.1914` | `0.0316` | `0.1045` | `0.0943` | `0.0758` | `0.1053` | `0.1885` |
| `B1_nb_alpha_0.5` | Default `0.50` | `0.3590` | `0.4786` | `0.6912` | `0.7087` | `0.1734` | `0.4727` | `0.3241` | `0.1161` | `0.5353` | `0.0654` |
| `B1_nb_alpha_0.5` | **Tuned $\tau^*$** | `0.5303` | `0.5570` | `0.7362` | `0.6491` | `0.5198` | `0.5766` | `0.5446` | `0.3615` | `0.5353` | `0.0654` |
| `B1_nb_alpha_1.0` | Default `0.50` | `0.2303` | `0.3750` | `0.6223` | `0.6843` | `0.0503` | `0.2866` | `0.1111` | `0.0193` | `0.4923` | `0.0717` |
| `B1_nb_alpha_1.0` | **Tuned $\tau^*$** | `0.4849` | `0.5309` | `0.7312` | `0.6703` | `0.4229` | `0.5448` | `0.4787` | `0.3077` | `0.4923` | `0.0717` |
| `B1_nb_alpha_2.0` | Default `0.50` | `0.1350` | `0.2514` | `0.4612` | `0.5588` | `0.0000` | `0.1064` | `0.0098` | `0.0000` | `0.4275` | `0.0798` |
| `B1_nb_alpha_2.0` | **Tuned $\tau^*$** | `0.4060` | `0.4812` | `0.7140` | `0.6403` | `0.2909` | `0.4843` | `0.3914` | `0.2230` | `0.4275` | `0.0798` |
| `M1_logreg_C0.5_cwNone` | Default `0.50` | `0.4382` | `0.5162` | `0.6210` | `0.6906` | `0.5114` | `0.5658` | `0.3765` | `0.0469` | `0.6698` | `0.0555` |
| `M1_logreg_C0.5_cwNone` | **Tuned $\tau^*$** | `0.6510` | `0.6592` | `0.7524` | `0.6736` | `0.6540` | `0.7430` | `0.6538` | `0.5305` | `0.6698` | `0.0555` |
| `M1_logreg_C1.0_cwNone` | Default `0.50` | `0.5211` | `0.5808` | `0.6624` | `0.7184` | `0.5362` | `0.6407` | `0.5000` | `0.2101` | `0.6831` | `0.0519` |
| `M1_logreg_C1.0_cwNone` | **Tuned $\tau^*$** | `0.6668` | `0.6726` | `0.7617` | `0.6819` | `0.6777` | `0.7633` | `0.6846` | `0.5263` | `0.6831` | `0.0519` |
| **`M1_logreg_C2.0_cwNone`** | Default `0.50` | `0.5770` | `0.6187` | `0.6723` | `0.7181` | `0.6016` | `0.6740` | `0.6025` | `0.2890` | **`0.6895`** | **`0.0498`** |
| **`M1_logreg_C2.0_cwNone`** | **Tuned $\tau^*$ (WINNER)** | **`0.6783`** | **`0.6835`** | **`0.7648`** | **`0.6854`** | **`0.7022`** | **`0.7724`** | **`0.6961`** | **`0.5356`** | **`0.6895`** | **`0.0498`** |
| `M1_logreg_C0.5_cwBalanced` | Default `0.50` | `0.6427` | `0.6486` | `0.7488` | `0.6877` | `0.6685` | `0.7534` | `0.6049` | `0.4991` | `0.6791` | `0.0881` |
| `M1_logreg_C0.5_cwBalanced` | **Tuned $\tau^*$** | `0.6719` | `0.6820` | `0.7544` | `0.6882` | `0.6872` | `0.7643` | `0.6918` | `0.5280` | `0.6791` | `0.0881` |
| `M1_logreg_C1.0_cwBalanced` | Default `0.50` | `0.6532` | `0.6599` | `0.7536` | `0.6921` | `0.6777` | `0.7586` | `0.6307` | `0.5068` | `0.6828` | `0.0764` |
| `M1_logreg_C1.0_cwBalanced` | **Tuned $\tau^*$** | `0.6740` | `0.6815` | `0.7642` | `0.6818` | `0.6866` | `0.7751` | `0.6944` | `0.5320` | `0.6828` | `0.0764` |
| `M1_logreg_C2.0_cwBalanced` | Default `0.50` | `0.6584` | `0.6674` | `0.7518` | `0.6969` | `0.6723` | `0.7719` | `0.6500` | `0.5010` | `0.6844` | `0.0678` |
| `M1_logreg_C2.0_cwBalanced` | **Tuned $\tau^*$** | `0.6724` | `0.6772` | `0.7652` | `0.6667` | `0.6990` | `0.7790` | `0.6983` | `0.5187` | `0.6844` | `0.0678` |
| `M1_logreg_C2.0_cwNone_sigmoid_cv3` | Default `0.50` | `0.6270` | `0.6529` | `0.6913` | `0.7258` | `0.6296` | `0.6946` | `0.6735` | `0.4116` | `0.6735` | `0.0513` |
| `M1_logreg_C2.0_cwNone_sigmoid_cv3` | **Tuned $\tau^*$** | `0.6571` | `0.6619` | `0.7507` | `0.6554` | `0.6772` | `0.7495` | `0.6841` | `0.5195` | `0.6735` | `0.0513` |

---

## F. Selected Model

- **Winner:** **`M1_logreg_C2.0_cwNone`** (`TfidfVectorizer` + `MultiOutputClassifier(LogisticRegression(C=2.0, class_weight=None, solver='lbfgs', max_iter=1000, random_state=42))`) with validation-selected thresholds.
- **Why It Won:**
  1. Achieved the **highest validation Macro F1 (`0.6783`)** and **highest validation Micro F1 (`0.6835`)** across all evaluated models under the pre-registered selection rule in `docs/phase2_selection_policy.md`.
  2. Achieved the **highest Mean Average Precision (`0.6895`)** and **lowest Mean Brier Score (`0.0498`)** of any model—substantially better calibrated than `class_weight="balanced"` models (`Mean Brier = 0.0678–0.0881`).
  3. Outperformed the best `B1` Naive Bayes baseline (`B1_nb_alpha_0.5`, Tuned Macro F1 = `0.5303`, `request` $F_2$ = `0.7362`) by **`+14.80` percentage points in Macro F1** and **`+2.85` percentage points in `request` $F_2$** on validation.

---

## G. Thresholds (Default `0.50` vs. Validation-Tuned $\tau^*$ for `M1_logreg_C2.0_cwNone`)

| Target Label | Tuning Objective | Default $\tau$ | Selected $\tau^*$ | Validation Metric Before ($\tau=0.50$) | Validation Metric After ($\tau^*$) | Default Precision / Recall | Tuned Precision / Recall | Tuned Pred. Positive Rate |
|---|---|---:|---:|---:|---:|---|---|---:|
| **`request`** | **$F_2$** | `0.50` | **`0.19`** | `0.6723` ($F_2$) | **`0.7648`** ($F_2$) | `0.8101` / `0.6448` | `0.5844` / **`0.8287`** | `24.17%` |
| **`water`** | **$F_1$** | `0.50` | **`0.22`** | `0.6016` ($F_1$) | **`0.7022`** ($F_1$) | `0.8132` / `0.4774` | `0.6829` / **`0.7226`** | `7.04%` |
| **`food`** | **$F_1$** | `0.50` | **`0.26`** | `0.6740` ($F_1$) | **`0.7724`** ($F_1$) | `0.8063` / `0.5789` | `0.7667` / **`0.7782`** | `11.59%` |
| **`shelter`** | **$F_1$** | `0.50` | **`0.32`** | `0.6025` ($F_1$) | **`0.6961`** ($F_1$) | `0.8220` / `0.4755` | `0.7403` / **`0.6569`** | `7.77%` |
| **`medical_help`** | **$F_1$** | `0.50` | **`0.20`** | `0.2890` ($F_1$) | **`0.5356`** ($F_1$) | `0.6333` / `0.1872` | `0.5343` / **`0.5369`** | `8.76%` |

---

## H. Held-Out Test Performance (`data/processed/test.csv`, 2,387 Rows — Single-Shot Evaluation)

Evaluated **exactly once** using `src.modeling.evaluate_test --confirm-frozen` and locked via `reports/phase2/final_evaluation_lock.json`:

### 1. Overall Multi-Label Summary on Cleaned Held-Out Test Set

| Metric | Frozen Tuned Thresholds ($\tau^*$) | Default Thresholds ($\tau = 0.50$ Reference) |
|---|---:|---:|
| **Macro F1 (5 Labels)** | **`0.7166`** | `0.6113` |
| **Micro F1 (5 Labels)** | **`0.7261`** | `0.6489` |
| **`request` $F_2$ Score (Primary Task A)** | **`0.7781`** | `0.6284` |
| **`request` $F_1$ Score** | **`0.7404`** | `0.7099` |
| **Mean Average Precision (mAP)** | **`0.7643`** | `0.7643` |
| **Mean Brier Score** | **`0.0495`** | `0.0495` |
| **Hamming Loss** | **`0.0654`** | `0.0650` |
| **Exact-Match Subset Accuracy (Secondary)** | `0.7264` | `0.7440` |

### 2. Per-Label Held-Out Test Metrics & Confusion Matrices ($\tau^*$)

| Target Label | Pos. Support | Neg. Support | Frozen $\tau^*$ | Precision | Recall | F1 | F2 | Average Precision | Brier Score | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`request`** | `478` | `1,909` | `0.19` | `0.6851` | **`0.8054`** | **`0.7404`** | **`0.7781`** | `0.8305` | `0.0739` | `1,732` | `177` | `93` | `385` |
| **`water`** | `191` | `2,196` | `0.22` | `0.8041` | **`0.8168`** | **`0.8104`** | `0.8142` | `0.8633` | `0.0285` | `2,158` | `38` | `35` | `156` |
| **`food`** | `308` | `2,079` | `0.26` | `0.8536` | **`0.7760`** | **`0.8129`** | `0.7903` | `0.8526` | `0.0442` | `2,038` | `41` | `69` | `239` |
| **`shelter`** | `224` | `2,163` | `0.32` | `0.7906` | **`0.6741`** | **`0.7277`** | `0.6946` | `0.8296` | `0.0381` | `2,123` | `40` | `73` | `151` |
| **`medical_help`** | `211` | `2,176` | `0.20` | `0.4906` | **`0.4929`** | **`0.4917`** | `0.4924` | `0.4453` | `0.0626` | `2,068` | `108` | `107` | `104` |

### 3. Exploratory Queue-Ranking Utility on Held-Out Test Set (`request` Score Ranking)

| Queue Fraction Reviewed | Messages Reviewed | True Requests Captured (out of `478`) | Model `Recall@k` | Model `Precision@k` | Random Baseline `Recall@k` | Lift Over Random |
|---|---:|---:|---:|---:|---:|---:|
| **Top 10% Reviewed** | `239` | `223` | **`46.65%`** | `93.31%` | `10.77%` | **`4.33x`** |
| **Top 20% Reviewed** | `477` | `359` | **`75.10%`** | `75.26%` | `19.98%` | **`3.76x`** |
| **Top 30% Reviewed** | `716` | `406` | **`84.94%`** | `56.70%` | `29.58%` | **`2.87x`** |

---

## I. Error Analysis & Subgroup Robustness Summary

Full details and redacted examples are recorded in [`reports/phase2/error_analysis.md`](error_analysis.md):
- **False Negatives (`request`):** `68` on validation (`47` in `news`, `21` in `direct`); `93` on test (`47` in `news`, `46` in `direct`). Primarily caused by third-person journalistic phrasing in `news` and indirect/telegraphic inquiries without explicit need nouns.
- **False Positives (`request`):** `234` on validation (`195` in `direct`, `39` in `news`); `177` on test (`148` in `direct`, `29` in `news`). Notably, several validation "false positives" are genuine survival pleas in the raw unreviewed validation set that lacked `request=1` tags; on the professionally reviewed test set, `request` Precision improves by `+10.07%` (`0.5844` $\rightarrow$ `0.6851`).
- **Genre Disparity (`direct` vs. `news` on Held-Out Test):**
  - **`direct` (`1,067` test rows, `421` true requests):** `request` Precision = `0.7156`, **Recall = `0.8907`**, **$F_1$ = `0.7937`**, **$F_2$ = `0.8492`**, Macro F1 = `0.7484`, Micro F1 = `0.7973`.
  - **`news` (`1,320` test rows, `57` true requests):** `request` Precision = `0.2632`, Recall = `0.1754`, $F_1$ = `0.2105`, $F_2$ = `0.1880`, Macro F1 = `0.5559`, Micro F1 = `0.5840`.

---

## J. Probability Calibration

- **Uncalibrated Winner (`M1_logreg_C2.0_cwNone`):** Validation Mean Brier Score = **`0.049786`** (`request`: `0.0679`, `water`: `0.0316`, `food`: `0.0467`, `shelter`: `0.0419`, `medical_help`: `0.0607`). Because `class_weight=None` was used with standard cross-entropy loss, the raw logistic probabilities are already well-aligned with empirical positive frequencies (`reports/phase2/figures/fig7_reliability_diagram.png`).
- **Comparison vs. 3-Fold Training CV Sigmoid Calibration (`M1_logreg_C2.0_cwNone_sigmoid_cv3`):** Post-hoc sigmoid calibration slightly worsened validation Mean Brier Score (`0.051313` vs. `0.049786`) and reduced tuned Macro F1 by `-0.0212` (`0.6571` vs. `0.6783`).
- **Decision:** Per the pre-registered rule in `docs/phase2_selection_policy.md` Section 5, we retained the uncalibrated `M1_logreg_C2.0_cwNone` pipeline and explicitly recorded `calibration_status = "uncalibrated_raw_probabilities"` in `models/model_manifest.json`.

---

## K. Human-Review Decision Policy

- **Exact Policy Rules (`models/decision_policy.json`):**
  1. `predicted_aid_request_verification`: `request_score >= 0.19` (all predicted aid requests are routed to human verification).
  2. `borderline_request_score_below_threshold`: `0.11 <= request_score < 0.19` (within $\delta = 0.08$ below threshold).
  3. `topic_detected_with_moderate_request_score`: $\ge 1$ need topic predicted positive AND `0.095 <= request_score < 0.19`.
  4. `borderline_need_topic_score`: $\ge 1$ need topic within $\pm 0.08$ of its threshold while `request_score >= 0.095`.
  5. Low-scoring messages (`review_recommended = False`) are retained in the routine stream and **never automatically discarded**.
- **Measured Review Policy Performance:**
  - **Validation Split (`2,329` rows):** Flagged for review = `810` (`34.78%`), Uncertainty indicated = `479` (`20.57%`). Captures **`357 / 397` (`89.92%` recall)** of all true requests (`+28` additional true requests rescued beyond binary classification). Remaining unflagged subset (`65.22%` coverage) has `97.37%` negative predictive accuracy for `request`.
  - **Cleaned Held-Out Test Split (`2,387` rows):** Flagged for review = `810` (`33.93%`), Uncertainty indicated = `462` (`19.35%`). Captures **`422 / 478` (`88.28%` recall)** of all true requests (`+37` additional true requests rescued beyond binary threshold alone). Remaining unflagged subset (`66.07%` coverage) has `96.45%` negative predictive accuracy for `request`.

---

## L. Serialized Artifacts

1. **`models/sinyal_model.joblib`** (`SHA-256: 0589ce46b0ed26e5aa229b1a1d07c47d9c0f396af58b615e532777bdf12b3142`) — End-to-end fitted `Pipeline` (`TfidfVectorizer` + `MultiOutputClassifier(LogisticRegression)`). Reload parity verified (`max_abs_diff = 0.00e+00`).
2. **`models/model_manifest.json`** (`SHA-256: b9a5bdcdfe6c7364ff22758a589b066e300da38bf0d4e6ab0660b80dadb84164`) — Frozen model metadata, dataset checksums, hyperparameters, thresholds, and environment versions.
3. **`models/decision_policy.json`** (`SHA-256: 48ebd3d04cc62632bd04bc3b2f8aff09d684291f553bb9f3929c389420e1b68d`) — Frozen thresholds, uncertainty bands ($\pm 0.08$), and human-review routing rules.
4. **`reports/phase2/final_evaluation_lock.json`** — Cryptographic single-shot evaluation lock preventing repeated test-set runs.

---

## M. Automated Tests

- **Phase 1 Data Pipeline Tests (`tests/test_data_pipeline.py`):** `10` passed, `0` failed.
- **Phase 2 Model & Inference Pipeline Tests (`tests/test_model_pipeline.py`):** `14` passed, `0` failed.
- **Total:** `24` passed, `0` failed, `0` unresolved.


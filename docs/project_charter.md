# Project Charter — SINYAL: AI-Powered Humanitarian Request Triage

## 1. Problem Statement
During natural disasters and humanitarian crises, disaster-response teams and volunteer groups face a high volume of incoming communications across multiple channels (direct SMS/helpline reports, social media messages, and news updates). These messages contain a heterogeneous mix of urgent requests for basic survival needs (drinking water, food, shelter, medical assistance), general situational updates, offers of charity or supplies, and off-topic noise.

Humanitarian volunteers must manually read and screen large queues of incoming messages to identify actionable aid requests and route them to appropriate topic teams. Under severe time pressure, manual screening of unfiltered message streams increases cognitive load and delays attention to relevant requests.

## 2. Target Users
- **Primary Users:** Humanitarian aid screening volunteers, disaster-response information coordinators, and digital humanitarian triage teams.
- **Secondary Users:** Humanitarian technology researchers and NGO operations analysts evaluating lightweight NLP screening workflows.

## 3. Research Question
> *Can a lightweight supervised NLP model classify humanitarian aid requests and relevant need categories more effectively than simple baseline models on held-out disaster-response messages?*

## 4. Project Scope & Explicit Non-Goals

### In Scope
- Assisting human reviewers with message screening and information organization.
- Supervised natural language processing for two distinct dimensions:
  1. **Task A (Request Intent):** Binary classification of whether a message contains an explicit or implicit request for humanitarian assistance (`request`).
  2. **Task B (Need-Topic Identification):** Multilabel binary classification of whether a message discusses four core humanitarian need topics (`water`, `food`, `shelter`, `medical_help`).
- Uncertainty-aware screening indicators to flag low-confidence or ambiguous predictions for prioritized human verification.
- A lightweight, CPU-friendly architecture consisting of a **Python + FastAPI + Scikit-learn/Joblib** backend and a **HTML5 + CSS3 + Vanilla JavaScript + Chart.js** command-center dashboard (implemented in Phase 3; no Streamlit is used).

### Explicit Non-Goals (Safety & Operational Boundaries)
SINYAL is a **decision-support prototype for information organization**, **NOT** an autonomous emergency dispatch platform. Specifically, SINYAL will **not**:
- Diagnose medical conditions or provide clinical triage.
- Determine clinical urgency, disaster severity, or life-safety priority scores.
- Verify or promise that an incoming message or request is authentic.
- Automatically deploy volunteers, ambulances, or physical supplies.
- Decide who deserves aid or rank human lives.
- Replace trained humanitarian professionals or human review.

## 5. Input / Output Definition

### Input
- An incoming disaster-related text message in English (or English translation of an original source message), optionally accompanied by source channel metadata (`genre`: `direct`, `social`, or `news`).

### Output
The intended end-to-end decision-support workflow is:
$$\text{Message} \rightarrow \text{Request Intent Prediction} \rightarrow \text{Need-Topic Predictions} \rightarrow \text{Uncertainty Indication} \rightarrow \text{Human Review Queue}$$

Specifically, for each message:
1. **Request Intent Prediction:** Binary indicator and probability estimate for `request` $\in \{0, 1\}$.
2. **Need-Topic Predictions:** Independent binary indicators and probability estimates for `water`, `food`, `shelter`, and `medical_help` $\in \{0, 1\}^4$.
   - *Critical Semantic Rule:* Need-topic labels indicate that the message *discusses* the topic, not necessarily that the sender is *requesting* that resource. Intent (`request`) and topic (`water`, `food`, `shelter`, `medical_help`) are modeled and presented as separate dimensions.
3. **Uncertainty Indication:** A validation-calibrated review flag indicating whether predictions fall within an uncertain margin requiring human verification.
4. **Human Review Queue Routing:** Structured presentation in the human review queue interface so volunteers can inspect original text, predicted tags, and uncertainty status before taking action.

## 6. Expected Contribution
- **Methodological:** A transparent, reproducible, leakage-audited data pipeline and pre-registered experimental evaluation comparing a Dummy/Majority baseline (`B0`), a TF-IDF + Multinomial Naive Bayes baseline (`B1`), and a TF-IDF + Logistic Regression candidate (`M1`) on official disaster-response splits.
- **Practical Artifact:** A lightweight, CPU-executable decision-support prototype (FastAPI REST backend + vanilla HTML5/CSS3/JS frontend) demonstrating how request intent, multilabel need topics, and uncertainty indicators can be presented to human reviewers without overclaiming autonomous capability.

*Note: All contributions are framed strictly as potential decision-support impact on held-out benchmark data. No field interviews, institutional partnerships, measured volunteer time savings, or live deployment impacts are claimed.*

## 7. Known Risks & Limitations
1. **False Negatives on Genuine Aid Requests:** Missing a real request (`request = 1` predicted as `0`) is the primary operational risk in message screening. To mitigate this, evaluation explicitly tracks Recall and the $F_2$ score (which weights recall twice as heavily as precision) for the `request` label, alongside human review routing.
2. **Semantic Conflation of Topic and Intent:** A news report stating *"Clean water was distributed in Port-au-Prince"* discusses `water = 1` but has `request = 0`. Treating topic mentions as aid requests causes false alarms. SINYAL enforces strict separation between `request` and need topics.
3. **Translation & Language Ambiguity:** The dataset's `message` field contains English text or English translations of messages originally written in Haitian Creole, Spanish, and other languages (`original`). Translation artifacts or colloquial phrasing may degrade classifier reliability.
4. **Class Imbalance:** Several critical need categories (such as `water`, `shelter`, and `medical_help`) represent a small minority of all messages, making naive accuracy misleading.
5. **Domain & Event Shift:** Messages originate from specific historical disasters (e.g., Haiti 2010, Pakistan 2010, Chile 2010) and specific channels (`direct`, `social`, `news`). Performance on held-out benchmark splits does not guarantee generalization to future disasters or unrepresented languages.

## 8. Alignment with Design Science Research Methodology (DSRM)
1. **Problem Identification and Motivation (Phase 1):** Documented the information overload problem in disaster message screening and strict safety boundaries against autonomous dispatch.
2. **Definition of Solution Objectives (Phase 1):** Defined quantitative evaluation metrics (Macro/Micro F1, per-label Precision/Recall/F1, and `request` $F_2$ score) and qualitative artifact constraints (CPU-only, separation of intent vs. topic, uncertainty flagging).
3. **Design and Development (Phase 1–3):**
   - *Phase 1:* Reproducible data audit, cross-split leakage remediation, and deterministic preprocessing pipeline.
   - *Phase 2:* Supervised baseline and candidate model development with validation threshold calibration.
   - *Phase 3:* FastAPI backend and HTML5/CSS3/JS humanitarian command-center interface.
4. **Demonstration (Phase 2–3):** Execution of the trained pipeline on held-out test instances and interactive screening workflows.
5. **Evaluation (Phase 2–3):** Rigorous comparison of `M1` against `B0` and `B1` under the pre-registered protocol (`docs/experiment_protocol.md`).
6. **Communication (Phase 1–3):** Reproducible reports, provenance documentation, audit logs, and transparent architectural documentation.

## 9. Alignment with CRISP-DM
1. **Business Understanding (Phase 1):** Framed the humanitarian screening task, user roles, semantic distinction between request intent and need topics, and ethical non-goals (`docs/project_charter.md`).
2. **Data Understanding (Phase 1):** Conducted automated verification of raw files, SHA-256 checksums, missingness, class prevalence, genre distributions, and duplicate/leakage audits (`src/audit_data.py`, `reports/data_audit.md`).
3. **Data Preparation (Phase 1):** Implemented deterministic filtering of unusable/leaking records while preserving official train/validation/test split boundaries (`src/prepare_data.py`, `reports/preprocessing_summary.json`).
4. **Modeling (Phase 2):** Fit TF-IDF vectorizers and multilabel classifiers (`B0`, `B1`, `M1`) strictly on training data.
5. **Evaluation (Phase 2):** Perform hyperparameter/threshold selection on validation data and single-shot evaluation on the held-out test set.
6. **Deployment (Phase 3):** Serve serialized Joblib models via FastAPI REST endpoints consumed by a custom HTML5/CSS3/JS dashboard.

## 10. Sustainable Development Goals (SDG) Relevance
- **SDG 11 (Sustainable Cities and Communities — Target 11.5):** Supports research into disaster preparedness and information triage tools that could potentially assist response volunteers during urban and community emergencies.
- **SDG 13 (Climate Action — Target 13.1):** Contributes a reproducible open-source prototype aimed at strengthening adaptive capacity and information organization for climate-related hazards and natural disasters.

*Disclaimer: SDG alignment reflects the thematic domain and potential utility of the research prototype, not measured real-world humanitarian outcomes.*

## 11. Success Criteria
- **Phase 1 Success Criteria:**
  1. Acquisition and cryptographic verification (SHA-256) of all three official raw dataset splits from the primary source repository.
  2. Complete, transparent audit of row counts, target label distributions, missing values, and within/cross-split duplicates or label conflicts.
  3. Deterministic generation of `train.csv`, `validation.csv`, and `test.csv` in `data/processed/` with zero cross-split data leakage.
  4. 100% pass rate on automated data pipeline integrity tests (`tests/test_data_pipeline.py`).
- **Overall Project Success Criteria (Phases 2–3):**
  1. Empirical determination of whether candidate model `M1` outperforms baselines `B0` and `B1` on Macro F1, Micro F1, and `request` $F_2$ score on the held-out test split.
  2. Delivery of a functional, accessible HTML5/CSS3/JS + FastAPI screening interface that faithfully surfaces request intent, need topics, and uncertainty flags for human review.


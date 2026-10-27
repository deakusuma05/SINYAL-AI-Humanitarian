# SINYAL Phase 3 — System Architecture & Full-Stack Integration Specification

**Project:** SINYAL — AI-Powered Humanitarian Request Triage  
**Competition:** IRDSC 2026 — Category D2 | **Team:** *Bismillah Bisa*  
**Version:** 3.0.0  

---

## 1. Architectural Overview

SINYAL Phase 3 delivers a local, self-contained, full-stack decision-support web application that integrates the frozen Phase 2 supervised multi-label NLP pipeline (`M1_logreg_C2.0_cwNone`) with a FastAPI REST backend and a semantic HTML5, CSS3, and Vanilla JavaScript (ES6+) command-center interface powered by Chart.js.

```
+-----------------------------------------------------------------------------------+
|                   CLIENT BROWSER (HTML5 + CSS3 + Vanilla JS ES6+)                 |
|                                                                                   |
|  +-----------------+  +--------------------+  +---------------+  +-------------+  |
|  | Module 1:       |  | Module 2 & 3:      |  | Module 4:     |  | Module 5-6: |  |
|  | Overview &      |  | Message Analyzer & |  | Human Review  |  | Evaluation  |  |
|  | Health Status   |  | 5 Category Cards   |  | Queue (Demo)  |  | & Ethics    |  |
|  +--------+--------+  +---------+----------+  +-------+-------+  +------+------+  |
|           |                     |                     |                 |         |
|           +---------------------+----------+----------+-----------------+         |
|                                            |                                      |
|                                 web/js/api.js (Fetch)                             |
+--------------------------------------------|--------------------------------------+
                                             | HTTP / JSON (127.0.0.1:8000)
+--------------------------------------------v--------------------------------------+
|                        FASTAPI BACKEND (src/api/main.py)                          |
|                                                                                   |
|  GET /               -> Serves web/index.html + /css/* + /js/*                    |
|  GET /api/health     -> Service & artifact readiness check                        |
|  GET /api/metadata   -> Sanitized model metadata & frozen thresholds              |
|  POST /api/predict   -> Single-message inference (Pydantic validated, <=10k chr)  |
|  POST /api/batch     -> Batch inference (Pydantic validated, 1..50 messages)      |
|  GET /api/evaluation -> Frozen Phase 2 held-out test & validation metrics         |
+--------------------------------------------|--------------------------------------+
                                             |
+--------------------------------------------v--------------------------------------+
|                    SERVICE & INFERENCE LAYER (src/api/services.py)                |
|                         + (src/modeling/inference.py)                             |
|                         + (src/modeling/review_policy.py)                         |
|                                                                                   |
|  1. Input validation & finite probability bounds verification [0.0, 1.0]          |
|  2. Cached artifact bundle loader (`load_inference_bundle`)                       |
|  3. Multi-label probability extraction (`extract_positive_probabilities`)         |
|  4. Frozen threshold comparison & uncertainty band evaluation (±0.08)             |
|  5. Human-readable review routing translation                                     |
+--------------------------------------------|--------------------------------------+
                                             |
+--------------------------------------------v--------------------------------------+
|                FROZEN PHASE 2 ARTIFACTS (models/ & reports/phase2/)               |
|                                                                                   |
|  * models/sinyal_model.joblib          (TF-IDF + MultiOutput LogisticRegression)  |
|  * models/model_manifest.json          (Version 2.0.0, SHA-256 checksums)         |
|  * models/decision_policy.json         (Frozen thresholds τ* & ±0.08 bands)       |
|  * reports/phase2/final_test_metrics.json & final_evaluation_lock.json            |
+-----------------------------------------------------------------------------------+
```

---

## 2. Model Loading & Artifact Integrity Flow

1. **Single-Source Artifact Loading:**
   - `src/modeling/inference.py` loads `models/sinyal_model.joblib`, `models/model_manifest.json`, and `models/decision_policy.json` via `load_inference_bundle()`.
   - In-memory caching (`_CACHED_BUNDLE`) ensures the serialized `scikit-learn` pipeline is loaded once per process and reused across requests with sub-10ms inference latency.
2. **Canonical Label Contract Verification:**
   - Upon loading, `load_inference_bundle()` verifies that both `model_manifest.json` and `decision_policy.json` declare the exact 5-label canonical order:
     1. `request` ($\tau^* = 0.19$)
     2. `water` ($\tau^* = 0.22$)
     3. `food` ($\tau^* = 0.26$)
     4. `shelter` ($\tau^* = 0.32$)
     5. `medical_help` ($\tau^* = 0.20$)
3. **Zero Silent Retraining or Mutation:**
   - The FastAPI backend has read-only interaction with `models/` and `reports/phase2/`. It never retrains the model, never recalculates held-out test metrics on the fly, and never alters the validation-frozen thresholds.

---

## 3. End-to-End Inference & Review Policy Flow

When a user submits a message via `POST /api/predict` or `POST /api/batch`:

1. **Schema Validation (`src/api/schemas.py`):**
   - Rejects non-string inputs, empty/whitespace-only strings, messages exceeding `10,000` characters, and batches exceeding `50` messages with a structured HTTP `422` JSON response.
   - `ConfigDict(extra="forbid")` rejects unexpected fields.
2. **Real Scikit-Learn Inference (`src/modeling/inference.py`):**
   - Transforms text using the fitted `TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)`.
   - Computes raw per-label positive probabilities (`predict_proba`) from `MultiOutputClassifier(LogisticRegression(C=2.0, class_weight=None, random_state=42))`.
3. **Finite Probability Verification (`src/api/services.py`):**
   - Verifies that every returned probability and threshold is a finite floating-point number within $[0.0, 1.0]$. Any `NaN`, `Inf`, or out-of-bounds value raises `InvalidModelOutputError` (HTTP `500` structured JSON).
4. **Uncertainty & Human Review Routing (`src/modeling/review_policy.py`):**
   - Evaluates each label score $p_l$ against its frozen threshold $\tau_l^*$ and uncertainty band $[\tau_l^* - 0.08, \tau_l^* + 0.08]$.
   - Assigns one of three human-readable routing statuses:
     - **`Human verification recommended`** (`request_prediction == 1`): All predicted aid requests require human verification before any operational action.
     - **`Borderline prediction — manual review`** (`request_prediction == 0` and `human_review_recommended == True`): Triggered when `request` or any need topic falls within $\pm 0.08$ of its threshold, or when a need topic is predicted positive alongside a moderate request score ($p_{\text{request}} \ge 0.095$).
     - **`Routine monitoring`** (`human_review_recommended == False`): Retained in the queue for routine situational awareness; never automatically deleted.

---

## 4. Security, Privacy & Scientific Boundaries

- **XSS Prevention:** All frontend DOM updates in `web/js/app.js` use `document.createElement()` and `element.textContent`. `innerHTML` is never used to render user-submitted messages or API text fields.
- **Path & Stack-Trace Hiding:** `GET /api/metadata`, `GET /api/health`, and custom exception handlers strip internal filesystem paths and Python tracebacks.
- **No Unconsented Persistence:** User-submitted messages in `POST /api/predict` and `POST /api/batch` are processed strictly in memory and are not written to disk or external logs.
- **CORS Restriction:** Restricted to local demonstration origins (`http://127.0.0.1:8000` and `http://localhost:8000`).


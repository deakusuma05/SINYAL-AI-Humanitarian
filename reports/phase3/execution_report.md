# SINYAL Phase 3 — Full-Stack Integration, FastAPI Backend, Dashboard & End-to-End Verification Report

**Project:** SINYAL — AI-Powered Humanitarian Request Triage  
**Competition:** IRDSC 2026 — Category D2  
**Team:** *Bismillah Bisa*  
**Workspace:** `D:\SEMESTER 7\LOMBA IRDSC\SINYAL`  
**Phase Status:** **COMPLETED — PASS (43 / 43 Automated Tests Passing)**  

---

## 1. Executive Summary

Phase 3 completes the end-to-end delivery of **SINYAL**, transforming the frozen Phase 2 supervised multi-label NLP pipeline (`M1_logreg_C2.0_cwNone`) into a working, verified local humanitarian decision-support web application.

- **Backend (`src/api/`):** Built with **Python 3.12, FastAPI (`0.162.0`), Uvicorn (`0.54.0`), and Pydantic v2 (`2.14.1`)**. Exposes 5 REST API endpoints (`GET /api/health`, `GET /api/metadata`, `POST /api/predict`, `POST /api/batch`, `GET /api/evaluation`) plus static frontend serving (`GET /`).
- **Real Model Integration:** All single and batch predictions invoke `predict_messages()` in `src/modeling/inference.py` using the verified local serialized artifact (`models/sinyal_model.joblib`, SHA-256: `0589ce46b0ed26e5aa229b1a1d07c47d9c0f396af58b615e532777bdf12b3142`) and frozen validation thresholds (`models/decision_policy.json`, SHA-256: `48ebd3d04cc62632bd04bc3b2f8aff09d684291f553bb9f3929c389420e1b68d`). Zero keyword fallbacks, zero hardcoded scores, and zero silent retraining were introduced.
- **Frontend (`web/`):** Built with semantic **HTML5, CSS3, Vanilla JavaScript (ES6+), and Chart.js** (strictly zero Streamlit/Gradio/React). Implements all 6 required command-center modules with XSS-safe DOM rendering (`textContent` / `createElement` only), responsive breakpoints (`1440px`, `1024px`, `768px`, `375px`), and explicit humanitarian safety governance.

---

## 2. Verified Phase 1 & Phase 2 Artifact Integrity

| Artifact | Path | Verified Count / Value | SHA-256 Checksum |
|---|---|---|---|
| Processed Training Split | `data/processed/train.csv` | `19,032` rows | `13d0c0336f5ed8bc89c19ee3529075e5638f7aaf1976f1aaeefd0a3a75ecb02b` |
| Processed Validation Split | `data/processed/validation.csv` | `2,329` rows | `e09d9842d17414e08946739d93853db1687a873a08a25210ea65bb5aec7b722e` |
| Cleaned Held-Out Test Split | `data/processed/test.csv` | `2,387` rows | `d0f7265d67753a27a40a0fcc6caab8f88849b352427a71929ac692b37b06d75a` |
| Serialized Model Pipeline | `models/sinyal_model.joblib` | `M1_logreg_C2.0_cwNone` | `0589ce46b0ed26e5aa229b1a1d07c47d9c0f396af58b615e532777bdf12b3142` |
| Model Manifest | `models/model_manifest.json` | Version `2.0.0` | `b9a5bdcdfe6c7364ff22758a589b066e300da38bf0d4e6ab0660b80dadb84164` |
| Frozen Decision Policy | `models/decision_policy.json` | $\tau^*$: `0.19, 0.22, 0.26, 0.32, 0.20` | `48ebd3d04cc62632bd04bc3b2f8aff09d684291f553bb9f3929c389420e1b68d` |
| Locked Test Evaluation | `reports/phase2/final_test_metrics.json` | Macro F1 `0.716627`, Req F2 `0.778092` | `c5b50e174d7a464fc3c174d5f15c5015c81780846410903d92c84fe5bf92bd81` |

---

## 3. FastAPI Endpoints & Validation Contract

| Method & Route | Request Schema | Response Schema | Key Validation & Security Controls |
|---|---|---|---|
| `GET /` | None | `FileResponse` (`text/html`) | Serves `web/index.html` command-center interface. |
| `GET /api/health` | None | `HealthResponse` | Checks API liveness and `sinyal_model.joblib` readiness (`application_mode="research_demo"`). |
| `GET /api/metadata` | None | `MetadataResponse` | Returns model ID, frozen thresholds $\tau^*$, $\pm 0.08$ uncertainty bands, and 8 limitations; strips all local filesystem paths. |
| `POST /api/predict` | `PredictRequest` | `PredictResponse` | Rejects empty/whitespace strings, non-strings, and messages > `10,000` chars (`422`); verifies finite probabilities in $[0, 1]$. |
| `POST /api/batch` | `BatchPredictRequest` | `BatchPredictResponse` | Enforces 1 to `50` messages per request (`422` on >50); preserves input ordering and retains all unflagged messages. |
| `GET /api/evaluation` | None | `EvaluationResponse` | Serves locked Phase 2 held-out test metrics and validation comparison rows without recomputation. |

---

## 4. Dashboard Modules & Scientific Visualizations

1. **Module 1 — Overview:** Displays the 5-step human-in-the-loop triage pipeline, live API/model health status pill, frozen thresholds $\tau^*$, dataset split counts, and strict non-claims.
2. **Module 2 — Message Analyzer:** Provides a validated input form (up to `10,000` chars), 4 clearly labeled *"Illustrative Example Inputs — Not Held-Out Evaluation Samples"*, routing recommendation banner (`Human verification recommended`, `Borderline prediction — manual review`, or `Routine monitoring`), and safe `textContent` rendering.
3. **Module 3 — Multi-Label Results:** Displays 5 cards in canonical order (`Request`, `Water`, `Food`, `Shelter`, `Medical Help`), showing `Model-estimated score`, frozen threshold $\tau^*$, and visual $\pm 0.08$ uncertainty band meter.
4. **Module 4 — Human Review Queue:** Supports batch screening (`POST /api/batch`, up to 50 messages), includes a 1-click 6-message disaster batch loader, retains 100% of unflagged messages in `Routine Monitoring`, and provides local volunteer actions (**Mark reviewed**, **Mark needs follow-up**) and status filters.
5. **Module 5 — Model Evaluation:** Displays 6 KPI cards and 3 interactive **Chart.js** charts (with offline canvas fallback) backed by `GET /api/evaluation`:
   - **Chart 1:** Per-Label Held-Out Test F1, Precision, and Recall (`request: 0.7404`, `water: 0.8104`, `food: 0.8129`, `shelter: 0.7277`, `medical_help: 0.4917`).
   - **Chart 2:** Frozen Thresholds ($\tau^*$) vs. Default ($\tau=0.50$) Aggregate Held-Out Test Scores (`Macro F1: 0.7166 vs 0.6113`, `Request F2: 0.7781 vs 0.6284`).
   - **Chart 3:** Source Genre Robustness Comparison (`direct` Request $F_2 = 0.8492$, Recall $89.07\%$ vs. `news` Request $F_2 = 0.1880$, Recall $17.54\%$).
6. **Module 6 — Limitations & Safety Governance:** Prominently discloses all 8 mandatory limitations (English/translated-English scope, Indonesian not validated, historical 2010–2012 corpus, lower `medical_help` F1, `direct` vs `news` disparity, `56` unflagged test requests, mandatory human verification, and non-dispatch status).

---

## 5. Automated Test Suite Results (`43 / 43 PASSED`)

- `tests/test_data_pipeline.py` (Phase 1): **10 / 10 PASSED**
- `tests/test_model_pipeline.py` (Phase 2): **14 / 14 PASSED**
- `tests/test_api.py` (Phase 3 Backend Tests 1–12): **12 / 12 PASSED**
- `tests/test_phase3_integration.py` (Phase 3 Integration Tests 13–19): **7 / 7 PASSED**


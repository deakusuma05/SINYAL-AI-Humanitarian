# SINYAL Phase 3 — Competition Demo & Jury Defense Guide

**Project:** SINYAL — AI-Powered Humanitarian Request Triage  
**Event:** IRDSC 2026 — Category D2  
**Team:** *Bismillah Bisa*  

---

## 1. Quick Pre-Demo Startup Checklist (Windows PowerShell)

1. Open PowerShell in `D:\SEMESTER 7\LOMBA IRDSC\SINYAL`.
2. Start the FastAPI + Uvicorn server:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
3. Open your browser at **`http://127.0.0.1:8000`**.
4. Confirm the top-right status pill turns green: **`API & Model Ready (M1_logreg_C2.0_cwNone)`**.

---

## 2. Reliable 90-Second Live Demonstration Script

- **0:00 – 0:15 | Command-Center Orientation (Module 1):**
  - Point to the header badge: *"Research Demonstration — Not for Emergency Dispatch"* and the green health pill (`M1_logreg_C2.0_cwNone`).
  - Highlight the 5-stage workflow: `Incoming Message → Request Intent (τ=0.19) → Need-Topic Screening → Uncertainty Policy (±0.08) → Human Review Queue`.
- **0:15 – 0:45 | Live Single-Message Inference & Multi-Label Inspection (Modules 2 & 3):**
  - Click **Illustrative Example 1 ("Direct Multi-Need Request")**.
  - Show that FastAPI runs the real serialized `scikit-learn` pipeline and flags **`Human verification recommended`** (`request` score > `0.19`), simultaneously detecting **`WATER`** and **`FOOD`** in the 5 canonical category cards.
  - Click **Illustrative Example 3 ("Implicit / Borderline Need Mention")** or **Example 4 ("General News / Non-Request Update")** to demonstrate how SINYAL distinguishes topic mentions from explicit aid requests and routes borderline cases via the $\pm 0.08$ uncertainty band.
- **0:45 – 1:10 | Batch Triage & Zero-Deletion Human Review Queue (Module 4):**
  - Scroll to **Module 4 (Human Review Queue)** and click **"Load Illustrative 6-Message Disaster Batch"**.
  - Show how all 6 messages are retained (`Total Retained: 6`), separated into **Flagged for Human Review** vs. **Routine Monitoring**, and click **"Mark reviewed"** and **"Mark needs follow-up"** to demonstrate volunteer workflow organization.
- **1:10 – 1:30 | Frozen Held-Out Test Evidence & Honest Limitations (Modules 5 & 6):**
  - Scroll to **Module 5** and highlight the locked single-shot held-out test results (`Macro F1 = 0.7166`, `Request F2 = 0.7781`, `88.28%` of true test requests captured in the review queue by flagging `33.93%` of messages) and **Chart 3** (`Direct` Request $F_2 = 0.8492$ vs. `News` Request $F_2 = 0.1880$).

---

## 3. 7-Minute Competition Presentation Narrative

### Minute 0:00 – 1:00 | Problem & Humanitarian Context
- During sudden-onset disasters, digital volunteer networks receive thousands of unstructured messages mixing urgent cries for water, food, shelter, and medical help with general news and commentary.
- **Research Question:** *Can a lightweight, reproducible supervised NLP pipeline classify humanitarian aid requests and core need categories more effectively than baseline models while routing uncertain predictions to human reviewers?*
- **Strict Scope:** SINYAL is a human-in-the-loop screening prototype aligned with SDG 11 and SDG 13 (framed strictly as *potential decision-support impact*), not an autonomous emergency dispatch system.

### Minute 1:00 – 2:15 | Phase 1: Data Governance & Deterministic Preparation
- Built on the open-source Munro Disaster Response Messages corpus (`1657e1c4e11acbfa56241d2b5cd627adbd989ca2`).
- Remediated exact normalized cross-split leakage, removed 10 duplicate test IDs, and redacted 1 privacy-sensitive validation record containing personal contact details.
- Locked SHA-256 checksums across `train.csv` (`19,032` rows), `validation.csv` (`2,329` rows), and the cleaned held-out `test.csv` (`2,387` rows).

### Minute 2:15 – 4:00 | Phase 2: Modeling, Threshold Optimization & Uncertainty Policy
- Compared 12 model candidates across Dummy (`B0`), Multinomial Naive Bayes (`B1`), and Logistic Regression (`M1` + sigmoid calibration) strictly on the validation split.
- Selected **`M1_logreg_C2.0_cwNone`** (TF-IDF 1–2 n-grams, 30k features, sublinear TF + Logistic Regression $C=2.0$).
- Because disaster labels are severely imbalanced (`4.5%` to `20.0%` positive), default $\tau=0.50$ thresholds suffer from high false-negative rates. We tuned `request` on validation using **$F_2$** ($\tau^* = 0.19$) to weight recall twice as high as precision, and need topics using $F_1$ ($\tau^* \in \{0.22, 0.26, 0.32, 0.20\}$).
- Added an uncertainty-aware review policy ($\pm 0.08$ margin + implicit request check) that never discards unflagged messages.

### Minute 4:00 – 5:30 | Phase 2 Single-Shot Held-Out Test Findings
- On the cleaned held-out test set (`2,387` rows):
  - **Macro F1:** Improved from `0.6113` (default 0.50) to **`0.7166`** (+0.1053).
  - **Micro F1:** Improved from `0.6489` to **`0.7261`**.
  - **Request $F_2$:** Improved from `0.6284` to **`0.7781`** (Request Recall = `80.54%`, Precision = `68.51%`).
  - **Human Review Queue Coverage:** Flagging `33.93%` (`810 / 2,387`) of test messages captured **`88.28%` (`422 / 478`)** of all true aid requests—rescuing `37` additional true requests that fell below the binary threshold.

### Minute 5:30 – 7:00 | Phase 3 Web Platform & Scientific Limitations
- Demonstrated live via our FastAPI + Vanilla JS + Chart.js command center (zero Streamlit, zero fake predictions).
- **Scientific Honesty:** We openly report three critical limitations:
  1. `medical_help` has lower held-out test $F_1 = 0.4917$ due to heterogeneous clinical vocabulary.
  2. Subgroup analysis reveals strong performance on `direct` citizen messages (`Request F2 = 0.8492`, Recall `89.07%`) but low recall on third-person `news` reports (`Request F2 = 0.1880`).
  3. `56` of `478` true requests remained unflagged on the held-out test set, which is why SINYAL retains all unflagged messages in routine monitoring and never deletes data.

---

## 4. Five Likely Jury Questions & Rigorous Answers

1. **Q1: Why did you choose TF-IDF + Logistic Regression instead of a large language model (LLM) or Transformer?**
   - **Answer:** Under disaster field constraints and a strict 2–3 hour reproducible engineering sprint, TF-IDF + Logistic Regression (`M1_logreg_C2.0_cwNone`) trains in `3.92` seconds on CPU, requires <5 MB of disk storage, runs inference in <5 ms per message without GPU or internet access, and provides transparent linear feature weights and deterministic probability outputs that can be audited against validation-selected thresholds.

2. **Q2: Why did you optimize the `request` label using $F_2$ instead of $F_1$ or Accuracy?**
   - **Answer:** Because `request` is only ~18–20% positive, a model predicting all zeros achieves ~80% accuracy while failing every disaster victim. In humanitarian screening, a false negative (missing a real cry for help) is much costlier than a false positive (asking a human volunteer to review a non-request). $F_2$ weights recall twice as heavily as precision, raising held-out `request` recall from `58.37%` (at $\tau=0.50$) to `80.54%` (at $\tau^*=0.19$), and to `88.28%` when combined with the $\pm 0.08$ uncertainty review policy.

3. **Q3: Why is `medical_help` performance ($F_1 = 0.4917$) noticeably lower than `food` ($0.8129$) or `water` ($0.8104$)?**
   - **Answer:** While `water` and `food` rely on concentrated lexical terms (*water, thirsty, rice, food, starving*), `medical_help` spans hundreds of distinct symptoms, injuries, chronic illnesses, clinic logistics, and public-health reports. Bag-of-words n-grams face higher false positives and false negatives when medical terminology appears in general news reports. We highlight this transparently in Module 5 and Module 6.

4. **Q4: What happens to the 56 true requests on the test set that were not flagged for priority review?**
   - **Answer:** SINYAL enforces a strict **zero-deletion policy**. Unflagged messages are routed to the `Routine Monitoring` stream (`96.45%` negative predictive accuracy on test) rather than being discarded. Volunteers can filter and inspect the routine stream once high-priority flagged items (`33.93%` of volume, capturing `88.28%` of true requests) are verified.

5. **Q5: Can SINYAL be deployed directly for Indonesian disaster messages (e.g., BNPB or local WhatsApp groups) today?**
   - **Answer:** Not without a validated translation layer or fine-tuning on labeled Indonesian disaster corpora. Our benchmark dataset consists of English messages and English translations from historical 2010–2012 crises. While the pipeline handles Unicode/Indonesian strings without crashing, its lexical vocabulary is English, and we explicitly disclose this scope boundary in both the UI and API metadata.

---

## 5. Transparent Disclosure of AI Tools Used During Development

In accordance with IRDSC 2026 scientific transparency guidelines:
- **Development Assistant:** Google Antigravity (AI pair-programming assistant) was used during software engineering to scaffold Python modules, FastAPI schemas, HTML/CSS/JS UI components, and `pytest` test suites.
- **Verification & Data Integrity:** All dataset preparation, SHA-256 checksum verification, scikit-learn model training, validation threshold selection, and held-out test evaluations were executed deterministically by local Python scripts (`src/prepare_data.py`, `src/modeling/train_validate.py`, `src/modeling/evaluate_test.py`) inside the local `.venv` environment. No evaluation metrics or model predictions were generated or altered by an LLM.


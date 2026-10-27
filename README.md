# SINYAL — AI-Powered Humanitarian Request Triage

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/deakusuma05/SINYAL-AI-Humanitarian)

**SINYAL** is a lightweight, CPU-friendly supervised natural language processing (NLP) decision-support prototype designed to help humanitarian volunteers screen and organize incoming disaster-response communications.

---

## 1. What SINYAL Solves

During disasters and humanitarian crises, incoming message streams contain a mixture of urgent requests for survival necessities (drinking water, food, shelter, medical assistance), general news reports, offers of assistance, and off-topic messages. Manually reviewing thousands of unfiltered messages increases volunteer workload and delays information routing.

SINYAL structures incoming messages through a human-centered screening pipeline:

$$\text{Message} \rightarrow \text{Request Intent Prediction} \rightarrow \text{Need-Topic Predictions} \rightarrow \text{Uncertainty Indication} \rightarrow \text{Human Review Queue}$$

---

## 2. Intended Users

- **Humanitarian aid screening volunteers** and digital disaster-response coordinators who review incoming text queues.
- **Humanitarian technology researchers** evaluating lightweight, reproducible text classification workflows on official benchmark splits.

---

## 3. Explicit Non-Goals & Safety Boundaries

SINYAL is an **information-organization and decision-support tool**, **NOT** an autonomous emergency dispatch system. The system will **not**:
- Diagnose medical conditions or assess clinical urgency.
- Verify or guarantee that a request is authentic.
- Automatically deploy volunteers, ambulances, or physical supplies.
- Decide who deserves aid or rank human lives.
- Replace trained humanitarian professionals or human oversight.

---

## 4. Dataset Source & Provenance

- **Primary Repository:** [rmunro/disaster_response_messages](https://github.com/rmunro/disaster_response_messages) (Curated by Dr. Robert Munro; CC BY 4.0 License).
- **Source Files:**
  1. `disaster_response_training.csv` (`19,116` raw rows; SHA-256: `3fd06465488100a51f97f60b3e27eb58bf8dfda568ee92ec6e574e78c84b12dd`)
  2. `disaster_response_validation.csv` (`2,337` raw rows; SHA-256: `2c6d4424e812b068ca2486a3c1b35bc0aea445643d922657e3023ae60387bc0b`)
  3. `disaster_response_test.csv` (`2,397` raw rows; SHA-256: `566402713139ad7c910737cf5f18f3fef9ba6cc711b6aa5054c65e3fa0ecfc48`)
- Detailed provenance, licensing, privacy compliance, and annotation notes are documented in [`docs/dataset_provenance.md`](docs/dataset_provenance.md).

---

## 5. Target Labels & Semantic Separation

SINYAL models **five binary output labels** (`{0, 1}`):

```python
TARGET_LABELS = [
    "request",
    "water",
    "food",
    "shelter",
    "medical_help",
]
```

- **Task A — Request Intent (`request`):** Predicts whether a message contains a request for humanitarian assistance.
- **Task B — Need-Topic Identification (`water`, `food`, `shelter`, `medical_help`):** Multilabel prediction of which core humanitarian need categories are discussed in the message.
- **Critical Semantic Rule:** A message discussing a need topic (e.g., a news headline about water purification) is **not** automatically an aid request. Intent and topic dimensions are kept strictly independent.

---

## 6. Project Architecture & Technology Stack

SINYAL is engineered for CPU execution without heavy deep learning dependencies or Streamlit:

- **Data & Machine Learning Layer (Python):**
  - `pandas`, `numpy`, `matplotlib` (Phase 1 data audit, preparation, and EDA).
  - `scikit-learn`, `joblib` (Phase 2 TF-IDF vectorization, Dummy/Majority baseline `B0`, Multinomial Naive Bayes baseline `B1`, and Logistic Regression candidate `M1`).
- **Backend API Layer (Phase 3):**
  - **FastAPI + Uvicorn** serving REST inference endpoints backed by serialized `joblib` pipelines. All prediction logic stays strictly on the Python backend.
- **Frontend Command-Center Dashboard (Phase 3):**
  - **HTML5 + CSS3 + Vanilla JavaScript + Chart.js** (Navy, teal, white, and subtle orange humanitarian command-center aesthetic; **no Streamlit**).

### Repository Structure

```text
SINYAL/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                        # Unchanged official source CSVs (git-ignored)
│   └── processed/                  # Deterministic, leakage-free splits (git-ignored)
├── src/
│   ├── __init__.py
│   ├── audit_data.py               # Dataset download, SHA-256 verification, audit & EDA
│   └── prepare_data.py             # Reproducible data cleaning & cross-split deduplication
├── tests/
│   └── test_data_pipeline.py       # Automated 10-check pytest suite
├── docs/
│   ├── project_charter.md          # Scope, DSRM, CRISP-DM, SDG 11/13 alignment
│   ├── dataset_provenance.md       # Source provenance, checksums, privacy & limitations
│   └── experiment_protocol.md      # Pre-registered Phase 2 modeling & metric protocol
└── reports/
    ├── data_audit.md               # Comprehensive raw dataset integrity & leakage report
    ├── audit_results.json          # Machine-readable audit telemetry
    ├── label_distribution.csv      # Measured positive/negative counts & rates per split
    ├── preprocessing_summary.json  # Deterministic filtering log & processed SHA-256s
    └── figures/                    # 4 matplotlib EDA charts
```

---

## 7. Research & Engineering Methodology

1. **Design Science Research Methodology (DSRM):**
   - *Steps 1 & 2 (Implemented in Phase 1):* Problem identification, ethical safety boundaries, and quantitative solution objectives (`Macro F1`, `Micro F1`, per-label `Precision/Recall/F1`, and `request` $F_2$ score).
   - *Step 3 Foundations (Implemented in Phase 1):* Traceable data audit and deterministic preparation pipeline.
2. **CRISP-DM:**
   - *Stages 1–3 (Implemented in Phase 1):* Business Understanding (`docs/project_charter.md`), Data Understanding (`src/audit_data.py`, `reports/data_audit.md`), and Data Preparation (`src/prepare_data.py`, `reports/preprocessing_summary.json`).

---

## 8. Python Setup for Windows PowerShell

Run all commands from the project root (`d:\SEMESTER 7\LOMBA IRDSC\SINYAL`) in Windows PowerShell:

```powershell
# 1. Create a project-local virtual environment (Python 3.12)
python -m venv .venv

# 2. Verify the project interpreter
.\.venv\Scripts\python.exe --version

# 3. Install required Phase 1 dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 9. Dataset Acquisition & Raw Data Audit Execution

To download the three official CSV files (if not already in `data/raw/`), verify their SHA-256 checksums, execute the data integrity/leakage audit, and generate the EDA charts:

```powershell
.\.venv\Scripts\python.exe -m src.audit_data
```

---

## 10. Reproducible Data Preparation Execution

To run the deterministic cleaning and cross-split leakage remediation pipeline and generate `data/processed/train.csv`, `validation.csv`, and `test.csv`:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_data
```

---

## 11. Automated Test Execution

To run the 10-check automated verification suite using `pytest`:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

---

## 12. Expected Output Files (Phase 1)

| File Path | Role |
|---|---|
| `data/raw/disaster_response_training.csv` | Unmodified official raw training split (`19,116` rows). |
| `data/raw/disaster_response_validation.csv` | Unmodified official raw validation split (`2,337` rows). |
| `data/raw/disaster_response_test.csv` | Unmodified official raw test split (`2,397` rows). |
| `data/processed/train.csv` | Cleaned, deduplicated, leakage-free training split (`19,032` rows). |
| `data/processed/validation.csv` | Cleaned, deduplicated, leakage-free validation split (`2,329` rows). |
| `data/processed/test.csv` | Cleaned, deduplicated held-out test reference split (`2,387` rows). |
| `reports/data_audit.md` | Full human-readable integrity, distribution, and leakage report. |
| `reports/label_distribution.csv` | Tabular positive/negative counts and rates across all splits. |
| `reports/preprocessing_summary.json` | Exact row-removal accounting and processed file SHA-256 digests. |
| `reports/figures/*.png` | Four measured matplotlib charts of class distributions and split counts. |

---

## 13. Current Project Status & Phase 3 Full-Stack Web Application

- **Phase 1 Status:** **COMPLETED (PASS)** — Environment verified, dataset acquired and cryptographically verified, audit and EDA completed, deterministic preprocessing executed (`10/10` tests passing).
- **Phase 2 Status:** **COMPLETED (PASS)** — Trained and evaluated `B0` Dummy baselines, `B1` TF-IDF + Multinomial Naive Bayes baselines, and `M1` TF-IDF + Logistic Regression candidates on `train.csv` and `validation.csv`. Selected and froze `M1_logreg_C2.0_cwNone` with validation-tuned thresholds, evaluated probability calibration and human-review routing, executed a single locked evaluation on the cleaned held-out test set (`Macro F1 = 0.7166`, `Micro F1 = 0.7261`, `request F2 = 0.7781`), serialized `models/sinyal_model.joblib`, and verified `24/24` automated tests (`10` Phase 1 + `14` Phase 2).
- **Phase 3 Status:** **COMPLETED (PASS)** — Built the **FastAPI + Uvicorn + Pydantic** backend (`src/api/`) integrated with the real Phase 2 model (`src/modeling/inference.py`), built the **HTML5 + CSS3 + Vanilla JavaScript ES6+ + Chart.js** command-center dashboard (`web/`), verified all 6 UI modules and 5 REST endpoints, and passed all **`43/43` automated tests** (`10` Phase 1 + `14` Phase 2 + `12` Phase 3 Backend API + `7` Phase 3 End-to-End Integration).

### Phase 2 Execution Commands (Windows PowerShell)

```powershell
# 1. Train baselines & candidates, optimize thresholds on validation, and serialize frozen winner
.\.venv\Scripts\python.exe -m src.modeling.train_validate

# 2. Execute single-shot evaluation on the cleaned held-out test set (locked after 1st run)
.\.venv\Scripts\python.exe -m src.modeling.evaluate_test --confirm-frozen
```

### Phase 3 Web Application & Test Commands (Windows PowerShell)

```powershell
# 1. Run the complete automated test suite (43 tests: Phase 1 + Phase 2 + Phase 3)
.\.venv\Scripts\python.exe -m pytest -v

# 2. Start the local FastAPI + Uvicorn server
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 3. Open the SINYAL Command-Center Dashboard in your browser:
#    http://127.0.0.1:8000
#    Interactive OpenAPI Docs: http://127.0.0.1:8000/api/docs
```



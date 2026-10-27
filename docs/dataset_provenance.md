# Dataset Provenance & Governance — SINYAL

## 1. Source Identification

- **Original Dataset Name:** Disaster Response Messages
- **Primary Repository URL:** https://github.com/rmunro/disaster_response_messages
- **Source Author / Curator:** Dr. Robert Munro (with acknowledgements to Mission 4636 in Haiti, Pakreport in Pakistan, and Joan Xiao for PII scrubbing).
- **Primary Academic Citation:**
  > Robert Munro. 2012. *Processing short message communications in low-resource languages*. PhD Dissertation, Stanford University. Stanford Digital Repository (`https://purl.stanford.edu/cg721hb0673`).
- **Repository Revision / Commit SHA (Main Branch):** `1657e1c4e11acbfa56241d2b5cd627adbd989ca2` (Committed: `2021-11-12T04:38:01Z`)
- **Date Downloaded (UTC):** `2026-10-27T21:02:27Z`
- **Source Access Failures:** None (`0` errors; all three files and repository metadata were fetched and verified directly).

---

## 2. Verified Raw Files & Cryptographic Checksums

| Split | Local File Path | Source Raw URL | File Size (Bytes) | Raw Row Count | Columns | SHA-256 Checksum |
|---|---|---|---:|---:|---:|---|
| **Training** | `data/raw/disaster_response_training.csv` | `https://raw.githubusercontent.com/rmunro/disaster_response_messages/main/disaster_response_training.csv` | `5,722,662` | `19,116` | `45` | `3fd06465488100a51f97f60b3e27eb58bf8dfda568ee92ec6e574e78c84b12dd` |
| **Validation** | `data/raw/disaster_response_validation.csv` | `https://raw.githubusercontent.com/rmunro/disaster_response_messages/main/disaster_response_validation.csv` | `736,486` | `2,337` | `45` | `2c6d4424e812b068ca2486a3c1b35bc0aea445643d922657e3023ae60387bc0b` |
| **Testing** | `data/raw/disaster_response_test.csv` | `https://raw.githubusercontent.com/rmunro/disaster_response_messages/main/disaster_response_test.csv` | `716,510` | `2,397` | `45` | `566402713139ad7c910737cf5f18f3fef9ba6cc711b6aa5054c65e3fa0ecfc48` |
| **Total** | — | — | `7,175,658` | `23,850` | `45` | — |

---

## 3. Dataset Description & Composition

The dataset comprises **23,850 real disaster-related messages** collected across multiple historical emergencies and news monitoring streams:
1. **Haiti Earthquake (2010) (`event == "haiti_earthquake"`):** SMS messages sent to the Mission 4636 short-code reporting service (`train`: 7,956; `validation`: 965; `test`: 981).
2. **Pakistan Floods (2010) (`event == "pakistan_floods"`):** SMS messages sent to the Pakreport disaster reporting service (`train`: 247; `validation`: 23; `test`: 26).
3. **Hurricane Sandy, USA (2012) (`event == "usa_sandy"`):** Posts from an online disaster-response message board (`train`: 463; `validation`: 59; `test`: 62).
4. **International Disaster News Headlines (`genre == "news"`, `event == NULL`):** Curated news articles and headlines covering hundreds of global disasters (`train`: 10,450; `validation`: 1,290; `test`: 1,328).

### Note on Omitted Social Media Records
As explicitly documented in the repository `README.md`, an earlier release contained ~5,000 social media posts. Those were omitted by the dataset author in this public release because being a victim of a disaster is considered sensitive health-related information that could be searched and linked back to individuals' social media profiles without explicit consent. Consequently, `genre` in this release consists strictly of `direct` (`10,782` total raw rows) and `news` (`13,068` total raw rows).

---

## 4. Target-Label Meanings & Semantic Separation

SINYAL selects exactly five binary target columns (`{0, 1}`) from the 38 annotation categories:

| Target Label | Task Dimension | Source Definition | Semantic Rule in SINYAL |
|---|---|---|---|
| `request` | **Task A: Request Intent** | `1` if the message is a request for aid; `0` otherwise. | Indicates whether the sender is asking for assistance. |
| `water` | **Task B: Need Topic** | `1` if the message is about drinking water; `0` otherwise. | Topic indicator only; does **not** automatically mean the sender is requesting water. |
| `food` | **Task B: Need Topic** | `1` if the message is about food; `0` otherwise. | Topic indicator only; does **not** automatically mean the sender is requesting food. |
| `shelter` | **Task B: Need Topic** | `1` if the message is about shelter; `0` otherwise. | Topic indicator only; does **not** automatically mean the sender is requesting shelter. |
| `medical_help` | **Task B: Need Topic** | `1` if the message is about medical help; `0` otherwise. | Topic indicator only; does **not** automatically mean the sender is requesting medical help. |

**Mandatory Modeling Constraint:** Intent (`request`) and need topics (`water`, `food`, `shelter`, `medical_help`) are kept strictly separate as five binary targets in a multilabel formulation. No urgency, clinical severity, disaster priority, or authenticity labels exist or are fabricated.

---

## 5. License, Redistribution, and Privacy Restrictions

### Stated License
Per the repository `README.md`, the dataset is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)** (`http://creativecommons.org/licenses/by/4.0/`), requiring attribution to Robert Munro (2012).

### Explicit Privacy & Use Restrictions
The repository `README.md` specifies the following mandatory ethical and privacy rules:
1. **No Re-Identification or Data Linkage:** *"The data should not be combined with other data such that it would reveal the identities of people sending or mentioned in the messages."*
2. **Exclusion of Residual PII:** *"If any message is found to still contain personal identifying information it should be excluded from your study."*
3. **IRB Requirement for Sensitive Entity Encoding:** Encoding the data further with potentially sensitive named entities (such as personal names and specific private locations) requires IRB or equivalent independent ethics review.

### SINYAL Privacy Compliance Action
During our Phase 1 automated PII audit (`src/audit_data.py` & `src/prepare_data.py`), we scanned all splits for unredacted email and telephone patterns:
- Seven matches in `train` (5 email, 2 phone) and one match in `validation` (`id=22028`) occurred in `genre == "news"` press releases citing institutional NGO/UN domains (`wfp.org`, `ofid.org`, `cgiar.org`, `InternationalMedicalCorps.org`, etc.).
- One record in `validation` (`id=4788`) was a `genre == "direct"` message from the Haiti earthquake containing an unredacted personal email pattern. In strict compliance with the source restriction and Section 4 of the Master Prompt, `src/prepare_data.py` automatically excluded `id=4788` from `data/processed/validation.csv` and logged only a redacted audit note in `reports/preprocessing_summary.json` without copying the sensitive message text into any report.

---

## 6. Known Language & Annotation Limitations

1. **Translation Artifacts (`message` vs. `original`):**
   - For non-English messages (primarily Haitian Kreyòl, French, and Spanish in Haiti; Latin-script Urdu and Sindhi in Pakistan), `original` holds the raw source message (`8,191` rows in `train`, `987` in `validation`, `1,006` in `test`), while `message` holds the English translation.
   - As documented by Munro (2012), these translations were produced rapidly by volunteer responders during live crises and are **not** always literal translations—they occasionally include translator summaries, paraphrases, or bracketed notes.
2. **Annotation Review Asymmetry:**
   - The dataset was annotated by paid and volunteer native speakers (many from crisis-affected communities).
   - According to the repository `README.md`, **only the test set annotations were additionally reviewed by disaster-response professionals**. Consequently, minor annotation noise and label conflicts exist in the raw training split (11 duplicate message groups / 22 rows in `train` had conflicting 5-target annotations and were deterministically removed by `src/prepare_data.py`).
3. **Official Split Definitions:**
   - The repository provides fixed random splits (`disaster_response_training.csv`, `disaster_response_validation.csv`, `disaster_response_test.csv`). SINYAL preserves these exact split assignments and remedies the 10 cross-split duplicate messages by preserving `test` as the held-out reference.


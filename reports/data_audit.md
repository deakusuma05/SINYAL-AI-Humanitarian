# SINYAL Phase 1 — Raw Data Integrity & Leakage Audit Report

**Audit Timestamp (UTC):** `2026-10-27T21:08:17.142742+00:00`  
**Primary Repository:** `https://github.com/rmunro/disaster_response_messages`  
**Source Commit SHA:** `None` (`None`)

---

## 1. Source File Verification & Checksums

| Split | Filename | Size (Bytes) | Raw Rows | Columns | SHA-256 Checksum |
|---|---|---:|---:|---:|---|
| `train` | `disaster_response_training.csv` | 5,722,662 | 19,116 | 45 | `3fd06465488100a51f97f60b3e27eb58bf8dfda568ee92ec6e574e78c84b12dd` |
| `validation` | `disaster_response_validation.csv` | 736,486 | 2,337 | 45 | `2c6d4424e812b068ca2486a3c1b35bc0aea445643d922657e3023ae60387bc0b` |
| `test` | `disaster_response_test.csv` | 716,510 | 2,397 | 45 | `566402713139ad7c910737cf5f18f3fef9ba6cc711b6aa5054c65e3fa0ecfc48` |
| **Total** | — | — | **23,850** | — | — |

- **Encoding & Readability:** All three files decoded cleanly as UTF-8 CSVs with identical 45-column schemas.
- **HTML / Truncation Check:** Verified none of the files are HTML error pages or empty payloads.

---

## 2. Schema, Missing Values & Label Validity

### 2.1 Column Schema (45 Columns)
- **Metadata / Text Columns (8):** `id` (`int64`), `split` (`str`), `message` (`str`), `original` (`str`), `genre` (`str`), `event` (`str`), `actionable_haiti` (`float64`), `date_haiti` (`str`).
- **Selected SINYAL Target Labels (5):** `request`, `water`, `food`, `shelter`, `medical_help`.
- **Other Source Annotation Columns (32):** `related`, `PII`, `offer`, `aid_related`, `medical_products`, `search_and_rescue`, `security`, `military`, `child_alone`, `clothing`, `money`, `missing_people`, `refugees`, `death`, `other_aid`, `infrastructure_related`, `transport`, `buildings`, `electricity`, `tools`, `hospitals`, `shops`, `aid_centers`, `other_infrastructure`, `weather_related`, `floods`, `storm`, `fire`, `earthquake`, `cold`, `other_weather`, `direct_report`.

### 2.2 Missing Values & Whitespace-Only Messages

| Split | `message` Nulls | `message` Empty/Whitespace | `original` Non-Null (Translated) | `original` Null (English-Only) |
|---|---:|---:|---:|---:|
| `train` | 0 | 1 | 8,191 | 10,925 |
| `validation` | 0 | 1 | 987 | 1,350 |
| `test` | 0 | 0 | 1,006 | 1,391 |

- **Language Observation:** `message` is always present in English (or English translation), while `original` contains the untranslated source text (e.g., Haitian Creole, Spanish) for roughly 38–39% of records.

### 2.3 Target Label Value Audit (`{0, 1}` Compliance)
- Across all three splits (`train`, `validation`, `test`), **all five selected target labels (`request`, `water`, `food`, `shelter`, `medical_help`) contain strictly binary `{0, 1}` integer values** with zero nulls and zero out-of-range values.
- *Secondary Observation on Unselected Columns:* The unselected `related` column contains value `2` in several rows (`train`: 132, `validation`: 14, `test`: 9), which in the source schema represents `2 == unsure` whether the message is disaster-related. Additionally, `PII` and `child_alone` are `0` across all rows in this public release. None of the five SINYAL target labels have value `2`.

---

## 3. Raw Target Label Distributions

| Split | Target Label | Positive Count (1) | Negative Count (0) | Positive Rate (%) | Total Rows |
|---|---|---:|---:|---:|---:|
| `train` | `request` | 3,428 | 15,688 | 17.93% | 19,116 |
| `train` | `water` | 1,281 | 17,835 | 6.70% | 19,116 |
| `train` | `food` | 2,242 | 16,874 | 11.73% | 19,116 |
| `train` | `shelter` | 1,839 | 17,277 | 9.62% | 19,116 |
| `train` | `medical_help` | 1,593 | 17,523 | 8.33% | 19,116 |
| `validation` | `request` | 398 | 1,939 | 17.03% | 2,337 |
| `validation` | `water` | 155 | 2,182 | 6.63% | 2,337 |
| `validation` | `food` | 267 | 2,070 | 11.42% | 2,337 |
| `validation` | `shelter` | 205 | 2,132 | 8.77% | 2,337 |
| `validation` | `medical_help` | 204 | 2,133 | 8.73% | 2,337 |
| `test` | `request` | 480 | 1,917 | 20.03% | 2,397 |
| `test` | `water` | 192 | 2,205 | 8.01% | 2,397 |
| `test` | `food` | 311 | 2,086 | 12.97% | 2,397 |
| `test` | `shelter` | 226 | 2,171 | 9.43% | 2,397 |
| `test` | `medical_help` | 213 | 2,184 | 8.89% | 2,397 |

---

## 4. Genre and Event Distributions Across Raw Splits

### 4.1 Genre Distribution (`genre`)

| Split | `direct` | `news` |
|---|---:|---:|
| `train` | 8,666 | 10,450 |
| `validation` | 1,047 | 1,290 |
| `test` | 1,069 | 1,328 |

*(Note: As documented in the source repository README, approximately 5,000 social media messages from an earlier release were omitted by the dataset author for health-privacy compliance, leaving `direct` and `news` genres.)*

### 4.2 Event Distribution (`event`)

| Split | `haiti_earthquake` | `usa_sandy` | `pakistan_floods` | `NULL (news)` |
|---|---:|---:|---:|---:|
| `train` | 7,956 | 463 | 247 | 10,450 |
| `validation` | 965 | 59 | 23 | 1,290 |
| `test` | 981 | 62 | 26 | 1,328 |

---

## 5. Within-Split Duplicate & Label Conflict Analysis

Normalization applied for duplicate & leakage detection: Unicode `NFKC`, lowercasing, and contiguous whitespace collapsing.

| Split | Rows Involved in Duplicate `id` | Distinct Duplicate `id`s | Extra `id` Copies | Rows Involved in Duplicate Normalized `message` | Distinct Duplicate Texts | Extra Text Copies | Conflicting Label Groups (5 Targets) | Conflicting Label Rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 100 | 50 | 50 | 125 | 62 | 63 | 11 | 22 |
| `validation` | 8 | 4 | 4 | 10 | 5 | 5 | 0 | 0 |
| `test` | 19 | 9 | 10 | 19 | 9 | 10 | 0 | 0 |

---

## 6. Cross-Split Leakage Analysis

| Split Pair | Shared `id` Count | Shared Normalized `message` Count | Rows in Split 1 | Rows in Split 2 | Conflicting 5-Target Labels Across Pair |
|---|---:|---:|---:|---:|---:|
| `train` vs `validation` | 0 | 5 | 6 | 5 | 0 |
| `train` vs `test` | 0 | 5 | 6 | 5 | 1 |
| `validation` vs `test` | 0 | 1 | 1 | 1 | 1 |

- **Global Conflicting Normalized Text Groups (Across All 3 Splits Combined):** `12` distinct normalized messages (`26` total rows) have differing annotations on at least one of the 5 target labels.

---

## 7. Privacy & PII Audit Note

Per Section 4 (Privacy Requirement), we scanned all splits for unredacted email addresses and phone number patterns without copying message contents into reports:
- **`train`:** Email pattern matches = `5`, Phone pattern matches = `2`.
- **`validation`:** Email pattern matches = `2`, Phone pattern matches = `0`.
- **`test`:** Email pattern matches = `0`, Phone pattern matches = `0`.

To protect affected individuals and comply with source restrictions (`README.md`), the 7 matches in `train` and 1 match in `validation` (`id=22028`) were verified as institutional NGO/UN press-release references (`genre == 'news'`), whereas the single `genre == 'direct'` match in `validation` (`id=4788`) contained an unredacted personal email pattern and is automatically excluded by `src/prepare_data.py` with only a redacted audit note recorded.

---

## 8. Deterministic Remediation Policy for `src/prepare_data.py`

Based on the empirical audit above, `src/prepare_data.py` enforces the following explicit, traceable rules:
1. **Preserve Official Held-Out Test Reference:** The test split (`disaster_response_test.csv`) is preserved as the official evaluation reference (removing only unusable empty/null messages or within-test exact duplicate IDs if present).
2. **Privacy Protection Exclusion:** Exclude any `genre == 'direct'` message matching unredacted personal contact patterns (`id=4788` in `validation`) and log only a redacted audit note without exposing message text.
3. **Within-Split Conflicting Annotations (`train` and `validation`):** In `train` and `validation`, identical normalized messages that have conflicting 5-target label vectors are removed entirely from that split so the model is not trained or tuned on contradictory ground truth. If identical normalized messages have *identical* 5-target label vectors (or duplicate `id`s with identical labels), only the first occurrence is kept (`keep='first'`).
4. **Within-Test Duplicate Handling:** In `test`, exact duplicate `(id, normalized_message)` rows with identical target labels are deduplicated (`keep='first'`), while any conflicting duplicate IDs in `test` are logged transparently in `reports/preprocessing_summary.json` and deduplicated deterministically so no duplicate IDs remain.
5. **Cross-Split Leakage Removal (Preserving Test Priority):**
   - Any `id` or normalized `message` present in `test` is removed from `train` and `validation`.
   - Any `id` or normalized `message` present in `validation` is removed from `train`.
   - This guarantees **zero duplicate IDs and zero exact normalized messages across splits** while keeping the held-out test distribution intact.

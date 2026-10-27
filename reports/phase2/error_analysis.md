# SINYAL Phase 2 — Validation Error Analysis, Subgroup Robustness & Interpretability Report

**Selected Model:** `M1_logreg_C2.0_cwNone` (`TfidfVectorizer` + `MultiOutputClassifier(LogisticRegression(C=2.0, class_weight=None))`)  
**Frozen Validation Thresholds:** `request = 0.19`, `water = 0.22`, `food = 0.26`, `shelter = 0.32`, `medical_help = 0.20`  
**Evaluated Validation Split:** `data/processed/validation.csv` (`2,329` rows)

---

## 1. False Negative Aid Requests (`request` True = 1, Predicted = 0)

- **Aggregate Count:** `68` False Negatives out of `397` true validation requests (`FN Rate = 17.13%`, `Recall = 82.87%`).
- **Breakdown by Genre:** `47` in `news` (`69.1%` of FNs) and `21` in `direct` (`30.9%` of FNs).
- **Root Cause Analysis:**
  1. **Indirect / Journalistic Appeals in `news`:** Out of `56` positive `request` rows in the `news` genre, `47` are missed because journalistic sentences describe ongoing crises or quote officials in the third person without first-person request markers (*"we need"*, *"please help"*).
  2. **Question-Form / Informational Inquiries in `direct`:** Direct SMS messages phrased as questions (e.g., asking about Red Cross recruitment or diplomatic visits) that were annotated as `request = 1` in the raw dataset lack survival-need vocabulary and receive low request scores.
  3. **Mitigation via Human-Review Policy:** When the validation uncertainty policy is applied, `28` of the `68` false negatives (`41.2%` of missed requests) are rescued into the **Human Review Queue** (`request` queue capture recall rises from `82.87%` to `89.92%`, leaving only `40` missed requests in the unflagged stream).

### Representative Redacted Validation False Negatives (`request`)

| Message ID | Genre | Translated (`original` present) | Model `request` Score | True Need Topics | Redacted Excerpt |
|---:|---|---|---:|---|---|
| `16498` | `news` | No | `0.0166` | `[]` | *"A general alert has been sounded across the state with a central control room set up to monitor the flood situ..."* |
| `15918` | `news` | No | `0.0262` | `[]` | *"From the weather warnings announced by the Ghana Meteorological Agency, we can expect more rain over an extend..."* |
| `10548` | `direct` | Yes | `0.0272` | `['medical_help']` | *"Does Haitian Red Cross recruiting drivers nowadays?"* |
| `27748` | `news` | No | `0.0275` | `['medical_help']` | *"Is an effective vaccine already available against the recently identified Influenza A(H1N1) virus?"* |
| `6968` | `direct` | Yes | `0.0280` | `[]` | *"what is the day the french president come in Haiti"* |
| `30218` | `news` | No | `0.0295` | `['medical_help']` | *"With more than 7,500 people infected already and the virus spreading rapidly, there is virtually an endless d..."* |

---

## 2. False Positive Aid Requests (`request` True = 0, Predicted = 1)

- **Aggregate Count:** `234` False Positives out of `1,932` true validation non-requests (`FP Rate = 12.11%`, `Precision = 58.44%`).
- **Breakdown by Genre:** `195` in `direct` and `39` in `news`.
- **Root Cause Analysis:**
  1. **Intentional Recall-Weighted Thresholding ($\tau_{\text{request}} = 0.19$):** Lowering the `request` threshold from `0.50` to `0.19` to maximize $F_2$ increased validation Recall from `64.48%` (`256` TP) to `82.87%` (`329` TP), while increasing False Positives from `60` to `234`.
  2. **Unverified Ground-Truth Annotation Noise in `direct` Messages:** As noted in `docs/dataset_provenance.md` (Section 6), the repository author documented that *only the test set annotations were reviewed by disaster-response professionals*. Inspecting the highest-scoring validation "false positives" reveals several direct SMS messages that explicitly plead for food, water, and shelter (e.g., `id=9888`, `id=5798`, `id=598`, `id=418`) but were labeled `request = 0` in the unreviewed validation split. We strictly preserve the ground-truth labels without retroactive relabeling, which explains why held-out test Precision on professionally reviewed annotations jumps from `58.44%` (validation) to **`68.51%`** (test)!

### Representative Redacted Validation False Positives (`request`)

| Message ID | Genre | Translated (`original` present) | Model `request` Score | True Need Topics | Redacted Excerpt |
|---:|---|---|---:|---|---|
| `9888` | `direct` | Yes | `0.9982` | `[]` | *"we need help please, it's emergency"* |
| `5798` | `direct` | Yes | `0.9963` | `[]` | *"WE NEED FOOD CAN YOU HELP US?"* |
| `598` | `direct` | Yes | `0.9919` | `[]` | *"We need food, water, money, energy, please around Santo, 2nd Street Alexie No. 4"* |
| `2828` | `direct` | Yes | `0.9872` | `[]` | *"My house collapsed, i need food and water, i don't have nay clothes, my adress is blvd 15 octobre innovat stre..."* |
| `418` | `direct` | Yes | `0.9868` | `['water', 'food']` | *"We need water and food in Akazya."* |
| `4308` | `direct` | Yes | `0.9609` | `[]` | *"We need sheltery we're in the street, sleeping outside. Please."* |

---

## 3. Multi-Label Need-Topic Co-occurrence Failures

- **Validation Messages with $\ge 2$ True Need Topics:** `182` messages.
- **Exact Match Across All 4 Need Topics:** `74` messages (`40.66%`).
- **Messages Where $\ge 1$ Co-occurring Need Was Missed:** `94` messages (`51.65%`).
- **Pattern Observed:** When a message expresses an umbrella plea (e.g., *"need help of every kind"* or *"the aids do not reach this area"*) or uses colloquial expressions (*"dying of thirst"* with misspelled *"wat"* in `id=438`), the classifier reliably captures the explicitly named dominant topic (`food` or `shelter`) but misses secondary co-occurring topics (`medical_help` or `water`).

### Representative Multi-Label Partial Misses

| Message ID | Genre | True Need Topics | Predicted Need Topics | Redacted Excerpt |
|---:|---|---|---|---|
| `78` | `direct` | `['food', 'medical_help']` | `['food']` | *"How can we find help and food in fontamara 43 rue menos"* |
| `118` | `direct` | `['water', 'food', 'medical_help']` | `['water', 'food']` | *"( im taking a guess at this sorry its a mix of creole and french. sms.) message as i understand. whos close to..."* |
| `438` | `direct` | `['water', 'food']` | `['food']` | *"WE ARE DYING OF HUNGER OF THIRST SEE WAT U CAN DO THE ADDRESS IS : PETION VILLE ROUTE CHAVANNE NEAR THE IMPASS..."* |
| `538` | `direct` | `['water', 'food', 'shelter', 'medical_help']` | `['water', 'food', 'medical_help']` | *"they need help of every king, food, water, health services at Thomassin 32, 12 19 km east of Port-au-Prince. T..."* |
| `558` | `direct` | `['water', 'food']` | `['water']` | *"we do not have water we are in Santo 15 thank you"* |
| `698` | `direct` | `['water', 'food', 'shelter', 'medical_help']` | `['shelter']` | *"Please people in Carrefour, Magloire Ambroide Avenue need shelter and the aids do not reach this area, please..."* |

---

## 4. Short Messages ($\le 8$ Words) & Translation Ambiguity

- **Short Messages ($\le 8$ words) in Validation:** `198` messages, with `32` `request` misclassifications (`Error Rate = 16.16%`, primarily terse SMS fragments lacking context).
- **Translated (`original` non-null) vs. English-Only (`original` null) Comparison:**
  - **Translated Messages (`984` validation rows, mostly `direct` Haiti/Pakistan SMS):** Macro F1 = `0.7185`, `request` $F_2$ = `0.8666`. Despite translator notes (such as `id=118`), direct translated SMS messages use strong survival vocabulary (`food`, `water`, `tents`, `we need`).
  - **English-Only Messages (`1,345` validation rows, predominantly `news`):** Macro F1 = `0.5322`, `request` $F_2$ = `0.1863`. Because `request` prevalence in `news` is only `4.36%` (`56 / 1,285`), request classification on journalistic prose is substantially harder than on direct survivor SMS.

---

## 5. Subgroup Robustness Analysis by `genre` (`direct` vs. `news`)

### 5.1 Validation Split (`2,329` rows)

| Subgroup (`genre`) | Record Count | `request` Positives (%) | `request` Precision | `request` Recall | `request` F1 | `request` F2 | Macro F1 (5 Labels) | Micro F1 (5 Labels) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`direct`** | `1,044` | `341` (`32.66%`) | `0.6214` | `0.9384` | `0.7477` | `0.8515` | `0.7015` | `0.7499` |
| **`news`** | `1,285` | `56` (`4.36%`) | `0.1875` | `0.1607` | `0.1731` | `0.1654` | `0.5285` | `0.5644` |

### 5.2 Cleaned Held-Out Test Split (`2,387` rows — Exploratory Subgroup Check)

| Subgroup (`genre`) | Record Count | `request` Positives (%) | `request` Precision | `request` Recall | `request` F1 | `request` F2 | Macro F1 (5 Labels) | Micro F1 (5 Labels) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`direct`** | `1,067` | `421` (`39.46%`) | `0.7156` | `0.8907` | `0.7937` | `0.8492` | `0.7484` | `0.7973` |
| **`news`** | `1,320` | `57` (`4.32%`) | `0.2632` | `0.1754` | `0.2105` | `0.1880` | `0.5559` | `0.5840` |

*Finding:* On `direct` survivor communications (the primary operational target of volunteer triage), the model achieves **`89.07%` recall (`0.8492` $F_2$) on the held-out test set** and **`93.84%` recall (`0.8515` $F_2$) on validation**. On `news` articles, need-topic identification remains solid (`water` F1 = `0.7447`, `shelter` F1 = `0.6780` on test), but `request` F1 is low (`0.2105`) due to extreme class rarity (`4.32%`) and indirect journalistic phrasing.

---

## 6. Model Interpretability (Top Lexical Associations)

Extracted linear coefficients (`coef_`) from the selected `M1_logreg_C2.0_cwNone` estimator (note: these reflect statistical associations in TF-IDF space, **not** causal effects):

| Target Label | Top Positive Lexical Features (Weight) | Top Negative Lexical Features (Weight) |
|---|---|---|
| **`request`** | `help` (+8.55), `need` (+7.45), `food` (+6.82), `we` (+5.92), `tent` (+5.21), `aid` (+4.89), `water` (+4.88), `hungry` (+4.76), `tents` (+4.57), `us` (+3.99), `please` (+3.97) | `job` (-5.07), `work` (-2.84), `need job` (-2.35), `items` (-2.17), `the` (-2.13), `4636` (-2.05), `were` (-2.03), `year` (-1.91), `news` (-1.85), `to help` (-1.62), `can help` (-1.59) |
| **`water`** | `water` (+24.89), `food water` (+4.80), `water food` (+4.71), `and water` (+4.63), `water and` (+4.61), `thirst` (+4.22), `thirsty` (+3.83), `supplies` (+3.69), `wells` (+2.96), `need water` (+2.73), `sanitation` (+2.70) | `the` (-1.37), `gallon` (-1.28), `food please` (-1.21), `level` (-1.11), `the water` (-1.06), `water resources` (-1.00) |
| **`food`** | `food` (+24.88), `hungry` (+9.54), `hunger` (+9.27), `foods` (+6.85), `rice` (+6.11), `starving` (+5.93), `eat` (+5.00), `to eat` (+4.66), `are hungry` (+4.10), `wfp` (+3.81), `of hunger` (+3.77) | `the` (-2.41), `information` (-1.83), `clean up` (-1.35), `how` (-1.32), `to know` (-1.31), `river` (-1.20) |
| **`shelter`** | `tents` (+14.82), `shelter` (+14.11), `tent` (+11.63), `shelters` (+9.12), `blankets` (+8.35), `house` (+6.39), `homeless` (+6.13), `houses` (+6.03), `sleeping` (+5.83), `homes` (+5.80), `housing` (+5.76), `camps` (+5.09) | `information` (-2.47), `the water` (-1.45), `work` (-1.38), `job` (-1.37), `need to` (-1.35) |
| **`medical_help`** | `medical` (+8.47), `health` (+6.95), `hospital` (+5.71), `injured` (+5.49), `disease` (+5.30), `wounded` (+5.09), `sick` (+4.99), `ebola` (+4.75), `fever` (+4.64), `outbreak` (+4.59), `malaria` (+4.26), `doctors` (+3.84) | `job` (-1.94), `school` (-1.90), `information` (-1.63), `help in` (-1.51), `need food` (-1.41) |

*Key Insight:* Notice how the `request` classifier learned negative weights for offer/volunteer phrases (`to help`, `can help`), employment inquiries (`job`, `need job`, `work`), and past-tense journalistic reporting (`were`, `year`, `news`), while assigning positive weights to first-person pronouns and urgent pleas (`we`, `us`, `help`, `need`, `please`).


# Methodological, Dataset, and Operational Limitations — SINYAL Phase 2

## 1. Offline Historical Benchmark vs. Live Humanitarian Operations
All metrics reported in Phase 2 (`Macro F1`, `Micro F1`, per-label `F1`, `request` $F_2$, Average Precision, and human-review simulation rates) are measured on held-out partitions of a historical benchmark dataset (`rmunro/disaster_response_messages`).
- **No Claim of Field Validation:** High offline benchmark scores do **not** prove real-world time savings, volunteer cognitive load reduction, or improved humanitarian outcomes.
- **Not an Emergency Dispatch System:** SINYAL is strictly an information-organization prototype to help volunteers screen and prioritize message review queues. It must never be used to automatically dispatch responders, allocate scarce aid, or reject incoming pleas for help.

---

## 2. Historical Domain Shift & Crisis Specificity
- **Event Concentration:** The direct messages in the dataset originate primarily from three historical emergencies: the **2010 Haiti Earthquake** (`7,956` raw training rows), **2012 Hurricane Sandy in the USA** (`463` raw training rows), and the **2010 Pakistan Floods** (`247` raw training rows), alongside international disaster news headlines (`10,450` raw training rows).
- **Geographical & Lexical Specificity:** High-weight TF-IDF features inevitably capture crisis-specific toponyms, organizations, and infrastructural terms from 2010–2012. A model trained on these historical distributions may experience substantial performance degradation under distribution shift when applied to future disasters in different regions, climates, or communication platforms.

---

## 3. Language Scope & Translation Artifacts
- **English & Translated-English Scope Only:** All models (`B0`, `B1`, `M1`) were trained exclusively on the English `message` column. Consequently, **the model is not validated for untranslated Indonesian, Haitian Kreyòl, Spanish, French, or Urdu inputs**.
- **Volunteer Translation Noise:** Approximately 43% of training messages (`8,191` raw rows) and 42% of validation/test messages were originally written in non-English languages (such as Haitian Kreyòl or Urdu) and translated into English by volunteer responders during live crises. As documented by Munro (2012), these translations include summaries, paraphrases, and translator annotations that differ from native English syntax.

---

## 4. Genre Confounding (`direct` vs. `news`) & Missing Social Media Stream
- **Omitted Social Media Data:** Approximately 5,000 social media messages from an earlier release were excluded from the public repository by the dataset curator to protect disaster victims' health privacy. Therefore, only `direct` and `news` genres are present in the processed splits.
- **Topic vs. Intent Asymmetry Across Genres:** In `news` messages, resource terms (`water`, `food`, `shelter`, `medical_help`) frequently appear in descriptive reporting about government or NGO relief efforts where `request = 0`. Conversely, in `direct` SMS reports, resource terms predominantly co-occur with `request = 1`. Lexical classifiers can still produce false positive request predictions on news reports that quote survivors or describe urgent shortages, and false negative predictions on indirect or telegraphic SMS pleas that omit explicit request verbs (*"need"*, *"help"*, *"please"*).

---

## 5. Class Imbalance & Threshold Sensitivity
- **Minority Need Categories:** Positive prevalence in `data/processed/train.csv` is low for `water` (`6.70%`), `medical_help` (`8.33%`), `shelter` (`9.61%`), `food` (`11.73%`), and `request` (`17.90%`).
- **Precision-Recall Trade-off:** Optimizing the `request` threshold for $F_2$ (or using `class_weight="balanced"`) intentionally increases Recall to reduce missed aid requests (False Negatives) at the cost of lower Precision (more False Positives routed to the human verification queue). Moreover, thresholds tuned on the validation split are not guaranteed to remain optimal under shifted class priors in future emergencies.

---

## 6. Probability Calibration & Interpretability Caveats
- **Model Scores Are Not Ground-Truth Certainties:** Even when evaluated with Brier scores or post-hoc sigmoid calibration (`CalibratedClassifierCV`), model-estimated scores reflect statistical regularities in bag-of-ngrams space and must never be interpreted as clinical urgency, life-safety severity, or verified authenticity.
- **Associational Feature Weights:** Top TF-IDF linear coefficients highlight words and bigrams that correlate with each label in the training set; they do not provide causal explanations or verify factual claims inside a message.


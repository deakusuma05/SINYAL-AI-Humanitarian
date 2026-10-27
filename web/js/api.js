/**
 * SINYAL Phase 3 — Frontend REST API Client (`web/js/api.js`)
 * Supports two execution environments seamlessly:
 * 1. FastAPI Backend Mode (Default on 127.0.0.1:8000 / Cloud Containers):
 *    Calls GET /api/health, GET /api/metadata, POST /api/predict, POST /api/batch, GET /api/evaluation.
 * 2. GitHub Pages Static Mode (`*.github.io`):
 *    Loads the exported Phase 2 artifacts (`static_data/model_weights.json`,
 *    `static_data/metadata.json`, `static_data/evaluation.json`) and executes the exact
 *    TF-IDF + LogisticRegression dot-product and frozen decision policy in the browser.
 */

(function (global) {
  "use strict";

  const TOPIC_LABELS = ["water", "food", "shelter", "medical_help"];
  const LABEL_DISPLAY_NAMES = {
    request: "Request (Aid Intent)",
    water: "Water",
    food: "Food",
    shelter: "Shelter",
    medical_help: "Medical Help",
  };
  const REASON_HUMAN_READABLE_MAP = {
    predicted_aid_request_verification:
      "Predicted aid request — human verification required before any action.",
    borderline_positive_request_score:
      "Borderline positive request score within ±0.08 uncertainty band.",
    borderline_request_score_below_threshold:
      "Near-threshold request score within ±0.08 uncertainty band — manual check recommended.",
    topic_detected_with_moderate_request_score:
      "Humanitarian need topic detected alongside moderate request score — verify whether aid is requested.",
    borderline_need_topic_score:
      "Borderline need-topic score within ±0.08 uncertainty band.",
    routine_low_score_stream_retained_not_discarded:
      "Routine monitoring stream — retained in queue and never automatically discarded.",
  };

  let staticBundleCache = null;

  function isGitHubPagesHost() {
    const host = (global.location && global.location.hostname) || "";
    return host.endsWith("github.io");
  }

  async function requestJSON(url, options = {}) {
    const defaultHeaders = {
      Accept: "application/json",
    };
    if (options.body) {
      defaultHeaders["Content-Type"] = "application/json";
    }

    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...(options.headers || {}),
      },
    });

    let payload = null;
    try {
      payload = await response.json();
    } catch (_err) {
      payload = null;
    }

    if (!response.ok) {
      const errorMessage =
        (payload && payload.message) ||
        (payload && payload.detail) ||
        `API Error (${response.status} ${response.statusText})`;
      const error = new Error(errorMessage);
      error.status = response.status;
      error.payload = payload;
      throw error;
    }

    return payload;
  }

  async function loadStaticModelBundle() {
    if (staticBundleCache) return staticBundleCache;
    const weights = await requestJSON("static_data/model_weights.json", { method: "GET" });
    staticBundleCache = weights;
    return weights;
  }

  function tokenizeAndCountNgrams(text, vocab) {
    const lower = String(text || "").toLowerCase();
    // Matches scikit-learn default token_pattern r"(?u)\b\w\w+\b"
    const tokens = lower.match(/[\p{L}\p{N}_]{2,}/gu) || [];
    const counts = new Map();

    for (let i = 0; i < tokens.length; i++) {
      const unigram = tokens[i];
      const uIdx = vocab[unigram];
      if (uIdx !== undefined) {
        counts.set(uIdx, (counts.get(uIdx) || 0) + 1);
      }
      if (i + 1 < tokens.length) {
        const bigram = `${unigram} ${tokens[i + 1]}`;
        const bIdx = vocab[bigram];
        if (bIdx !== undefined) {
          counts.set(bIdx, (counts.get(bIdx) || 0) + 1);
        }
      }
    }
    return counts;
  }

  function evaluateStaticSingleMessage(message, bundle) {
    const vocab = bundle.tfidf.vocab;
    const idf = bundle.tfidf.idf;
    const coefs = bundle.classifiers.coefs;
    const intercepts = bundle.classifiers.intercepts;
    const labels = bundle.target_labels;
    const policy = bundle.policy;
    const thresholds = policy.thresholds;
    const halfWidth = Number(policy.uncertainty_half_width || 0.08);
    const implicitRatio = Number(policy.implicit_request_ratio || 0.5);

    const counts = tokenizeAndCountNgrams(message, vocab);
    const tfidfFeatures = [];
    let normSq = 0.0;

    counts.forEach((count, featIdx) => {
      const tfVal = 1.0 + Math.log(count);
      const val = tfVal * idf[featIdx];
      tfidfFeatures.push([featIdx, val]);
      normSq += val * val;
    });

    const norm = normSq > 0.0 ? Math.sqrt(normSq) : 1.0;

    const scores = {};
    const predictions = {};
    const categories = [];

    labels.forEach((lbl) => {
      const w = coefs[lbl];
      let z = Number(intercepts[lbl] || 0.0);
      for (let i = 0; i < tfidfFeatures.length; i++) {
        const [fIdx, rawVal] = tfidfFeatures[i];
        z += (rawVal / norm) * w[fIdx];
      }
      const prob = Number((1.0 / (1.0 + Math.exp(-z))).toFixed(6));
      const tau = Number(thresholds[lbl]);
      const pred = prob >= tau ? 1 : 0;
      const lower = Number(Math.max(0.0, tau - halfWidth).toFixed(4));
      const upper = Number(Math.min(1.0, tau + halfWidth).toFixed(4));
      const inBand = Math.abs(prob - tau) <= halfWidth + 1e-9;

      scores[lbl] = prob;
      predictions[lbl] = pred;
      categories.push({
        label: lbl,
        display_name: LABEL_DISPLAY_NAMES[lbl] || lbl,
        is_request_intent: lbl === "request",
        predicted: pred,
        score: prob,
        threshold: tau,
        in_uncertainty_band: inBand,
        lower_uncertainty_bound: lower,
        upper_uncertainty_bound: upper,
      });
    });

    const reqScore = scores.request;
    const reqTau = Number(thresholds.request);
    const reqPred = predictions.request;
    const positiveNeeds = TOPIC_LABELS.filter((lbl) => predictions[lbl] === 1);

    const reqInBand = Math.abs(reqScore - reqTau) <= halfWidth + 1e-9;
    const topicInBand = TOPIC_LABELS.some(
      (lbl) => Math.abs(scores[lbl] - Number(thresholds[lbl])) <= halfWidth + 1e-9
    );

    const reasons = [];
    if (reqPred === 1) {
      reasons.push("predicted_aid_request_verification");
    }
    if (reqInBand) {
      reasons.push(
        reqPred === 1
          ? "borderline_positive_request_score"
          : "borderline_request_score_below_threshold"
      );
    }
    if (
      reqPred === 0 &&
      positiveNeeds.length > 0 &&
      reqScore >= reqTau * implicitRatio
    ) {
      reasons.push("topic_detected_with_moderate_request_score");
    }
    if (topicInBand) {
      reasons.push("borderline_need_topic_score");
    }

    const uncertaintyIndicated =
      reqInBand ||
      topicInBand ||
      reasons.includes("topic_detected_with_moderate_request_score");
    const reviewRecommended = reqPred === 1 || uncertaintyIndicated;

    const finalReasons =
      reasons.length > 0
        ? reasons
        : ["routine_low_score_stream_retained_not_discarded"];

    let reviewStatusLabel = "Routine monitoring";
    if (reqPred === 1) {
      reviewStatusLabel = "Human verification recommended";
    } else if (reviewRecommended) {
      reviewStatusLabel = "Borderline prediction — manual review";
    }

    return {
      message,
      request_prediction: reqPred,
      request_score: reqScore,
      request_threshold: reqTau,
      detected_need_categories: positiveNeeds,
      need_topic_predictions: {
        water: predictions.water,
        food: predictions.food,
        shelter: predictions.shelter,
        medical_help: predictions.medical_help,
      },
      need_topic_scores: {
        water: scores.water,
        food: scores.food,
        shelter: scores.shelter,
        medical_help: scores.medical_help,
      },
      predictions,
      scores,
      thresholds: {
        request: Number(thresholds.request),
        water: Number(thresholds.water),
        food: Number(thresholds.food),
        shelter: Number(thresholds.shelter),
        medical_help: Number(thresholds.medical_help),
      },
      categories,
      human_review_recommended: reviewRecommended,
      uncertainty_indicated: uncertaintyIndicated,
      review_status_label: reviewStatusLabel,
      review_reason_code: finalReasons.join("; "),
      review_reasons_list: finalReasons,
      review_reason_human_readable: finalReasons.map(
        (code) => REASON_HUMAN_READABLE_MAP[code] || code
      ),
      model_id: bundle.model_id,
      model_version: bundle.model_version,
      calibration_status: bundle.calibration_status,
      score_display_label: "Model-estimated score",
      supported_language_disclaimer:
        "Trained on English disaster-response messages and English translations; not validated on untranslated Indonesian text.",
    };
  }

  const SinyalAPI = {
    async fetchHealth() {
      if (isGitHubPagesHost()) {
        const bundle = await loadStaticModelBundle();
        return {
          status: "ok",
          api_available: true,
          model_available: true,
          selected_model_id: bundle.model_id,
          model_version: bundle.model_version,
          application_mode: "research_demo",
        };
      }
      return requestJSON("/api/health", { method: "GET" });
    },

    async fetchMetadata() {
      if (isGitHubPagesHost()) {
        return requestJSON("static_data/metadata.json", { method: "GET" });
      }
      return requestJSON("/api/metadata", { method: "GET" });
    },

    async predictSingle(message) {
      if (isGitHubPagesHost()) {
        const bundle = await loadStaticModelBundle();
        return evaluateStaticSingleMessage(message, bundle);
      }
      return requestJSON("/api/predict", {
        method: "POST",
        body: JSON.stringify({ message }),
      });
    },

    async predictBatch(messages) {
      if (isGitHubPagesHost()) {
        const bundle = await loadStaticModelBundle();
        const results = messages.map((msg) => evaluateStaticSingleMessage(msg, bundle));
        const flagged = results.filter((r) => r.human_review_recommended).length;
        const unc = results.filter((r) => r.uncertainty_indicated).length;
        return {
          total_messages: results.length,
          flagged_for_review_count: flagged,
          routine_monitoring_count: results.length - flagged,
          uncertainty_indicated_count: unc,
          results,
          retention_policy_note:
            "All messages are retained in the triage stream. Unflagged messages remain in routine monitoring and are never automatically deleted.",
          supported_language_disclaimer:
            "Trained on English disaster-response messages and English translations; not validated on untranslated Indonesian text.",
        };
      }
      return requestJSON("/api/batch", {
        method: "POST",
        body: JSON.stringify({ messages }),
      });
    },

    async fetchEvaluation() {
      if (isGitHubPagesHost()) {
        return requestJSON("static_data/evaluation.json", { method: "GET" });
      }
      return requestJSON("/api/evaluation", { method: "GET" });
    },
  };

  global.SinyalAPI = SinyalAPI;
})(window);

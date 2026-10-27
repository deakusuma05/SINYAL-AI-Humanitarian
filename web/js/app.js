/**
 * SINYAL Phase 3 — Primary Application Controller (`web/js/app.js`)
 * Implements:
 * - Safe DOM rendering (strictly `textContent` and `createElement`, preventing XSS)
 * - Live connection to FastAPI endpoints (`/api/health`, `/api/metadata`, `/api/predict`, `/api/batch`, `/api/evaluation`)
 * - Canonical 5-label ordering (`request`, `water`, `food`, `shelter`, `medical_help`)
 * - In-memory demonstration Human Review Queue with zero auto-deletion
 */

(function (global) {
  "use strict";

  const CANONICAL_LABELS = ["request", "water", "food", "shelter", "medical_help"];

  const DEFAULT_CATEGORY_SPECS = [
    { label: "request", display_name: "Request (Aid Intent)", is_request_intent: true, threshold: 0.19, lower: 0.11, upper: 0.27 },
    { label: "water", display_name: "Water", is_request_intent: false, threshold: 0.22, lower: 0.14, upper: 0.30 },
    { label: "food", display_name: "Food", is_request_intent: false, threshold: 0.26, lower: 0.18, upper: 0.34 },
    { label: "shelter", display_name: "Shelter", is_request_intent: false, threshold: 0.32, lower: 0.24, upper: 0.40 },
    { label: "medical_help", display_name: "Medical Help", is_request_intent: false, threshold: 0.20, lower: 0.12, upper: 0.28 },
  ];

  // Illustrative Example Inputs (Explicitly labeled as illustrative demo inputs, not held-out test rows)
  const ILLUSTRATIVE_EXAMPLES = {
    direct_multi:
      "Please help us urgently, we have 40 families with small children trapped near the school with no clean drinking water and no food since yesterday.",
    shelter_medical:
      "Our house roof collapsed in the storm and my elderly mother is injured and bleeding. We need emergency tents, blankets, and medical assistance.",
    borderline_implicit:
      "The river water level keeps rising near the market area and local food shops are closed for the second day.",
    non_request_news:
      "Meteorological officials reported that tropical storm winds have weakened as international relief agencies coordinate logistics at the main port.",
  };

  const ILLUSTRATIVE_BATCH_MESSAGES = [
    "We desperately need clean drinking water and baby food at the community center, over 60 people are waiting here.",
    "Several families lost their homes in the landslide and need tents and tarpaulins tonight before heavy rain returns.",
    "There are injured villagers with broken bones and high fever after the quake, please send medical supplies and a doctor.",
    "Local radio stations resumed broadcasting weather advisories and road closure updates across the northern district.",
    "Water pipes near the bridge are damaged and residents are boiling rainwater while waiting for utility repair crews.",
    "Thank you to the volunteers who distributed rice packets yesterday; highways to the eastern province are now open.",
  ];

  const state = {
    lastSingleResult: null,
    reviewQueue: [],
    nextQueueId: 1,
    activeQueueFilter: "all",
    metadata: null,
    evaluation: null,
  };

  /* ========================================================================
     DOM Helper Utilities (Safe Element Creation)
     ======================================================================== */
  function clearChildren(el) {
    if (!el) return;
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  function showGlobalAlert(title, message) {
    const banner = document.getElementById("global-alert-banner");
    const titleEl = document.getElementById("global-alert-title");
    const msgEl = document.getElementById("global-alert-message");
    if (!banner || !titleEl || !msgEl) return;
    titleEl.textContent = title;
    msgEl.textContent = message;
    banner.classList.remove("hidden");
  }

  function hideGlobalAlert() {
    const banner = document.getElementById("global-alert-banner");
    if (banner) banner.classList.add("hidden");
  }

  /* ========================================================================
     Module 3: Render 5 Multi-Label Category Cards
     ======================================================================== */
  function renderCategoryCards(categoriesData) {
    const grid = document.getElementById("multilabel-cards-grid");
    if (!grid) return;
    clearChildren(grid);

    const items = categoriesData && categoriesData.length === 5
      ? categoriesData
      : DEFAULT_CATEGORY_SPECS.map((spec) => ({
          label: spec.label,
          display_name: spec.display_name,
          is_request_intent: spec.is_request_intent,
          predicted: null,
          score: null,
          threshold: spec.threshold,
          in_uncertainty_band: false,
          lower_uncertainty_bound: spec.lower,
          upper_uncertainty_bound: spec.upper,
        }));

    items.forEach((cat) => {
      const card = document.createElement("article");
      card.className = "category-card";
      if (cat.is_request_intent) card.classList.add("card-intent");
      if (cat.predicted === 1) card.classList.add("card-positive");

      // Top Header
      const top = document.createElement("div");
      top.className = "cat-card-top";

      const titleWrap = document.createElement("div");
      const nameEl = document.createElement("h4");
      nameEl.className = "cat-name";
      nameEl.textContent = cat.display_name;
      const codeEl = document.createElement("span");
      codeEl.className = "cat-code";
      codeEl.textContent = cat.is_request_intent ? `label: ${cat.label} (Intent)` : `label: ${cat.label} (Need Topic)`;
      titleWrap.appendChild(nameEl);
      titleWrap.appendChild(codeEl);

      const badge = document.createElement("span");
      if (cat.predicted === null) {
        badge.className = "pred-badge pred-negative";
        badge.textContent = "Awaiting Input";
      } else if (cat.predicted === 1) {
        badge.className = "pred-badge pred-positive";
        badge.textContent = "Predicted Positive (1)";
      } else {
        badge.className = "pred-badge pred-negative";
        badge.textContent = "Predicted Negative (0)";
      }

      top.appendChild(titleWrap);
      top.appendChild(badge);

      // Score Block
      const scoreBlock = document.createElement("div");
      scoreBlock.className = "cat-score-block";
      const scoreCaption = document.createElement("span");
      scoreCaption.className = "cat-score-caption";
      scoreCaption.textContent = "Model-estimated score";
      const scoreNum = document.createElement("strong");
      scoreNum.className = "cat-score-number";
      scoreNum.textContent = cat.score !== null ? Number(cat.score).toFixed(4) : "—";
      scoreBlock.appendChild(scoreCaption);
      scoreBlock.appendChild(scoreNum);

      // Visual Threshold & Uncertainty Meter
      const meter = document.createElement("div");
      meter.className = "threshold-meter";
      meter.setAttribute(
        "title",
        `Frozen threshold τ* = ${cat.threshold.toFixed(2)} | Uncertainty band [${cat.lower_uncertainty_bound.toFixed(2)}, ${cat.upper_uncertainty_bound.toFixed(2)}]`
      );

      const bandZone = document.createElement("div");
      bandZone.className = "meter-uncertainty-zone";
      const leftPct = Math.max(0, Math.min(100, cat.lower_uncertainty_bound * 100));
      const widthPct = Math.max(0, Math.min(100 - leftPct, (cat.upper_uncertainty_bound - cat.lower_uncertainty_bound) * 100));
      bandZone.style.left = `${leftPct}%`;
      bandZone.style.width = `${widthPct}%`;

      const fill = document.createElement("div");
      fill.className = "meter-fill";
      const scorePct = cat.score !== null ? Math.max(0, Math.min(100, cat.score * 100)) : 0;
      fill.style.width = `${scorePct}%`;

      const threshLine = document.createElement("div");
      threshLine.className = "meter-threshold-line";
      threshLine.style.left = `${Math.max(0, Math.min(100, cat.threshold * 100))}%`;

      meter.appendChild(bandZone);
      meter.appendChild(fill);
      meter.appendChild(threshLine);

      // Footer metadata
      const footer = document.createElement("div");
      footer.className = "cat-meta-footer";
      const threshText = document.createElement("span");
      threshText.textContent = `Frozen Threshold (τ*): ${cat.threshold.toFixed(2)}`;
      const bandText = document.createElement("span");
      bandText.textContent = `Uncertainty Band: [${cat.lower_uncertainty_bound.toFixed(2)}, ${cat.upper_uncertainty_bound.toFixed(2)}]`;
      footer.appendChild(threshText);
      footer.appendChild(bandText);

      if (cat.in_uncertainty_band) {
        const bandTag = document.createElement("span");
        bandTag.className = "band-flag-tag";
        bandTag.textContent = "Within ±0.08 Uncertainty Band";
        footer.appendChild(bandTag);
      }

      card.appendChild(top);
      card.appendChild(scoreBlock);
      card.appendChild(meter);
      card.appendChild(footer);
      grid.appendChild(card);
    });
  }

  /* ========================================================================
     Module 2: Render Single-Message Analysis Summary
     ======================================================================== */
  function renderSinglePredictionResult(result) {
    state.lastSingleResult = result;

    const emptyState = document.getElementById("result-empty-state");
    const contentBox = document.getElementById("result-summary-content");
    const sendQueueBtn = document.getElementById("btn-send-to-queue");

    if (emptyState) emptyState.classList.add("hidden");
    if (contentBox) contentBox.classList.remove("hidden");
    if (sendQueueBtn) sendQueueBtn.disabled = false;

    const routingBanner = document.getElementById("result-routing-banner");
    const statusBadge = document.getElementById("result-review-status-badge");
    const uncBadge = document.getElementById("result-uncertainty-badge");
    const reasonsList = document.getElementById("result-reasons-list");

    if (routingBanner) {
      routingBanner.className = "routing-banner";
      if (result.request_prediction === 1) {
        routingBanner.classList.add("routing-verify");
      } else if (result.human_review_recommended) {
        routingBanner.classList.add("routing-borderline");
      } else {
        routingBanner.classList.add("routing-routine");
      }
    }

    if (statusBadge) {
      statusBadge.textContent = result.review_status_label;
    }

    if (uncBadge) {
      uncBadge.textContent = result.uncertainty_indicated
        ? "Uncertainty Indicated (±0.08 Band / Implicit Rule)"
        : "Outside Uncertainty Band";
    }

    if (reasonsList) {
      clearChildren(reasonsList);
      (result.review_reason_human_readable || []).forEach((reasonText) => {
        const li = document.createElement("li");
        li.textContent = reasonText;
        reasonsList.appendChild(li);
      });
    }

    const reqBadge = document.getElementById("result-request-badge");
    const reqScore = document.getElementById("result-request-score");
    if (reqBadge) {
      if (result.request_prediction === 1) {
        reqBadge.className = "pred-badge pred-positive";
        reqBadge.textContent = "Aid Request Detected (1)";
      } else {
        reqBadge.className = "pred-badge pred-negative";
        reqBadge.textContent = "No Direct Request (0)";
      }
    }
    if (reqScore) {
      reqScore.textContent = Number(result.request_score).toFixed(4);
    }

    const needTags = document.getElementById("result-need-tags");
    if (needTags) {
      clearChildren(needTags);
      const detected = result.detected_need_categories || [];
      if (detected.length === 0) {
        const noneSpan = document.createElement("span");
        noneSpan.className = "need-tag-none";
        noneSpan.textContent = "None above threshold";
        needTags.appendChild(noneSpan);
      } else {
        detected.forEach((lbl) => {
          const tag = document.createElement("span");
          tag.className = "need-tag";
          tag.textContent = lbl.replace("_", " ").toUpperCase();
          needTags.appendChild(tag);
        });
      }
    }

    const msgQuote = document.getElementById("result-original-message");
    if (msgQuote) {
      msgQuote.textContent = result.message;
    }

    const disclaimerEl = document.getElementById("result-language-disclaimer");
    if (disclaimerEl) {
      disclaimerEl.textContent = `${result.score_display_label}: Raw probability from ${result.model_id} (${result.calibration_status}). ${result.supported_language_disclaimer}`;
    }

    renderCategoryCards(result.categories);
  }

  /* ========================================================================
     Module 4: Human Review Queue Management
     ======================================================================== */
  function addPredictionsToQueue(predResults) {
    predResults.forEach((pred) => {
      const initialDemoStatus = pred.human_review_recommended
        ? "Pending Human Verification"
        : "Routine Monitoring Stream";
      state.reviewQueue.unshift({
        id: state.nextQueueId++,
        prediction: pred,
        demoStatus: initialDemoStatus,
        operatorReviewed: false,
        needsFollowUp: false,
      });
    });
    renderReviewQueueTable();
  }

  function updateQueueItemStatus(itemId, actionType) {
    const item = state.reviewQueue.find((q) => q.id === itemId);
    if (!item) return;
    if (actionType === "reviewed") {
      item.operatorReviewed = true;
      item.needsFollowUp = false;
      item.demoStatus = "Reviewed — Verified";
    } else if (actionType === "followup") {
      item.operatorReviewed = true;
      item.needsFollowUp = true;
      item.demoStatus = "Needs Volunteer Follow-up";
    }
    renderReviewQueueTable();
  }

  function renderReviewQueueTable() {
    const tbody = document.getElementById("queue-table-body");
    const badgeCount = document.getElementById("queue-count-badge");
    const statTotal = document.getElementById("q-stat-total");
    const statFlagged = document.getElementById("q-stat-flagged");
    const statRoutine = document.getElementById("q-stat-routine");
    const statReviewed = document.getElementById("q-stat-reviewed");

    const total = state.reviewQueue.length;
    const flagged = state.reviewQueue.filter((i) => i.prediction.human_review_recommended).length;
    const routine = total - flagged;
    const reviewed = state.reviewQueue.filter((i) => i.operatorReviewed).length;

    if (badgeCount) badgeCount.textContent = String(total);
    if (statTotal) statTotal.textContent = String(total);
    if (statFlagged) statFlagged.textContent = String(flagged);
    if (statRoutine) statRoutine.textContent = String(routine);
    if (statReviewed) statReviewed.textContent = String(reviewed);

    if (!tbody) return;
    clearChildren(tbody);

    const filtered = state.reviewQueue.filter((item) => {
      if (state.activeQueueFilter === "flagged") {
        return item.prediction.human_review_recommended;
      }
      if (state.activeQueueFilter === "routine") {
        return !item.prediction.human_review_recommended;
      }
      if (state.activeQueueFilter === "followup") {
        return item.needsFollowUp;
      }
      return true;
    });

    if (filtered.length === 0) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.className = "queue-empty-cell";
      td.textContent =
        total === 0
          ? 'No messages in the local session queue yet. Click "Load Illustrative 6-Message Disaster Batch" above or analyze a message in Module 2.'
          : "No messages match the selected queue filter.";
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    filtered.forEach((item) => {
      const pred = item.prediction;
      const tr = document.createElement("tr");

      // 1. ID
      const tdId = document.createElement("td");
      tdId.textContent = `#${item.id}`;

      // 2. Message Preview (Safe textContent)
      const tdMsg = document.createElement("td");
      tdMsg.className = "queue-msg-preview";
      tdMsg.textContent = pred.message;

      // 3. Request Intent & Score
      const tdReq = document.createElement("td");
      const reqBadge = document.createElement("span");
      reqBadge.className =
        pred.request_prediction === 1 ? "pred-badge pred-positive" : "pred-badge pred-negative";
      reqBadge.textContent = pred.request_prediction === 1 ? "Request (1)" : "Non-Request (0)";
      const scoreText = document.createElement("div");
      scoreText.style.marginTop = "4px";
      scoreText.style.fontSize = "12px";
      scoreText.textContent = `Score: ${Number(pred.request_score).toFixed(4)} (τ=0.19)`;
      tdReq.appendChild(reqBadge);
      tdReq.appendChild(scoreText);

      // 4. Detected Need Topics
      const tdNeeds = document.createElement("td");
      const needsWrap = document.createElement("div");
      needsWrap.className = "need-tags-container";
      if (!pred.detected_need_categories || pred.detected_need_categories.length === 0) {
        const noneEl = document.createElement("span");
        noneEl.className = "need-tag-none";
        noneEl.textContent = "None";
        needsWrap.appendChild(noneEl);
      } else {
        pred.detected_need_categories.forEach((lbl) => {
          const tag = document.createElement("span");
          tag.className = "need-tag";
          tag.textContent = lbl.replace("_", " ");
          needsWrap.appendChild(tag);
        });
      }
      tdNeeds.appendChild(needsWrap);

      // 5. Routing & Review Reason
      const tdReason = document.createElement("td");
      const routeStrong = document.createElement("strong");
      routeStrong.style.display = "block";
      routeStrong.style.fontSize = "12px";
      routeStrong.textContent = pred.review_status_label;
      const reasonSub = document.createElement("span");
      reasonSub.style.fontSize = "12px";
      reasonSub.style.color = "#334E68";
      reasonSub.textContent = (pred.review_reason_human_readable || []).join(" ");
      tdReason.appendChild(routeStrong);
      tdReason.appendChild(reasonSub);

      // 6. Demo Reviewer Status
      const tdStatus = document.createElement("td");
      const statusChip = document.createElement("span");
      statusChip.className = "status-chip";
      if (item.needsFollowUp) {
        statusChip.classList.add("status-followup");
      } else if (item.operatorReviewed) {
        statusChip.classList.add("status-reviewed");
      } else if (pred.human_review_recommended) {
        statusChip.classList.add("status-pending");
      } else {
        statusChip.classList.add("status-routine");
      }
      statusChip.textContent = item.demoStatus;
      tdStatus.appendChild(statusChip);

      // 7. Volunteer Actions
      const tdActions = document.createElement("td");
      const actionWrap = document.createElement("div");
      actionWrap.className = "queue-action-btns";

      const btnReview = document.createElement("button");
      btnReview.type = "button";
      btnReview.className = "btn-table-action";
      btnReview.textContent = "Mark reviewed";
      btnReview.addEventListener("click", () => updateQueueItemStatus(item.id, "reviewed"));

      const btnFollow = document.createElement("button");
      btnFollow.type = "button";
      btnFollow.className = "btn-table-action";
      btnFollow.textContent = "Mark needs follow-up";
      btnFollow.addEventListener("click", () => updateQueueItemStatus(item.id, "followup"));

      actionWrap.appendChild(btnReview);
      actionWrap.appendChild(btnFollow);
      tdActions.appendChild(actionWrap);

      tr.appendChild(tdId);
      tr.appendChild(tdMsg);
      tr.appendChild(tdReq);
      tr.appendChild(tdNeeds);
      tr.appendChild(tdReason);
      tr.appendChild(tdStatus);
      tr.appendChild(tdActions);

      tbody.appendChild(tr);
    });
  }

  /* ========================================================================
     Module 5: Populate Frozen Evaluation Table & KPIs
     ======================================================================== */
  function populateEvaluationSection(evalData) {
    if (!evalData) return;

    const frozen = evalData.held_out_test_metrics_frozen_thresholds || {};
    const defRef = evalData.held_out_test_metrics_default_0_50_reference || {};
    const policy = evalData.human_review_policy_test || {};

    const kpiMacro = document.getElementById("kpi-macro-f1");
    const kpiMicro = document.getElementById("kpi-micro-f1");
    const kpiReqF2 = document.getElementById("kpi-request-f2");
    const kpiReqRec = document.getElementById("kpi-request-recall");
    const kpiFlagRate = document.getElementById("kpi-flag-rate");
    const kpiCapture = document.getElementById("kpi-queue-capture");

    if (kpiMacro && frozen.macro_f1 !== undefined) {
      kpiMacro.textContent = Number(frozen.macro_f1).toFixed(4);
    }
    if (kpiMicro && frozen.micro_f1 !== undefined) {
      kpiMicro.textContent = Number(frozen.micro_f1).toFixed(4);
    }
    if (kpiReqF2 && frozen.request_f2 !== undefined) {
      kpiReqF2.textContent = Number(frozen.request_f2).toFixed(4);
    }
    if (
      kpiReqRec &&
      frozen.per_label &&
      frozen.per_label.request &&
      frozen.per_label.request.recall !== undefined
    ) {
      kpiReqRec.textContent = `${(Number(frozen.per_label.request.recall) * 100).toFixed(2)}%`;
    }
    if (kpiFlagRate && policy.flagged_for_review_rate !== undefined) {
      kpiFlagRate.textContent = `${(Number(policy.flagged_for_review_rate) * 100).toFixed(2)}%`;
    }
    if (kpiCapture && policy.request_recall_among_flagged_queue !== undefined) {
      kpiCapture.textContent = `${(Number(policy.request_recall_among_flagged_queue) * 100).toFixed(2)}%`;
    }

    // Populate Per-Label Table
    const tbody = document.getElementById("eval-per-label-tbody");
    if (tbody && frozen.per_label) {
      clearChildren(tbody);
      CANONICAL_LABELS.forEach((lbl) => {
        const m = frozen.per_label[lbl];
        const d = defRef.per_label ? defRef.per_label[lbl] : null;
        if (!m) return;

        const tr = document.createElement("tr");
        const cm = m.confusion_matrix || {};

        const cells = [
          lbl,
          String(m.positive_support),
          Number(m.threshold).toFixed(2),
          Number(m.precision).toFixed(4),
          Number(m.recall).toFixed(4),
          Number(m.f1).toFixed(4),
          Number(m.f2).toFixed(4),
          d ? Number(d.f1).toFixed(4) : "—",
          Number(m.average_precision).toFixed(4),
          `TP:${cm.tp} / FP:${cm.fp} / FN:${cm.fn} / TN:${cm.tn}`,
        ];

        cells.forEach((val, idx) => {
          const td = document.createElement("td");
          if (idx === 0) {
            const code = document.createElement("code");
            code.textContent = val;
            td.appendChild(code);
          } else {
            td.textContent = val;
          }
          tr.appendChild(td);
        });

        tbody.appendChild(tr);
      });
    }

    if (global.SinyalCharts) {
      global.SinyalCharts.renderAll(evalData);
    }
  }

  /* ========================================================================
     Initialization & Event Listeners
     ======================================================================== */
  async function initializeDashboard() {
    renderCategoryCards(null);

    const healthPill = document.getElementById("system-health-pill");
    const healthText = document.getElementById("health-status-text");

    try {
      const health = await global.SinyalAPI.fetchHealth();
      if (health && health.status === "ok" && health.model_available) {
        if (healthPill) {
          healthPill.classList.remove("health-checking", "health-error");
          healthPill.classList.add("health-ok");
        }
        if (healthText) {
          healthText.textContent = `API & Model Ready (${health.selected_model_id})`;
        }
      } else {
        if (healthPill) {
          healthPill.classList.remove("health-checking", "health-ok");
          healthPill.classList.add("health-error");
        }
        if (healthText) {
          healthText.textContent = "Model Unavailable (Degraded Mode)";
        }
        showGlobalAlert(
          "Model Service Degraded:",
          "The FastAPI backend is reachable, but the serialized scikit-learn model bundle could not be loaded."
        );
      }
    } catch (err) {
      if (healthPill) {
        healthPill.classList.remove("health-checking", "health-ok");
        healthPill.classList.add("health-error");
      }
      if (healthText) {
        healthText.textContent = "API Connection Error";
      }
      showGlobalAlert(
        "Backend Connection Error:",
        `Could not connect to FastAPI server (${err.message}). Ensure Uvicorn is running.`
      );
    }

    try {
      const metadata = await global.SinyalAPI.fetchMetadata();
      state.metadata = metadata;
      const metaId = document.getElementById("meta-model-id");
      const metaCal = document.getElementById("meta-calibration");
      const metaScoreLbl = document.getElementById("meta-score-label");
      if (metaId && metadata.model_id) metaId.textContent = metadata.model_id;
      if (metaCal && metadata.probability_calibration_status) {
        metaCal.textContent = metadata.probability_calibration_status;
      }
      if (metaScoreLbl && metadata.score_display_label) {
        metaScoreLbl.textContent = metadata.score_display_label;
      }
    } catch (_metaErr) {
      // Handled by health check banner
    }

    try {
      const evalData = await global.SinyalAPI.fetchEvaluation();
      state.evaluation = evalData;
      populateEvaluationSection(evalData);
    } catch (evalErr) {
      showGlobalAlert(
        "Evaluation Artifact Notice:",
        `Could not load frozen Phase 2 evaluation metrics: ${evalErr.message}`
      );
    }
  }

  function bindEvents() {
    const closeAlertBtn = document.getElementById("global-alert-close");
    if (closeAlertBtn) {
      closeAlertBtn.addEventListener("click", hideGlobalAlert);
    }

    // Character counter for single message input
    const msgInput = document.getElementById("message-input");
    const charCounter = document.getElementById("message-char-counter");
    if (msgInput && charCounter) {
      msgInput.addEventListener("input", () => {
        charCounter.textContent = `${msgInput.value.length.toLocaleString()} / 10,000 chars`;
      });
    }

    // Single message form submission
    const singleForm = document.getElementById("single-predict-form");
    const analyzeBtn = document.getElementById("btn-analyze");
    const analyzeBtnText = document.getElementById("btn-analyze-text");
    const analyzerError = document.getElementById("analyzer-error-box");

    async function executeSingleAnalysis(messageText) {
      if (analyzerError) analyzerError.classList.add("hidden");
      const trimmed = (messageText || "").trim();
      if (!trimmed) {
        if (analyzerError) {
          analyzerError.textContent = "Please enter a non-empty disaster message before analyzing.";
          analyzerError.classList.remove("hidden");
        }
        return;
      }
      if (messageText.length > 10000) {
        if (analyzerError) {
          analyzerError.textContent = "Message exceeds maximum allowed length of 10,000 characters.";
          analyzerError.classList.remove("hidden");
        }
        return;
      }

      if (analyzeBtn) analyzeBtn.disabled = true;
      if (analyzeBtnText) analyzeBtnText.textContent = "Running Real Model Inference...";

      try {
        const result = await global.SinyalAPI.predictSingle(messageText);
        renderSinglePredictionResult(result);
      } catch (err) {
        if (analyzerError) {
          analyzerError.textContent = err.message || "Failed to classify message.";
          analyzerError.classList.remove("hidden");
        }
      } finally {
        if (analyzeBtn) analyzeBtn.disabled = false;
        if (analyzeBtnText) analyzeBtnText.textContent = "Analyze Message with SINYAL";
      }
    }

    if (singleForm) {
      singleForm.addEventListener("submit", (e) => {
        e.preventDefault();
        if (msgInput) executeSingleAnalysis(msgInput.value);
      });
    }

    // Clear button
    const clearBtn = document.getElementById("btn-clear-input");
    if (clearBtn && msgInput) {
      clearBtn.addEventListener("click", () => {
        msgInput.value = "";
        if (charCounter) charCounter.textContent = "0 / 10,000 chars";
        if (analyzerError) analyzerError.classList.add("hidden");
        state.lastSingleResult = null;
        const emptyState = document.getElementById("result-empty-state");
        const contentBox = document.getElementById("result-summary-content");
        const sendQueueBtn = document.getElementById("btn-send-to-queue");
        if (emptyState) emptyState.classList.remove("hidden");
        if (contentBox) contentBox.classList.add("hidden");
        if (sendQueueBtn) sendQueueBtn.disabled = true;
        renderCategoryCards(null);
        msgInput.focus();
      });
    }

    // Illustrative Example Buttons
    document.querySelectorAll(".btn-example").forEach((btn) => {
      btn.addEventListener("click", () => {
        const exId = btn.getAttribute("data-example-id");
        const sampleText = ILLUSTRATIVE_EXAMPLES[exId];
        if (sampleText && msgInput) {
          msgInput.value = sampleText;
          if (charCounter) charCounter.textContent = `${sampleText.length} / 10,000 chars`;
          executeSingleAnalysis(sampleText);
        }
      });
    });

    // Send single result to Review Queue
    const sendToQueueBtn = document.getElementById("btn-send-to-queue");
    if (sendToQueueBtn) {
      sendToQueueBtn.addEventListener("click", () => {
        if (state.lastSingleResult) {
          addPredictionsToQueue([state.lastSingleResult]);
          sendToQueueBtn.disabled = true;
        }
      });
    }

    // Batch Input Line Counter
    const batchInput = document.getElementById("batch-messages-input");
    const batchCounter = document.getElementById("batch-line-counter");
    function updateBatchLineCount() {
      if (!batchInput || !batchCounter) return;
      const lines = batchInput.value
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter((l) => l.length > 0);
      batchCounter.textContent = `${lines.length} / 50 messages`;
    }
    if (batchInput) {
      batchInput.addEventListener("input", updateBatchLineCount);
    }

    // Load Illustrative 6-Message Batch
    const loadSampleBatchBtn = document.getElementById("btn-load-sample-batch");
    if (loadSampleBatchBtn && batchInput) {
      loadSampleBatchBtn.addEventListener("click", async () => {
        batchInput.value = ILLUSTRATIVE_BATCH_MESSAGES.join("\n");
        updateBatchLineCount();
        await executeBatchScreening(ILLUSTRATIVE_BATCH_MESSAGES);
      });
    }

    // Batch Form Submit
    const batchForm = document.getElementById("batch-predict-form");
    const batchErrorBox = document.getElementById("batch-error-box");
    const runBatchBtn = document.getElementById("btn-run-batch");

    async function executeBatchScreening(messagesList) {
      if (batchErrorBox) batchErrorBox.classList.add("hidden");
      if (!messagesList || messagesList.length === 0) {
        if (batchErrorBox) {
          batchErrorBox.textContent = "Please enter at least one non-empty message (one per line).";
          batchErrorBox.classList.remove("hidden");
        }
        return;
      }
      if (messagesList.length > 50) {
        if (batchErrorBox) {
          batchErrorBox.textContent = `Batch size (${messagesList.length}) exceeds the maximum limit of 50 messages per request.`;
          batchErrorBox.classList.remove("hidden");
        }
        return;
      }

      if (runBatchBtn) {
        runBatchBtn.disabled = true;
        runBatchBtn.textContent = "Screening Batch...";
      }

      try {
        const batchRes = await global.SinyalAPI.predictBatch(messagesList);
        if (batchRes && Array.isArray(batchRes.results)) {
          // Add in reverse so index 0 appears at top of queue or preserve chronological order
          addPredictionsToQueue([...batchRes.results].reverse());
        }
      } catch (err) {
        if (batchErrorBox) {
          batchErrorBox.textContent = err.message || "Batch screening failed.";
          batchErrorBox.classList.remove("hidden");
        }
      } finally {
        if (runBatchBtn) {
          runBatchBtn.disabled = false;
          runBatchBtn.textContent = "Run Batch Triage Screening";
        }
      }
    }

    if (batchForm && batchInput) {
      batchForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const lines = batchInput.value
          .split(/\r?\n/)
          .map((l) => l.trim())
          .filter((l) => l.length > 0);
        executeBatchScreening(lines);
      });
    }

    // Clear Queue button
    const clearQueueBtn = document.getElementById("btn-clear-queue");
    if (clearQueueBtn) {
      clearQueueBtn.addEventListener("click", () => {
        state.reviewQueue = [];
        state.nextQueueId = 1;
        renderReviewQueueTable();
      });
    }

    // Queue Filter buttons
    document.querySelectorAll(".btn-filter").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-filter").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeQueueFilter = btn.getAttribute("data-filter") || "all";
        renderReviewQueueTable();
      });
    });

    // Navigation active state highlight
    document.querySelectorAll(".nav-link").forEach((link) => {
      link.addEventListener("click", () => {
        document.querySelectorAll(".nav-link").forEach((l) => l.classList.remove("active"));
        link.classList.add("active");
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    initializeDashboard();
  });
})(window);


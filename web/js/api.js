/**
 * SINYAL Phase 3 — Frontend REST API Client (`web/js/api.js`)
 * Communicates strictly with the local FastAPI backend endpoints:
 * - GET  /api/health
 * - GET  /api/metadata
 * - POST /api/predict
 * - POST /api/batch
 * - GET  /api/evaluation
 */

(function (global) {
  "use strict";

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

  const SinyalAPI = {
    async fetchHealth() {
      return requestJSON("/api/health", { method: "GET" });
    },

    async fetchMetadata() {
      return requestJSON("/api/metadata", { method: "GET" });
    },

    async predictSingle(message) {
      return requestJSON("/api/predict", {
        method: "POST",
        body: JSON.stringify({ message }),
      });
    },

    async predictBatch(messages) {
      return requestJSON("/api/batch", {
        method: "POST",
        body: JSON.stringify({ messages }),
      });
    },

    async fetchEvaluation() {
      return requestJSON("/api/evaluation", { method: "GET" });
    },
  };

  global.SinyalAPI = SinyalAPI;
})(window);


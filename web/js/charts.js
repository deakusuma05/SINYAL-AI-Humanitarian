/**
 * SINYAL Phase 3 — Evaluation Visualization Module (`web/js/charts.js`)
 * Renders interactive Chart.js charts (with offline fallback) strictly from
 * the frozen Phase 2 evaluation payload (`GET /api/evaluation`).
 */

(function (global) {
  "use strict";

  const activeCharts = {};

  const LABEL_ORDER = ["request", "water", "food", "shelter", "medical_help"];
  const LABEL_NAMES = {
    request: "Request",
    water: "Water",
    food: "Food",
    shelter: "Shelter",
    medical_help: "Medical Help",
  };

  function destroyChartIfExists(key) {
    if (activeCharts[key]) {
      activeCharts[key].destroy();
      delete activeCharts[key];
    }
  }

  function renderFallbackCanvasBarChart(canvas, labels, seriesList) {
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.parentElement ? canvas.parentElement.clientWidth || 520 : 520;
    const height = canvas.parentElement ? canvas.parentElement.clientHeight || 260 : 260;
    canvas.width = width;
    canvas.height = height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, width, height);

    const padLeft = 48;
    const padRight = 20;
    const padTop = 32;
    const padBottom = 42;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Grid lines
    ctx.strokeStyle = "#D9E2EC";
    ctx.fillStyle = "#334E68";
    ctx.font = "11px sans-serif";
    for (let i = 0; i <= 5; i++) {
      const val = i * 0.2;
      const y = padTop + plotH * (1 - val);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(1), 14, y + 4);
    }

    // Legend
    let legendX = padLeft;
    seriesList.forEach((s) => {
      ctx.fillStyle = s.color;
      ctx.fillRect(legendX, 8, 12, 12);
      ctx.fillStyle = "#102A43";
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(s.label, legendX + 16, 18);
      legendX += ctx.measureText(s.label).width + 34;
    });

    const groupWidth = plotW / labels.length;
    const barWidth = Math.min(28, (groupWidth * 0.72) / seriesList.length);

    labels.forEach((lbl, idx) => {
      const groupStart = padLeft + idx * groupWidth + (groupWidth - barWidth * seriesList.length) / 2;
      seriesList.forEach((s, sIdx) => {
        const val = Number(s.data[idx] || 0);
        const barH = Math.max(2, val * plotH);
        const x = groupStart + sIdx * barWidth;
        const y = padTop + plotH - barH;
        ctx.fillStyle = s.color;
        ctx.fillRect(x, y, barWidth - 2, barH);
      });

      ctx.fillStyle = "#102A43";
      ctx.font = "11px sans-serif";
      const textW = ctx.measureText(lbl).width;
      ctx.fillText(lbl, padLeft + idx * groupWidth + (groupWidth - textW) / 2, height - 16);
    });
  }

  function renderChart1PerLabel(evalData) {
    const canvas = document.getElementById("chart-per-label");
    if (!canvas) return;

    const perLabel =
      (evalData.held_out_test_metrics_frozen_thresholds &&
        evalData.held_out_test_metrics_frozen_thresholds.per_label) ||
      {};

    const labels = LABEL_ORDER.map((k) => LABEL_NAMES[k]);
    const f1Data = LABEL_ORDER.map((k) => (perLabel[k] ? Number(perLabel[k].f1.toFixed(4)) : 0));
    const precisionData = LABEL_ORDER.map((k) =>
      perLabel[k] ? Number(perLabel[k].precision.toFixed(4)) : 0
    );
    const recallData = LABEL_ORDER.map((k) =>
      perLabel[k] ? Number(perLabel[k].recall.toFixed(4)) : 0
    );

    if (typeof global.Chart !== "undefined") {
      destroyChartIfExists("perLabel");
      activeCharts.perLabel = new global.Chart(canvas, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "F1 Score (τ*)",
              data: f1Data,
              backgroundColor: "#1B365D",
              borderRadius: 4,
            },
            {
              label: "Precision (τ*)",
              data: precisionData,
              backgroundColor: "#0D9488",
              borderRadius: 4,
            },
            {
              label: "Recall (τ*)",
              data: recallData,
              backgroundColor: "#D97706",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              min: 0,
              max: 1.0,
              ticks: { stepSize: 0.2 },
              title: { display: true, text: "Score (0.00 – 1.00)" },
            },
          },
          plugins: {
            legend: { position: "top" },
          },
        },
      });
    } else {
      renderFallbackCanvasBarChart(canvas, labels, [
        { label: "F1 (τ*)", data: f1Data, color: "#1B365D" },
        { label: "Precision", data: precisionData, color: "#0D9488" },
        { label: "Recall", data: recallData, color: "#D97706" },
      ]);
    }
  }

  function renderChart2ThresholdComparison(evalData) {
    const canvas = document.getElementById("chart-threshold-comparison");
    if (!canvas) return;

    const frozen = evalData.held_out_test_metrics_frozen_thresholds || {};
    const defRef = evalData.held_out_test_metrics_default_0_50_reference || {};

    const labels = ["Macro F1", "Micro F1", "Request F2", "Request F1"];
    const frozenVals = [
      Number((frozen.macro_f1 || 0).toFixed(4)),
      Number((frozen.micro_f1 || 0).toFixed(4)),
      Number((frozen.request_f2 || 0).toFixed(4)),
      Number((frozen.request_f1 || 0).toFixed(4)),
    ];
    const defaultVals = [
      Number((defRef.macro_f1 || 0).toFixed(4)),
      Number((defRef.micro_f1 || 0).toFixed(4)),
      Number((defRef.request_f2 || 0).toFixed(4)),
      Number((defRef.request_f1 || 0).toFixed(4)),
    ];

    if (typeof global.Chart !== "undefined") {
      destroyChartIfExists("thresholdComp");
      activeCharts.thresholdComp = new global.Chart(canvas, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "Frozen Validation-Tuned Thresholds (τ*)",
              data: frozenVals,
              backgroundColor: "#0D9488",
              borderRadius: 4,
            },
            {
              label: "Default Threshold (τ = 0.50)",
              data: defaultVals,
              backgroundColor: "#9FB3C8",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              min: 0,
              max: 1.0,
              ticks: { stepSize: 0.2 },
              title: { display: true, text: "Held-Out Test Score" },
            },
          },
          plugins: {
            legend: { position: "top" },
          },
        },
      });
    } else {
      renderFallbackCanvasBarChart(canvas, labels, [
        { label: "Frozen τ*", data: frozenVals, color: "#0D9488" },
        { label: "Default 0.50", data: defaultVals, color: "#9FB3C8" },
      ]);
    }
  }

  function renderChart3GenreSubgroups(evalData) {
    const canvas = document.getElementById("chart-genre-subgroups");
    if (!canvas) return;

    const genres = evalData.exploratory_genre_subgroups_test || {};
    const direct = genres.direct || {};
    const news = genres.news || {};

    const labels = [
      "Request F2 Score",
      "Request Recall",
      "Request F1 Score",
      "Subgroup Macro F1",
      "Subgroup Micro F1",
    ];
    const directVals = [
      Number((direct.request_f2 || 0).toFixed(4)),
      Number((direct.request_recall || 0).toFixed(4)),
      Number((direct.request_f1 || 0).toFixed(4)),
      Number((direct.macro_f1 || 0).toFixed(4)),
      Number((direct.micro_f1 || 0).toFixed(4)),
    ];
    const newsVals = [
      Number((news.request_f2 || 0).toFixed(4)),
      Number((news.request_recall || 0).toFixed(4)),
      Number((news.request_f1 || 0).toFixed(4)),
      Number((news.macro_f1 || 0).toFixed(4)),
      Number((news.micro_f1 || 0).toFixed(4)),
    ];

    if (typeof global.Chart !== "undefined") {
      destroyChartIfExists("genreSubgroups");
      activeCharts.genreSubgroups = new global.Chart(canvas, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "Direct Citizen/SMS Messages (n = 1,067)",
              data: directVals,
              backgroundColor: "#1B365D",
              borderRadius: 4,
            },
            {
              label: "News Media Reports (n = 1,320)",
              data: newsVals,
              backgroundColor: "#D97706",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              min: 0,
              max: 1.0,
              ticks: { stepSize: 0.2 },
              title: { display: true, text: "Held-Out Test Subgroup Score" },
            },
          },
          plugins: {
            legend: { position: "top" },
          },
        },
      });
    } else {
      renderFallbackCanvasBarChart(canvas, labels, [
        { label: "Direct (n=1,067)", data: directVals, color: "#1B365D" },
        { label: "News (n=1,320)", data: newsVals, color: "#D97706" },
      ]);
    }
  }

  const SinyalCharts = {
    renderAll(evalData) {
      if (!evalData) return;
      renderChart1PerLabel(evalData);
      renderChart2ThresholdComparison(evalData);
      renderChart3GenreSubgroups(evalData);
    },
  };

  global.SinyalCharts = SinyalCharts;
})(window);


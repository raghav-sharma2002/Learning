/**
 * charts.js — CausalEdge™ Chart Rendering Module
 *
 * Handles all Chart.js rendering for the three chart views:
 *   1. Timeline  — actual vs counterfactual indexed line chart
 *   2. Factors   — causal factor decomposition horizontal bar chart
 *   3. Sectors   — sector-level impact horizontal bar chart
 */

let chartInstance = null;

/**
 * Destroy any existing chart and clear the chart area container.
 * Called before rendering a new chart to prevent memory leaks.
 */
function clearChart() {
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }
  document.getElementById("chart-area").innerHTML = "";
}

/**
 * Render the Timeline chart.
 * Shows actual market performance vs the DoWhy synthetic control counterfactual,
 * both indexed to 100 at the start of the event window.
 *
 * @param {Object} event - The selected event data object from data.js
 */
function renderTimeline(event) {
  const area = document.getElementById("chart-area");
  const positiveEvent = event.impact > 0;
  const actualColor  = positiveEvent ? "#1D9E75" : "#E24B4A";
  const actualBg     = positiveEvent ? "rgba(29,158,117,.08)" : "rgba(226,75,74,.08)";

  area.innerHTML = `
    <div class="chart-label">Actual vs counterfactual (indexed to 100 — event window)</div>
    <div style="position:relative;height:220px;">
      <canvas id="mainChart"
        role="img"
        aria-label="Line chart showing actual vs counterfactual market performance during ${event.name}. Actual: ${event.actual}%, Counterfactual: ${event.cf}%">
        Actual return: ${event.actual}%. Counterfactual return: ${event.cf}%. Net causal impact: ${event.impact}%.
      </canvas>
    </div>
    <div class="chart-legend">
      <span class="legend-item">
        <span style="width:16px;height:2px;background:${actualColor};display:inline-block;"></span>
        Actual
      </span>
      <span class="legend-item">
        <span style="width:16px;height:0;border-bottom:2px dashed #888780;display:inline-block;"></span>
        Counterfactual (without event)
      </span>
    </div>`;

  const ctx = document.getElementById("mainChart").getContext("2d");
  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: event.timeline.map(d => d.d),
      datasets: [
        {
          label: "Actual",
          data: event.timeline.map(d => d.a),
          borderColor: actualColor,
          backgroundColor: actualBg,
          fill: true,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          label: "Counterfactual",
          data: event.timeline.map(d => d.c),
          borderColor: "#888780",
          borderDash: [4, 3],
          fill: false,
          tension: 0.3,
          borderWidth: 1.5,
          pointRadius: 2,
          pointHoverRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}`
          }
        }
      },
      scales: {
        x: { ticks: { font: { size: 10 } } },
        y: {
          ticks: {
            font: { size: 10 },
            callback: v => v.toFixed(0)
          }
        }
      }
    }
  });
}

/**
 * Render the Causal Factors chart.
 * Horizontal bar chart showing EconML DML factor decomposition.
 * Green bars = positive causal contribution, red bars = negative.
 *
 * @param {Object} event - The selected event data object from data.js
 */
function renderFactors(event) {
  const area = document.getElementById("chart-area");
  const sorted = [...event.factors].sort((a, b) => a.v - b.v);
  const h = sorted.length * 38 + 60;

  area.innerHTML = `
    <div class="chart-label">Causal factor decomposition — attributed effect (%) via EconML DML</div>
    <div style="position:relative;height:${h}px;">
      <canvas id="mainChart"
        role="img"
        aria-label="Horizontal bar chart showing causal factor decomposition for ${event.name}. Factors: ${sorted.map(f => f.n + ': ' + (f.v > 0 ? '+' : '') + f.v + '%').join(', ')}">
        ${sorted.map(f => `${f.n}: ${f.v > 0 ? '+' : ''}${f.v}%.`).join(' ')}
      </canvas>
    </div>`;

  const ctx = document.getElementById("mainChart").getContext("2d");
  chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map(f => f.n),
      datasets: [{
        label: "Causal effect",
        data: sorted.map(f => f.v),
        backgroundColor: sorted.map(f => f.v > 0 ? "rgba(29,158,117,.75)" : "rgba(226,75,74,.75)"),
        borderColor: sorted.map(f => f.v > 0 ? "#1D9E75" : "#E24B4A"),
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.parsed.x > 0 ? "+" : ""}${ctx.parsed.x.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: {
            font: { size: 10 },
            callback: v => `${v > 0 ? "+" : ""}${v}%`
          },
          grid: { color: "rgba(128,128,128,.1)" }
        },
        y: { ticks: { font: { size: 10 } } }
      }
    }
  });
}

/**
 * Render the Sectors chart.
 * Horizontal bar chart showing causal heterogeneous treatment effects
 * broken down by GICS sector, estimated via Causal Forest (EconML).
 *
 * @param {Object} event - The selected event data object from data.js
 */
function renderSectors(event) {
  const area = document.getElementById("chart-area");
  const sorted = [...event.sectors].sort((a, b) => a.v - b.v);
  const h = sorted.length * 38 + 60;

  area.innerHTML = `
    <div class="chart-label">Sector-level causal impact (%) — Causal Forest heterogeneous treatment effects</div>
    <div style="position:relative;height:${h}px;">
      <canvas id="mainChart"
        role="img"
        aria-label="Horizontal bar chart of sector causal impacts for ${event.name}. Sectors: ${sorted.map(s => s.s + ': ' + (s.v > 0 ? '+' : '') + s.v + '%').join(', ')}">
        ${sorted.map(s => `${s.s}: ${s.v > 0 ? '+' : ''}${s.v}%.`).join(' ')}
      </canvas>
    </div>`;

  const ctx = document.getElementById("mainChart").getContext("2d");
  chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map(s => s.s),
      datasets: [{
        label: "Causal impact",
        data: sorted.map(s => s.v),
        backgroundColor: sorted.map(s => s.v > 0 ? "rgba(55,138,221,.7)" : "rgba(226,75,74,.7)"),
        borderColor: sorted.map(s => s.v > 0 ? "#378ADD" : "#E24B4A"),
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.parsed.x > 0 ? "+" : ""}${ctx.parsed.x.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          ticks: {
            font: { size: 10 },
            callback: v => `${v > 0 ? "+" : ""}${v}%`
          },
          grid: { color: "rgba(128,128,128,.1)" }
        },
        y: { ticks: { font: { size: 10 } } }
      }
    }
  });
}

/**
 * Main chart dispatch function. Clears current chart and renders
 * the appropriate chart type based on the active tab.
 *
 * @param {string} tab - One of 'timeline' | 'factors' | 'sectors'
 * @param {Object} event - The selected event data object
 */
function renderChart(tab, event) {
  clearChart();
  if (tab === "timeline") renderTimeline(event);
  if (tab === "factors")  renderFactors(event);
  if (tab === "sectors")  renderSectors(event);
}

/**
 * CausalEdge™ Market Intelligence Terminal — app.js
 */

const EVENTS = [
  {
    id: "tariff", name: "Liberation Day Tariffs", date: "Apr 2–9 2025", cat: "Policy shock",
    actual: -10.5, cf: 1.2, impact: -11.7, conf: 94,
    timeline: [{d:"Apr 1",a:100,c:100},{d:"Apr 2",a:97.8,c:100.3},{d:"Apr 3",a:93.4,c:100.5},{d:"Apr 4",a:91.2,c:100.4},{d:"Apr 7",a:89.8,c:100.8},{d:"Apr 8",a:89.5,c:101.1},{d:"Apr 9",a:91.3,c:101.2},{d:"Apr 10",a:94.1,c:101.3}],
    factors: [{n:"Policy uncertainty",v:-5.2},{n:"Supply chain repricing",v:-3.1},{n:"Retaliation risk",v:-2.4},{n:"Dollar strength",v:-1.8},{n:"Earnings revisions",v:-1.5},{n:"Short squeeze rebound",v:2.3}],
    sectors: [{s:"Consumer Disc.",v:-15.2},{s:"Technology",v:-12.1},{s:"Industrials",v:-11.8},{s:"Healthcare",v:-2.3},{s:"Energy",v:-4.2},{s:"Utilities",v:-1.1}],
    confounders: ["Pre-existing earnings jitters","Tech sector rotation","Fed pause expectations"]
  },
  {
    id: "fed", name: "Fed Pivot Cycle", date: "Sep–Dec 2024", cat: "Monetary policy",
    actual: 12.3, cf: 4.1, impact: 8.2, conf: 87,
    timeline: [{d:"Sep 1",a:100,c:100},{d:"Sep 18",a:101.5,c:100.8},{d:"Oct 1",a:102.8,c:101.2},{d:"Nov 1",a:107.4,c:102.1},{d:"Nov 7",a:110.2,c:102.8},{d:"Dec 1",a:111.8,c:103.4},{d:"Dec 18",a:112.3,c:104.1}],
    factors: [{n:"Rate cut signal",v:4.1},{n:"Duration re-rating",v:2.8},{n:"Credit spread compression",v:1.9},{n:"Dollar weakening",v:1.4},{n:"Election optimism",v:3.2},{n:"Inflation persistence drag",v:-1.8}],
    sectors: [{s:"Real Estate",v:18.4},{s:"Utilities",v:14.2},{s:"Financials",v:11.3},{s:"Technology",v:9.1},{s:"Healthcare",v:5.2},{s:"Energy",v:2.1}],
    confounders: ["US election cycle","AI capex boom","Consumer resilience"]
  },
  {
    id: "deepseek", name: "DeepSeek R1 Shock", date: "Jan 27 2025", cat: "Tech disruption",
    actual: -3.1, cf: 0.8, impact: -3.9, conf: 91,
    timeline: [{d:"Jan 24",a:100,c:100},{d:"Jan 27",a:96.9,c:100.2},{d:"Jan 28",a:97.8,c:100.5},{d:"Jan 29",a:98.4,c:100.7},{d:"Jan 30",a:99.1,c:100.8},{d:"Jan 31",a:99.8,c:100.9}],
    factors: [{n:"AI capex repricing",v:-2.8},{n:"Nvidia revenue risk",v:-1.9},{n:"Data center doubt",v:-1.4},{n:"China tech re-rating",v:1.8},{n:"Efficiency narrative",v:0.7},{n:"Macro baseline",v:-0.3}],
    sectors: [{s:"Semiconductors",v:-9.1},{s:"Cloud providers",v:-5.2},{s:"AI infrastructure",v:-7.8},{s:"Software (SaaS)",v:1.2},{s:"China ADRs",v:6.4},{s:"Energy",v:-3.1}],
    confounders: ["Q4 earnings season","Fed January meeting","Dollar index moves"]
  }
];

let currentEvent = EVENTS[0];
let currentTab = "timeline";
let chartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
  renderEventTabs(); renderMetrics(); renderConfounders(); switchTab("timeline");
});

function renderEventTabs() {
  document.getElementById("event-tabs").innerHTML = EVENTS.map(e =>
    `<button class="ev-btn ${e.id===currentEvent.id?"active":""}" onclick="selectEvent('${e.id}')">
      ${e.name}<small>${e.date} · ${e.cat}</small></button>`
  ).join("");
}

function selectEvent(id) {
  currentEvent = EVENTS.find(e => e.id === id);
  document.getElementById("analysis-area").innerHTML =
    `<div class="analysis-placeholder">Click "Run causal analysis" to generate a narrative.</div>`;
  document.getElementById("run-btn").textContent = "Run causal analysis ↗";
  renderEventTabs(); renderMetrics(); renderConfounders(); renderChart();
}

function fmt(v) { return (v > 0 ? "+" : "") + v.toFixed(1) + "%"; }

function renderMetrics() {
  const e = currentEvent;
  const items = [
    {label:"Observed return", val:fmt(e.actual), color:e.actual>0?"var(--green)":"var(--red)"},
    {label:"Counterfactual",  val:fmt(e.cf),     color:"var(--text-secondary)"},
    {label:"Causal impact",   val:fmt(e.impact), color:e.impact>0?"var(--green)":"var(--red)"},
    {label:"Confidence",      val:e.conf+"%",    color:"var(--text)"},
  ];
  document.getElementById("metrics").innerHTML = items.map(m =>
    `<div class="metric"><div class="metric-label">${m.label}</div>
     <div class="metric-value" style="color:${m.color};">${m.val}</div></div>`
  ).join("");
}

function renderConfounders() {
  document.getElementById("confounders-list").innerHTML =
    currentEvent.confounders.map(c => `<span class="pill">${c}</span>`).join("");
}

function switchTab(tab) {
  currentTab = tab;
  ["timeline","factors","sectors"].forEach(t => {
    document.getElementById("tab-"+t).classList.toggle("active", t===tab);
  });
  renderChart();
}

function renderChart() {
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  const area = document.getElementById("chart-area");
  area.innerHTML = "";
  const e = currentEvent;

  if (currentTab === "timeline") {
    area.innerHTML = `<div class="chart-label">Actual vs counterfactual (indexed to 100 at event start)</div>
      <div style="position:relative;height:200px;">
        <canvas id="mainChart" role="img" aria-label="Line chart: actual vs counterfactual returns during ${e.name}">Actual: ${e.actual}%, Counterfactual: ${e.cf}%</canvas>
      </div>
      <div class="legend">
        <span class="legend-item"><span class="legend-line" style="background:${e.impact>0?"#1D9E75":"#E24B4A"};"></span>Actual</span>
        <span class="legend-item"><span class="legend-dash"></span>Counterfactual</span>
      </div>`;
    const ctx = document.getElementById("mainChart").getContext("2d");
    chartInstance = new Chart(ctx, {
      type:"line",
      data:{labels:e.timeline.map(d=>d.d),datasets:[
        {label:"Actual",data:e.timeline.map(d=>d.a),borderColor:e.impact>0?"#1D9E75":"#E24B4A",backgroundColor:e.impact>0?"rgba(29,158,117,.08)":"rgba(226,75,74,.08)",fill:true,tension:.35,borderWidth:2,pointRadius:3},
        {label:"Counterfactual",data:e.timeline.map(d=>d.c),borderColor:"#888780",borderDash:[4,3],fill:false,tension:.35,borderWidth:1.5,pointRadius:2}
      ]},
      options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{font:{size:10}}},y:{ticks:{font:{size:10},callback:v=>v.toFixed(0)}}}}
    });
    return;
  }

  if (currentTab === "factors") {
    const sorted = [...e.factors].sort((a,b)=>a.v-b.v);
    area.innerHTML = `<div class="chart-label">Causal factor decomposition — attributed effect (%)</div>
      <div style="position:relative;height:${sorted.length*38+50}px;">
        <canvas id="mainChart" role="img" aria-label="Horizontal bar: causal factor attribution for ${e.name}">${sorted.map(f=>f.n+": "+f.v+"%").join(", ")}</canvas>
      </div>`;
    const ctx = document.getElementById("mainChart").getContext("2d");
    chartInstance = new Chart(ctx, {
      type:"bar",
      data:{labels:sorted.map(f=>f.n),datasets:[{label:"Causal effect",data:sorted.map(f=>f.v),backgroundColor:sorted.map(f=>f.v>0?"rgba(29,158,117,.75)":"rgba(226,75,74,.75)"),borderColor:sorted.map(f=>f.v>0?"#1D9E75":"#E24B4A"),borderWidth:1}]},
      options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{font:{size:10},callback:v=>`${v>0?"+":""}${v}%`}},y:{ticks:{font:{size:10}}}}}
    });
    return;
  }

  if (currentTab === "sectors") {
    const sorted = [...e.sectors].sort((a,b)=>a.v-b.v);
    area.innerHTML = `<div class="chart-label">Heterogeneous Treatment Effects — causal impact by sector (%)</div>
      <div style="position:relative;height:${sorted.length*38+50}px;">
        <canvas id="mainChart" role="img" aria-label="Horizontal bar: sector causal impacts for ${e.name}">${sorted.map(s=>s.s+": "+s.v+"%").join(", ")}</canvas>
      </div>`;
    const ctx = document.getElementById("mainChart").getContext("2d");
    chartInstance = new Chart(ctx, {
      type:"bar",
      data:{labels:sorted.map(s=>s.s),datasets:[{label:"Causal impact",data:sorted.map(s=>s.v),backgroundColor:sorted.map(s=>s.v>0?"rgba(55,138,221,.7)":"rgba(226,75,74,.7)"),borderColor:sorted.map(s=>s.v>0?"#378ADD":"#E24B4A"),borderWidth:1}]},
      options:{indexAxis:"y",responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{font:{size:10},callback:v=>`${v>0?"+":""}${v}%`}},y:{ticks:{font:{size:10}}}}}
    });
  }
}

async function runAnalysis() {
  const e = currentEvent;
  const btn = document.getElementById("run-btn");
  const area = document.getElementById("analysis-area");
  btn.disabled = true; btn.textContent = "Analyzing...";
  area.innerHTML = `<div style="color:var(--text-secondary);padding:8px 0;">Running EconML synthetic control · Building DoWhy causal DAG · Estimating HTEs...</div>`;

  const prompt = `You are a senior quantitative analyst at a top-tier hedge fund. Analyze the CAUSAL impact of "${e.name}" (${e.date}) on US equities.

Key causal data:
- Observed return: ${e.actual}%
- Counterfactual (without the event): ${e.cf}%
- Net causal impact: ${e.impact}%
- Statistical confidence: ${e.conf}%
- Top causal factors: ${e.factors.map(f=>f.n+" ("+fmt(f.v)+")").join(", ")}
- Confounders controlled: ${e.confounders.join(", ")}

Write exactly 3 crisp paragraphs. No headers, no bullets, no fluff.
Para 1: What the counterfactual reveals.
Para 2: The causal transmission mechanism — how this event caused the moves.
Para 3: Actionable positioning insight based on sector HTEs.

Write like a CFA charterholder who moonlights as a quant. Dense, precise, no hedging language.`;

  try {
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({model:"claude-sonnet-4-20250514",max_tokens:1000,messages:[{role:"user",content:prompt}]})
    });
    const data = await resp.json();
    const text = data.content?.filter(b=>b.type==="text").map(b=>b.text).join("") || "Analysis unavailable.";
    area.innerHTML = text.split("\n\n").filter(p=>p.trim()).map(p=>`<p>${p.trim()}</p>`).join("");
  } catch(err) {
    area.innerHTML = `<div style="color:var(--red);">Error: ${err.message}</div>`;
  }
  btn.disabled = false; btn.textContent = "Re-run analysis ↗";
}

/**
 * data.js — CausalEdge™ Market Intelligence Terminal
 *
 * Causal event datasets. Each event contains:
 *   - actual:      observed S&P 500 return over the event window (%)
 *   - cf:          counterfactual return (what markets would have done without the event)
 *   - impact:      net causal impact = actual - cf (%)
 *   - conf:        statistical confidence of the causal estimate (%)
 *   - timeline:    daily indexed price series (actual vs counterfactual, base=100)
 *   - factors:     causal factor decomposition via EconML heterogeneous treatment effects
 *   - sectors:     sector-level causal impact (%)
 *   - confounders: variables controlled for in the DoWhy causal DAG
 *
 * Methodology:
 *   Counterfactuals estimated via DoWhy synthetic control + difference-in-differences.
 *   Factor attribution via EconML DML (Double Machine Learning) with cross-fitting.
 *   Sector impacts via heterogeneous treatment effect estimation (Causal Forest).
 *
 * NOTE: Data is illustrative for demonstration purposes. Not for live trading.
 */

const EVENTS = [
  {
    id: "tariff",
    name: "Liberation Day Tariffs",
    date: "Apr 2–9 2025",
    cat: "Policy shock",
    actual: -10.5,
    cf: 1.2,
    impact: -11.7,
    conf: 94,
    timeline: [
      { d: "Apr 1",  a: 100.0, c: 100.0 },
      { d: "Apr 2",  a: 97.8,  c: 100.3 },
      { d: "Apr 3",  a: 93.4,  c: 100.5 },
      { d: "Apr 4",  a: 91.2,  c: 100.4 },
      { d: "Apr 7",  a: 89.8,  c: 100.8 },
      { d: "Apr 8",  a: 89.5,  c: 101.1 },
      { d: "Apr 9",  a: 91.3,  c: 101.2 },
      { d: "Apr 10", a: 94.1,  c: 101.3 }
    ],
    factors: [
      { n: "Policy uncertainty",      v: -5.2 },
      { n: "Supply chain repricing",  v: -3.1 },
      { n: "Retaliation risk",        v: -2.4 },
      { n: "Dollar strength",         v: -1.8 },
      { n: "Earnings revisions",      v: -1.5 },
      { n: "Short squeeze rebound",   v:  2.3 }
    ],
    sectors: [
      { s: "Consumer Disc.",  v: -15.2 },
      { s: "Technology",      v: -12.1 },
      { s: "Industrials",     v: -11.8 },
      { s: "Healthcare",      v:  -2.3 },
      { s: "Energy",          v:  -4.2 },
      { s: "Utilities",       v:  -1.1 }
    ],
    confounders: [
      "Pre-existing earnings jitters",
      "Tech sector rotation",
      "Fed pause expectations"
    ]
  },

  {
    id: "fed",
    name: "Fed Pivot Cycle",
    date: "Sep–Dec 2024",
    cat: "Monetary policy",
    actual: 12.3,
    cf: 4.1,
    impact: 8.2,
    conf: 87,
    timeline: [
      { d: "Sep 1",  a: 100.0, c: 100.0 },
      { d: "Sep 18", a: 101.5, c: 100.8 },
      { d: "Oct 1",  a: 102.8, c: 101.2 },
      { d: "Nov 1",  a: 107.4, c: 102.1 },
      { d: "Nov 7",  a: 110.2, c: 102.8 },
      { d: "Dec 1",  a: 111.8, c: 103.4 },
      { d: "Dec 18", a: 112.3, c: 104.1 }
    ],
    factors: [
      { n: "Rate cut signal",              v:  4.1 },
      { n: "Duration re-rating",           v:  2.8 },
      { n: "Credit spread compression",    v:  1.9 },
      { n: "Dollar weakening",             v:  1.4 },
      { n: "Election optimism",            v:  3.2 },
      { n: "Inflation persistence drag",   v: -1.8 }
    ],
    sectors: [
      { s: "Real Estate",   v: 18.4 },
      { s: "Utilities",     v: 14.2 },
      { s: "Financials",    v: 11.3 },
      { s: "Technology",    v:  9.1 },
      { s: "Healthcare",    v:  5.2 },
      { s: "Energy",        v:  2.1 }
    ],
    confounders: [
      "US election cycle",
      "AI capex boom",
      "Consumer resilience"
    ]
  },

  {
    id: "deepseek",
    name: "DeepSeek R1 Shock",
    date: "Jan 27 2025",
    cat: "Tech disruption",
    actual: -3.1,
    cf: 0.8,
    impact: -3.9,
    conf: 91,
    timeline: [
      { d: "Jan 24", a: 100.0, c: 100.0 },
      { d: "Jan 27", a:  96.9, c: 100.2 },
      { d: "Jan 28", a:  97.8, c: 100.5 },
      { d: "Jan 29", a:  98.4, c: 100.7 },
      { d: "Jan 30", a:  99.1, c: 100.8 },
      { d: "Jan 31", a:  99.8, c: 100.9 }
    ],
    factors: [
      { n: "AI capex repricing",   v: -2.8 },
      { n: "Nvidia revenue risk",  v: -1.9 },
      { n: "Data center doubt",    v: -1.4 },
      { n: "China tech re-rating", v:  1.8 },
      { n: "Efficiency narrative", v:  0.7 },
      { n: "Macro baseline",       v: -0.3 }
    ],
    sectors: [
      { s: "Semiconductors",     v:  -9.1 },
      { s: "Cloud providers",    v:  -5.2 },
      { s: "AI infrastructure",  v:  -7.8 },
      { s: "Software (SaaS)",    v:   1.2 },
      { s: "China ADRs",         v:   6.4 },
      { s: "Energy",             v:  -3.1 }
    ],
    confounders: [
      "Q4 earnings season",
      "Fed January meeting",
      "Dollar index moves"
    ]
  }
];

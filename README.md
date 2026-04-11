# CausalEdge™ Market Intelligence Terminal

A causal AI demo application for finance and quantitative market analysis firms,
built on the PyWhy ecosystem (EconML + DoWhy) concepts with a live Claude AI narrative engine.

## Project Structure

```
causaledge/
├── index.html    — Application shell and layout
├── styles.css    — Flat design system, dark mode, responsive grid
├── data.js       — Causal event datasets with methodology comments
├── charts.js     — Chart.js rendering module (timeline, factors, sectors)
└── app.js        — Application controller, state management, Claude API integration
```

## Features

- **3 real market events**: Liberation Day Tariffs (Apr 2025), Fed Pivot (Sep–Dec 2024), DeepSeek R1 Shock (Jan 2025)
- **Causal vs counterfactual**: DoWhy synthetic control shows what markets would have done without the event
- **Factor decomposition**: EconML DML (Double Machine Learning) attributes the causal impact to individual drivers
- **Sector HTEs**: Causal Forest heterogeneous treatment effects show which sectors were most causally impacted
- **Live AI analysis**: Claude AI generates a 3-paragraph quant-grade causal narrative on demand
- **Dark mode**: Full light/dark mode support via CSS variables

## Causal Methodology (Illustrated)

| Layer | Tool | What it does |
|-------|------|--------------|
| Causal graph | DoWhy | Defines DAG, identifies confounders, validates assumptions |
| Counterfactual | DoWhy synthetic control | Estimates what would have happened without the event |
| Factor attribution | EconML DML | Decomposes causal impact into individual drivers |
| Sector HTEs | EconML Causal Forest | Estimates heterogeneous effects across sectors |

## Setup

Open `index.html` in any modern browser. No build step required.

The Claude AI analysis panel calls `https://api.anthropic.com/v1/messages`.
An API key must be configured in your deployment environment.

## Data Note

All numerical data is illustrative for demonstration purposes.
Not for live trading or investment decisions.

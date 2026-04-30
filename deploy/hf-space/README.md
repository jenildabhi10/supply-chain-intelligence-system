---
title: Supply Chain Intelligence API
emoji: ⚓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Supply Chain Intelligence API

FastAPI backend powering the [Supply Chain Intelligence System](https://github.com/jenildabhi/supply-chain-intelligence-system) frontend.

Provides real-time port disruption risk scores, SHAP explanations, active signals, and a Groq-powered LLM agent for natural-language queries.

## Endpoints

- `GET /health` — liveness check
- `GET /ports` — list monitored U.S. ports
- `GET /risk` — risk summaries for all ports
- `GET /risk/{port_id}` — full risk card for one port
- `GET /alerts/{port_id}` — active signals
- `POST /explain` — LLM agent query
- `POST /simulate` — what-if risk scoring
- `GET /docs` — interactive OpenAPI docs

## Source

https://github.com/jenildabhi/supply-chain-intelligence-system

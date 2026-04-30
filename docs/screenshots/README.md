# Screenshot Capture Guide

Three screenshots embedded in the project README. Use a wide window
(~1440px), browser zoom 100%, and let the page fully load before capturing.

## Setup (run once)

```bash
# Terminal 1 — backend
uvicorn src.api.main:app --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

Open http://localhost:3000 and wait until the API status dot in the top-right
turns green (live).

## Shot list

### 1. `overview.png` — http://localhost:3000

- Capture the hero, KPI strip, the dark Leaflet map with port circles, and at
  least one "top risks" card on the right.
- Best result: scroll to top, capture viewport (no need to capture the whole
  page).
- **Aim for ~1440 × 900px**.

### 2. `port-detail.png` — http://localhost:3000/ports/la_lb

- Use **la_lb** (LA / Long Beach) — has the highest impact index, makes a
  better demo than a low-traffic port.
- Capture should show: gauge + 4 KPI tiles in the top section, plus the
  SHAP chart and signals list below.
- Scroll until both the gauge row AND the SHAP/signals row are visible
  in one viewport, then capture. Or capture two viewports and use the SHAP
  one.

### 3. `agent.png` — http://localhost:3000/agent

- Click one of the suggestion chips (e.g. "Which port has the highest
  disruption risk right now?") and wait for the LLM response.
- Capture should show: the user message bubble, the assistant reply (with
  the tool-call attribution at the bottom), and the input box.
- If Groq is rate-limited (429) the fallback response is also fine to
  capture — it shows graceful degradation.

## Capture tools

- **Windows:** `Win + Shift + S` (Snipping Tool) → save as PNG.
- Save into this folder with the exact filenames above.

## Optional: animated GIF

If you want a hero GIF instead of a static screenshot for the overview, use
[ScreenToGif](https://www.screentogif.com/) (free, Windows). Record ~5s of
hovering over the map markers, then save as `overview.gif` and update the
README to reference the GIF.

## After capturing

```bash
# verify the files are in the right place
ls docs/screenshots/
# expect: overview.png  port-detail.png  agent.png
```

The README image references will then resolve correctly on GitHub.

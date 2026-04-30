"""
Bootstrap — one-time setup to get the full pipeline running with real data.

Steps:
  1. Clear stale model registry (pytest left temp-dir paths in manifest.json)
  2. Ingest: NWS, USGS, FIRMS, GDELT, BTS  (each runs once)
  3. Build Gold feature table from Bronze data
  4. Train XGBoost + IsolationForest models  (n_trials=10 for speed)
  5. Print a summary of what was ingested and the model metrics

Run once before starting the dashboard:
    python bootstrap.py

After bootstrap, start the dashboard normally:
    python -m streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import structlog
from rich.console import Console
from rich.table import Table

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)  # quiet third-party libs

log     = structlog.get_logger(__name__)
console = Console()


def step(n: int, title: str) -> None:
    console.print(f"\n[bold cyan]Step {n}:[/bold cyan] {title}")


def ok(msg: str) -> None:
    console.print(f"  [green]OK[/green]   {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]WARN[/yellow] {msg}")


def err(msg: str) -> None:
    console.print(f"  [red]FAIL[/red] {msg}")


# ─── Step 1: Clear stale manifest ────────────────────────────────────────────

step(1, "Clearing stale model registry")

manifest_path = Path("models/manifest.json")
if manifest_path.exists():
    data = json.loads(manifest_path.read_text())
    champion_id = data.get("champion", "")
    versions    = data.get("versions", [])

    # Check if any pkl files actually exist at the recorded paths
    stale = True
    for v in versions:
        for _component, fpath in v.get("files", {}).items():
            if Path(fpath).exists():
                stale = False
                break

    if stale:
        manifest_path.write_text(json.dumps({"champion": None, "versions": []}, indent=2))
        ok("Cleared stale manifest (all model file paths were temp pytest dirs)")
    else:
        ok("Manifest looks valid — skipping reset")
else:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"champion": None, "versions": []}, indent=2))
    ok("Created fresh manifest")


# ─── Step 2: Ingest all sources ───────────────────────────────────────────────

step(2, "Running ingestion pipeline (NWS > USGS > FIRMS > GDELT > BTS)")

from config import settings
from src.ingestion.bts_client import BTSClient
from src.ingestion.firms_client import FIRMSClient
from src.ingestion.gdelt_client import GDELTClient
from src.ingestion.nws_client import NWSClient
from src.ingestion.usgs_client import USGSClient
from src.storage.lake import DataLake

lake = DataLake()

def _run_nws() -> int:
    client = NWSClient()
    alerts, run = client.fetch_alerts()
    client.close()
    if alerts:
        path = lake.write_bronze_json("nws", [a.model_dump(mode="json") for a in alerts], run.run_id)
        run.bronze_paths = [str(path)]
    lake.record_run(run)
    return run.records_stored


def _run_usgs() -> int:
    client = USGSClient()
    events, run = client.fetch_earthquakes()
    client.close()
    if events:
        path = lake.write_bronze_json("usgs", [e.model_dump(mode="json") for e in events], run.run_id)
        run.bronze_paths = [str(path)]
    lake.record_run(run)
    return run.records_stored


def _run_firms() -> int:
    if not settings.NASA_FIRMS_MAP_KEY:
        return -1
    client = FIRMSClient()
    fires, run = client.fetch_fires()
    client.close()
    if fires:
        path = lake.write_bronze_json("firms", [f.model_dump(mode="json") for f in fires], run.run_id)
        run.bronze_paths = [str(path)]
    lake.record_run(run)
    return run.records_stored


def _run_gdelt() -> int:
    import pandas as pd
    client = GDELTClient()
    events, run, _ = client.fetch_events()
    if events:
        df   = pd.DataFrame([e.model_dump(mode="json") for e in events])
        path = lake.write_bronze_parquet("gdelt", df, run.run_id)
        run.bronze_paths = [str(path)]
    lake.record_run(run)
    return run.records_stored


def _run_bts() -> int:
    import pandas as pd
    client = BTSClient()
    metrics, run = client.fetch_port_metrics()
    client.close()
    if metrics:
        df   = pd.DataFrame([m.model_dump(mode="json") for m in metrics])
        path = lake.write_bronze_parquet("bts", df, run.run_id)
        run.bronze_paths = [str(path)]
    lake.record_run(run)
    return run.records_stored


sources = [
    ("NWS weather alerts",   _run_nws),
    ("USGS earthquakes",     _run_usgs),
    ("NASA FIRMS fires",     _run_firms),
    ("GDELT news events",    _run_gdelt),
    ("BTS berthing metrics", _run_bts),
]

ingestion_results = {}
for label, fn in sources:
    try:
        t0 = time.time()
        n  = fn()
        elapsed = time.time() - t0
        if n == -1:
            warn(f"{label}: skipped (no API key)")
        elif n == 0:
            warn(f"{label}: connected but 0 records returned")
        else:
            ok(f"{label}: {n:,} records in {elapsed:.1f}s")
        ingestion_results[label] = n
    except Exception as exc:
        err(f"{label}: {exc}")
        ingestion_results[label] = 0

lake.close()


# ─── Step 3: Build Gold feature table ─────────────────────────────────────────

step(3, "Building Gold feature table (Bronze > feature engineering > Gold)")

bts_records = ingestion_results.get("BTS berthing metrics", 0)
if bts_records == 0:
    warn("BTS returned 0 records — Gold table may be empty or missing disruption labels.")
    warn("The model will still train but quality will be limited.")

try:
    from src.features.pipeline import run_pipeline
    from src.storage.lake import DataLake

    lake2 = DataLake()
    run_pipeline(lake2)
    lake2.close()

    gold_path = Path("data/gold/port_week_features.parquet")
    if gold_path.exists():
        import pandas as pd
        df = pd.read_parquet(gold_path)
        ok(f"Gold table: {len(df):,} rows × {len(df.columns)} columns across {df['port_id'].nunique()} ports")
        pos_rate = df["is_disruption_week"].mean() if "is_disruption_week" in df.columns else None
        if pos_rate is not None:
            ok(f"Disruption rate: {pos_rate:.1%}  (target for model training)")
    else:
        err("Gold table not created — check Bronze data above")
        sys.exit(1)

except Exception as exc:
    err(f"Feature pipeline failed: {exc}")
    raise


# ─── Step 4: Train models ─────────────────────────────────────────────────────

step(4, "Training XGBoost + IsolationForest (n_trials=10 for speed)")

try:
    from src.models.train_pipeline import run_training_pipeline

    t0     = time.time()
    result = run_training_pipeline(n_optuna_trials=10, use_mlflow=False)
    elapsed = time.time() - t0

    m = result["metrics"]
    ok(f"Trained in {elapsed:.0f}s  |  version: {result['version']}")
    ok(f"CV PR-AUC: {m.get('tuned_cv_pr_auc', m.get('cv_pr_auc', '?')):.4f}  "
       f"(baseline: {m.get('baseline_cv_pr_auc', '?')})")
    ok(f"Training rows: {m.get('n_train_rows', '?')}  |  "
       f"Positive rate: {m.get('pos_rate', '?'):.1%}" if isinstance(m.get('pos_rate'), float)
       else f"Training rows: {m.get('n_train_rows', '?')}")

except Exception as exc:
    err(f"Model training failed: {exc}")
    raise


# ─── Step 5: Summary ──────────────────────────────────────────────────────────

console.print("\n")
table = Table(title="Bootstrap Complete — System Ready", show_lines=True)
table.add_column("Component",  style="cyan")
table.add_column("Status",     style="green")

table.add_row("Bronze data",     "OK - Ingested")
table.add_row("Gold table",      "OK - Built")
table.add_row("ML model",        f"OK - Trained  (v {result['version']})")
table.add_row("Dashboard",       "python -m streamlit run src/dashboard/app.py")
table.add_row("API",             "uvicorn src.api.main:app --reload")

console.print(table)
console.print(
    "\n[bold green]Next:[/bold green] "
    "python -m streamlit run src/dashboard/app.py\n"
)

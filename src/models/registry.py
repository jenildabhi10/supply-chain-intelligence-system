"""
ModelRegistry — versioned model persistence with champion/challenger tracking.

Models are stored as pickle files in the `models/` directory.
A JSON manifest tracks all versions and designates one as "champion."

Champion/challenger pattern:
  - Every new trained model is saved as a "challenger."
  - If its CV PR-AUC beats the current champion, it is automatically promoted.
  - The champion is always what the scoring engine and API use for predictions.
  - You can manually promote/demote via registry.promote(version).

This is a simplified version of what MLflow Model Registry does — we implement
it ourselves to keep the system free and dependency-light for the portfolio demo.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

MODELS_DIR     = Path("models")
MANIFEST_PATH  = MODELS_DIR / "manifest.json"


# ─── Manifest helpers (module-level, used only as fallback) ───────────────────

def _load_manifest_from(path: Path) -> dict:
    if not path.exists():
        return {"champion": None, "versions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest_to(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


# ─── ModelRegistry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    File-based model registry with champion/challenger versioning.

    Usage:
        registry = ModelRegistry()

        # Save a new version
        version = registry.save(
            trainer          = trainer,
            anomaly_detector = detector,
            calibrator       = calibrator,
            metrics          = {"cv_pr_auc": 0.71},
        )

        # Load champion for inference
        bundle = registry.load_champion()
        proba  = bundle["trainer"].predict_proba(X)

        # Promote a specific version
        registry.promote(version)
    """

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        self._dir      = Path(models_dir)
        self._manifest = self._dir / "manifest.json"
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        trainer,
        anomaly_detector,
        metrics:   dict[str, Any],
        calibrator = None,
        explainer  = None,
    ) -> str:
        """
        Persist all model components and update the manifest.

        Returns:
            version string (e.g. "v20240415_120000")
        """
        version = f"v{datetime.utcnow():%Y%m%d_%H%M%S}"
        files   = {}

        # Pickle each component
        for name, obj in [
            ("trainer",          trainer),
            ("anomaly_detector", anomaly_detector),
            ("calibrator",       calibrator),
            ("explainer",        explainer),
        ]:
            if obj is not None:
                path = self._dir / f"{name}_{version}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                files[name] = str(path)
                log.debug("model_saved", component=name, path=str(path))

        # Update manifest
        manifest = _load_manifest_from(self._manifest)

        entry = {
            "version":     version,
            "trained_at":  datetime.utcnow().isoformat(),
            "metrics":     metrics,
            "files":       files,
            "feature_count": (
                len(trainer.feature_columns) if trainer.feature_columns else 0
            ),
        }
        manifest["versions"].append(entry)

        # Auto-promote if this version has better PR-AUC than current champion
        new_pr_auc  = metrics.get("tuned_cv_pr_auc", metrics.get("cv_pr_auc", 0.0))
        champ_entry = self._get_champion_entry(manifest)

        if champ_entry is None:
            # No champion yet — auto-promote
            manifest["champion"] = version
            log.info("champion_set", version=version, reason="first_model")
        elif new_pr_auc > champ_entry["metrics"].get("tuned_cv_pr_auc", champ_entry["metrics"].get("cv_pr_auc", 0.0)):
            manifest["champion"] = version
            log.info(
                "champion_promoted",
                version    = version,
                new_pr_auc = round(new_pr_auc, 4),
                old_pr_auc = round(champ_entry["metrics"].get("tuned_cv_pr_auc", 0.0), 4),
            )
        else:
            log.info(
                "challenger_saved",
                version    = version,
                pr_auc     = round(new_pr_auc, 4),
                champion   = manifest["champion"],
            )

        _save_manifest_to(self._manifest, manifest)
        return version

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_champion(self) -> dict[str, Any]:
        """
        Load the champion model bundle.

        Returns:
            Dict with keys: trainer, anomaly_detector, calibrator, explainer, version, metrics
            Components are None if not saved for that version.
        """
        manifest = _load_manifest_from(self._manifest)
        version  = manifest.get("champion")

        if not version:
            raise RuntimeError("No champion model. Train a model first.")

        return self.load_version(version)

    def load_version(self, version: str) -> dict[str, Any]:
        """Load a specific model version by version string."""
        manifest = _load_manifest_from(self._manifest)
        entry    = next((v for v in manifest["versions"] if v["version"] == version), None)

        if entry is None:
            raise ValueError(f"Version {version!r} not found in manifest.")

        bundle: dict[str, Any] = {"version": version, "metrics": entry["metrics"]}

        for name, path in entry.get("files", {}).items():
            try:
                with open(path, "rb") as f:
                    bundle[name] = pickle.load(f)
                log.debug("model_loaded", component=name, version=version)
            except FileNotFoundError:
                log.warning("model_file_missing", component=name, path=path)
                bundle[name] = None

        return bundle

    # ── Promote / demote ─────────────────────────────────────────────────────

    def promote(self, version: str) -> None:
        """Manually set the champion to a specific version."""
        manifest = _load_manifest_from(self._manifest)
        versions = [v["version"] for v in manifest["versions"]]

        if version not in versions:
            raise ValueError(f"Version {version!r} not found. Available: {versions}")

        manifest["champion"] = version
        _save_manifest_to(self._manifest, manifest)
        log.info("champion_manually_promoted", version=version)

    # ── List / audit ─────────────────────────────────────────────────────────

    def list_versions(self) -> pd.DataFrame:
        """Return a summary DataFrame of all saved model versions."""
        manifest = _load_manifest_from(self._manifest)
        champion = manifest.get("champion", "")
        rows = []
        for entry in manifest["versions"]:
            rows.append({
                "version":        entry["version"],
                "trained_at":     entry["trained_at"],
                "is_champion":    entry["version"] == champion,
                "cv_pr_auc":      entry["metrics"].get("tuned_cv_pr_auc", entry["metrics"].get("cv_pr_auc", None)),
                "n_train_rows":   entry["metrics"].get("n_train_rows", None),
                "feature_count":  entry.get("feature_count", None),
            })
        return pd.DataFrame(rows).sort_values("trained_at", ascending=False)

    def get_champion_version(self) -> str | None:
        return _load_manifest_from(self._manifest).get("champion")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_champion_entry(manifest: dict) -> dict | None:
        champ = manifest.get("champion")
        if not champ:
            return None
        return next((v for v in manifest["versions"] if v["version"] == champ), None)

"""Model training — fit, save, and log a classifier."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from models.evaluate import compute_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
MODELS_DIR  = ROOT / "models"
METRICS_DIR = ROOT / "reports" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ── Core training function ─────────────────────────────────────────────────────

def train(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str,
    model_filename: str | None = None,
    feature_names: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Fit `model` on train data, evaluate on test, persist artefacts.

    Parameters
    ----------
    model          : unfitted sklearn-compatible estimator
    X_train / y_train : training arrays
    X_test  / y_test  : held-out test arrays
    model_name     : human label used for metrics folder and JSON keys
    model_filename : fixed .pkl stem (e.g. "logistic") — omit for timestamped name
    feature_names  : optional list of feature names for logging
    params         : extra metadata to store alongside metrics

    Returns
    -------
    dict with fitted model, metrics, and saved file paths
    """
    # ── 1. Fit ─────────────────────────────────────────────────────────────────
    print(f"[train] Fitting {model_name} …")
    model.fit(X_train, y_train)

    # ── 2. Evaluate on both splits ─────────────────────────────────────────────
    train_metrics = compute_metrics(y_train, model.predict(X_train),
                                    model.predict_proba(X_train)[:, 1]
                                    if hasattr(model, "predict_proba") else None,
                                    prefix="train_")

    test_metrics  = compute_metrics(y_test,  model.predict(X_test),
                                    model.predict_proba(X_test)[:, 1]
                                    if hasattr(model, "predict_proba") else None,
                                    prefix="test_")

    metrics = {**train_metrics, **test_metrics}
    _print_metrics(model_name, metrics)

    # ── 3. Persist model ───────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_stem   = model_filename if model_filename else f"{model_name}_{timestamp}"
    model_path = MODELS_DIR / f"{pkl_stem}.pkl"
    joblib.dump(model, model_path)
    print(f"[train] Model saved -> {model_path}")

    # ── 4. Persist metrics JSON into per-model subdirectory ──────────────────────
    run_record = {
        "model_name":    model_name,
        "timestamp":     timestamp,
        "params":        params or {},
        "feature_names": feature_names or [],
        "metrics":       metrics,
        "model_path":    str(model_path),
    }
    model_metrics_dir = METRICS_DIR / f"{model_name}_metrics"
    model_metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = model_metrics_dir / f"{model_name}_{timestamp}.json"
    with open(metrics_path, "w") as f:
        json.dump(run_record, f, indent=2)
    print(f"[train] Metrics saved -> {metrics_path}")

    return {
        "model":        model,
        "metrics":      metrics,
        "model_path":   str(model_path),
        "metrics_path": str(metrics_path),
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{'-'*45}")
    print(f"  {model_name}")
    print(f"{'-'*45}")
    for k, v in metrics.items():
        print(f"  {k:<25}: {v:.4f}")
    print(f"{'-'*45}\n")


def load_model(path: str | Path) -> BaseEstimator:
    """Load a previously saved model from disk."""
    return joblib.load(path)


def list_saved_models() -> list[Path]:
    """Return all .pkl files in the models directory, newest first."""
    return sorted(MODELS_DIR.glob("*.pkl"), key=os.path.getmtime, reverse=True)

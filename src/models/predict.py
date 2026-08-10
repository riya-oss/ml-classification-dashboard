"""Load a saved model and run inference on new data."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from features.pipelines import build_pipeline, run_pipeline
from features.preprocess import prepare

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


# ── Load ───────────────────────────────────────────────────────────────────────

def load_model(path: str | Path) -> BaseEstimator:
    """Load a persisted model from a .pkl file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def latest_model(pattern: str = "*.pkl") -> Path:
    """Return the most recently saved model matching `pattern`."""
    candidates = sorted(MODELS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No models found in {MODELS_DIR} matching '{pattern}'")
    return candidates[0]


# ── Predict ────────────────────────────────────────────────────────────────────

def predict(
    model: BaseEstimator,
    X: np.ndarray,
) -> np.ndarray:
    """Return binary class predictions (0 / 1)."""
    return model.predict(X)


def predict_proba(
    model: BaseEstimator,
    X: np.ndarray,
) -> np.ndarray:
    """Return churn probability scores (class=1). Falls back to predict if unavailable."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        # Calibrate decision scores to [0, 1] range via sigmoid
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X).astype(float)


# ── End-to-end inference from raw DataFrame ───────────────────────────────────

def predict_from_raw(
    df_raw: pd.DataFrame,
    model: BaseEstimator,
    preprocessing_pipeline,
) -> pd.DataFrame:
    """
    Run full inference from a raw DataFrame (mirrors the training pipeline).

    Parameters
    ----------
    df_raw                  : raw input (same schema as training data)
    model                   : fitted classifier
    preprocessing_pipeline  : fitted sklearn Pipeline from pipelines.run_pipeline()

    Returns
    -------
    DataFrame with columns: churn_pred (0/1), churn_proba (float)
    """
    X, _ = prepare(df_raw)
    X_proc = preprocessing_pipeline.transform(X)

    preds  = predict(model, X_proc)
    probas = predict_proba(model, X_proc)

    return pd.DataFrame({
        "churn_pred":  preds,
        "churn_proba": probas.round(4),
    }, index=df_raw.index)

"""Shared utilities for the Streamlit dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features.pipelines import build_pipeline, _SCALE_COLS, OHE_COLS

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR      = _ROOT / "models"
METRICS_DIR     = _ROOT / "reports" / "metrics"
PROCESSED_PATH  = _ROOT / "data" / "processed" / "telco_processed.csv"
PIPELINE_PATH   = MODELS_DIR / "preprocessing_pipeline.pkl"

# Minimum columns that must be present in any uploaded inference file
REQUIRED_INPUT_COLS: list[str] = [
    "Tenure Months", "Monthly Charges", "Total Charges",
    "Partner", "Dependents", "Phone Service", "Paperless Billing", "Senior Citizen",
    "Gender", "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Payment Method",
]

# Maps display name → (pkl stem, metrics folder prefix)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "Logistic Regression": ("logistic",       "logistic_regression"),
    "Decision Tree":       ("decision_tree",  "decision_tree"),
    "KNN":                 ("knn",            "knn"),
    "Naive Bayes":         ("naive_bayes",    "naive_bayes"),
    "Random Forest":       ("random_forest",  "random_forest"),
    "XGBoost":             ("xgboost",        "xgboost"),
}

METRIC_LABELS = {
    "test_accuracy":  "Accuracy",
    "test_auc":       "AUC",
    "test_precision": "Precision",
    "test_recall":    "Recall",
    "test_f1":        "F1 Score",
    "test_mcc":       "MCC",
}


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model(display_name: str):
    """Load a persisted classifier .pkl for the given display name."""
    pkl_stem, _ = MODEL_REGISTRY[display_name]
    path = MODELS_DIR / f"{pkl_stem}.pkl"
    if not path.exists():
        st.error(f"Model file not found: {path}")
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner="Building preprocessing pipeline…")
def get_fitted_pipeline():
    """
    Return the fitted sklearn preprocessing Pipeline.
    Loads from disk if already saved; otherwise re-fits from processed training data
    and saves to models/preprocessing_pipeline.pkl for future calls.
    """
    if PIPELINE_PATH.exists():
        return joblib.load(PIPELINE_PATH)

    if not PROCESSED_PATH.exists():
        st.error(
            "Processed data not found. Please run `run_training.py` first to generate "
            "`data/processed/telco_processed.csv`."
        )
        return None

    df = pd.read_csv(PROCESSED_PATH)
    y = df["Churn Label"]
    X = df.drop(columns=["Churn Label"])

    scale_cols = [c for c in _SCALE_COLS if c in X.columns]
    ohe_cols   = [c for c in OHE_COLS    if c in X.columns]

    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = build_pipeline(scale_cols, ohe_cols)
    pipeline.fit(X_train)

    joblib.dump(pipeline, PIPELINE_PATH)
    return pipeline


@st.cache_data(show_spinner=False)
def load_processed_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (X_train, X_test, y_train, y_test) from the saved processed CSV."""
    df = pd.read_csv(PROCESSED_PATH)
    y  = df["Churn Label"]
    X  = df.drop(columns=["Churn Label"])
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ── Metrics helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_latest_metrics(display_name: str) -> dict:
    """Load the most recent metrics JSON for `display_name`."""
    _, metrics_prefix = MODEL_REGISTRY[display_name]
    folder = METRICS_DIR / f"{metrics_prefix}_metrics"
    jsons  = sorted(folder.glob("*.json"))
    if not jsons:
        return {}
    with open(jsons[-1]) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_all_model_metrics() -> pd.DataFrame:
    """Return a DataFrame with one row per model (latest run only)."""
    rows = []
    for display_name, (_, metrics_prefix) in MODEL_REGISTRY.items():
        folder = METRICS_DIR / f"{metrics_prefix}_metrics"
        jsons  = sorted(folder.glob("*.json"))
        if not jsons:
            continue
        with open(jsons[-1]) as f:
            data = json.load(f)
        row = {"Model": display_name}
        row.update(data.get("metrics", {}))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # sort by AUC descending
    if "test_auc" in df.columns:
        df = df.sort_values("test_auc", ascending=False).reset_index(drop=True)
    return df


# ── Inference helper ───────────────────────────────────────────────────────────

def run_inference(
    df_raw: pd.DataFrame,
    display_name: str,
) -> pd.DataFrame | None:
    """
    Full inference from a raw uploaded DataFrame.
    Returns DataFrame with churn_pred and churn_proba columns, or None on error.
    """
    from features.preprocess import prepare
    from models.predict import predict, predict_proba

    pipeline = get_fitted_pipeline()
    model    = load_model(display_name)
    if pipeline is None or model is None:
        return None

    try:
        X, _ = prepare(df_raw)
    except Exception as exc:
        # Target column may be missing — that's fine for inference
        try:
            from features.preprocess import clean_raw, engineer_features, encode_binary
            df_clean = clean_raw(df_raw)
            df_clean = engineer_features(df_clean)
            df_clean = encode_binary(df_clean)
            # Drop target if present
            if "Churn Label" in df_clean.columns:
                X = df_clean.drop(columns=["Churn Label"])
            else:
                X = df_clean
        except Exception as exc2:
            st.error(f"Preprocessing failed: {exc2}")
            return None

    X_proc = pipeline.transform(X)
    preds  = predict(model, X_proc)
    probas = predict_proba(model, X_proc)

    result = df_raw.copy()
    result["Churn Prediction"] = preds
    result["Churn Probability"] = probas.round(4)
    result["Prediction Label"] = result["Churn Prediction"].map({1: "Churned", 0: "Retained"})
    return result


@st.cache_data(show_spinner=False)
def get_test_predictions(display_name: str) -> tuple:
    """
    Return (y_test, y_pred, y_proba) on the held-out test split.
    Cached per model name so the page only computes this once per session.
    """
    pipeline = get_fitted_pipeline()
    model    = load_model(display_name)
    if pipeline is None or model is None:
        return None, None, None

    _, X_test, _, y_test = load_processed_split()
    X_proc = pipeline.transform(X_test)

    y_pred  = model.predict(X_proc)
    y_proba = (
        model.predict_proba(X_proc)[:, 1]
        if hasattr(model, "predict_proba")
        else model.predict(X_proc).astype(float)
    )
    return y_test.values, y_pred, y_proba

"""Page 3 — Deep-dive into a single model: metrics, confusion matrix, feature importance."""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parents[1]
_ROOT    = _APP_DIR.parent
for _p in [str(_APP_DIR), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import streamlit as st

from components.metrics import kpi_cards, model_params_expander, train_vs_test_table
from components.plots import (
    display_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from utils import (
    MODEL_REGISTRY,
    get_test_predictions,
    load_latest_metrics,
    load_model,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Insights",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Model Insights")
st.caption("In-depth view of a single model: saved metrics, live confusion matrix on the test split, and feature importances.")

# ── Model selector ─────────────────────────────────────────────────────────────
model_choice = st.selectbox(
    "Select a model to inspect",
    options=list(MODEL_REGISTRY.keys()),
    index=0,
)

# ── Load saved metrics ─────────────────────────────────────────────────────────
metrics_data = load_latest_metrics(model_choice)
if not metrics_data:
    st.warning(f"No saved metrics found for {model_choice}.")
    st.stop()

metrics = metrics_data.get("metrics", {})
params  = metrics_data.get("params",  {})

# ── Section 1: KPI Cards ───────────────────────────────────────────────────────
st.subheader("Test-set KPI Cards")
kpi_cards(metrics)

# ── Section 2: Train vs Test table ────────────────────────────────────────────
st.subheader("Train vs Test — Overfitting Check")
train_vs_test_table(metrics)

# ── Section 3: Hyperparameters ────────────────────────────────────────────────
model_params_expander(params)

st.divider()

# ── Section 4: Live Confusion Matrix + ROC from test split ────────────────────
st.subheader("Live Evaluation on Held-out Test Split")
st.caption(
    "Re-computed on the same 80/20 stratified split (seed=42) used during training."
)


y_test, y_pred, y_proba = get_test_predictions(model_choice)

if y_test is None:
    st.error("Could not compute live predictions. Check that processed data exists.")
else:
    col_cm, col_roc = st.columns(2)
    with col_cm:
        plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix — {model_choice}")
    with col_roc:
        plot_roc_curve(y_test, y_proba, model_name=model_choice)

    st.subheader("Classification Report")
    display_classification_report(y_test, y_pred)

st.divider()

# ── Section 5: Feature Importances ────────────────────────────────────────────
st.subheader("Feature Importances")

_TREE_MODELS = {"Decision Tree", "Random Forest", "XGBoost"}

if model_choice not in _TREE_MODELS:
    # For linear models, use absolute coefficient values if available
    model_obj = load_model(model_choice)
    if model_obj is not None and hasattr(model_obj, "coef_"):
        feature_names = metrics_data.get("feature_names", [])
        if feature_names:
            importances = np.abs(model_obj.coef_[0])
            top_n       = st.slider("Top N features", 5, min(30, len(feature_names)), 15)
            plot_feature_importance(importances, feature_names, top_n=top_n, model_name=model_choice)
            st.caption("Logistic Regression: absolute coefficient magnitudes (after scaling).")
        else:
            st.info("Feature names not recorded in metrics JSON for this run.")
    else:
        st.info(
            f"{model_choice} does not expose feature importances or coefficients "
            "(e.g. KNN, Naive Bayes). Select Decision Tree, Random Forest, or XGBoost "
            "for importance rankings."
        )
else:
    model_obj     = load_model(model_choice)
    feature_names = metrics_data.get("feature_names", [])

    if model_obj is not None and hasattr(model_obj, "feature_importances_") and feature_names:
        importances = model_obj.feature_importances_
        top_n       = st.slider("Top N features", 5, min(30, len(feature_names)), 20)
        plot_feature_importance(importances, feature_names, top_n=top_n, model_name=model_choice)

        # Importance table
        with st.expander("Full importance table"):
            imp_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
            imp_df["Importance"] = imp_df["Importance"].round(6)
            st.dataframe(imp_df, width="stretch", hide_index=True)
    else:
        st.info("Feature names or importances not available for this model.")

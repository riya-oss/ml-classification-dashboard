"""Reusable Streamlit metric display components."""

from __future__ import annotations

import streamlit as st


def kpi_cards(metrics: dict, prefix: str = "test_") -> None:
    """
    Render 6 KPI metric cards: Accuracy, AUC, Precision, Recall, F1, MCC.

    Parameters
    ----------
    metrics : dict from the metrics JSON (e.g. {"test_accuracy": 0.82, ...})
    prefix  : key prefix to look up  ("test_" or "train_")
    """
    definitions = [
        ("test_accuracy",  "Accuracy",  "📊", "Fraction of correct predictions"),
        ("test_auc",       "AUC",       "📈", "Area under the ROC curve"),
        ("test_precision", "Precision", "🎯", "True positives / predicted positives"),
        ("test_recall",    "Recall",    "🔍", "True positives / actual positives"),
        ("test_f1",        "F1 Score",  "⚖️", "Harmonic mean of precision & recall"),
        ("test_mcc",       "MCC",       "🧮", "Matthews Correlation Coefficient"),
    ]

    cols = st.columns(3)
    for i, (key, label, icon, help_text) in enumerate(definitions):
        value = metrics.get(key)
        display = f"{value:.4f}" if isinstance(value, float) else "N/A"
        with cols[i % 3]:
            st.metric(label=f"{icon} {label}", value=display, help=help_text)


def train_vs_test_table(metrics: dict) -> None:
    """Show a two-column table comparing train vs test metrics."""
    import pandas as pd

    keys = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]
    rows = []
    for k in keys:
        train_val = metrics.get(f"train_{k}")
        test_val  = metrics.get(f"test_{k}")
        gap = (
            round(float(train_val) - float(test_val), 4)
            if train_val is not None and test_val is not None
            else None
        )
        rows.append({
            "Metric":  k.upper() if k == "auc" else k.capitalize(),
            "Train":   f"{train_val:.4f}" if train_val is not None else "N/A",
            "Test":    f"{test_val:.4f}"  if test_val  is not None else "N/A",
            "Gap (Train − Test)": f"{gap:+.4f}" if gap is not None else "N/A",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def model_params_expander(params: dict) -> None:
    """Show model hyperparameters in a collapsed expander."""
    if not params:
        return
    with st.expander("Hyperparameters", expanded=False):
        import pandas as pd
        rows = [{"Parameter": k, "Value": str(v)} for k, v in params.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

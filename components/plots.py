"""Reusable Streamlit plot components using Matplotlib / Seaborn."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_curve,
)

_PALETTE = "#4C72B0"


# ── Confusion Matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    title: str = "Confusion Matrix",
) -> None:
    """Render a seaborn heatmap confusion matrix in Streamlit."""
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Retained (0)", "Churned (1)"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label",    fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── ROC Curve ─────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    model_name: str = "Model",
) -> None:
    """Render an ROC curve with AUC annotation."""
    from sklearn.metrics import roc_auc_score

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score   = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=_PALETTE, lw=2, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Classification Report Table ───────────────────────────────────────────────

def display_classification_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> None:
    """Parse sklearn classification_report and render as a styled DataFrame."""
    report_dict = classification_report(
        y_true, y_pred,
        target_names=["Retained (0)", "Churned (1)"],
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for label in ["Retained (0)", "Churned (1)", "macro avg", "weighted avg"]:
        if label not in report_dict:
            continue
        r = report_dict[label]
        rows.append({
            "Class":     label,
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1 Score":  round(r["f1-score"],  4),
            "Support":   int(r["support"]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


# ── Model Comparison Bar Chart ────────────────────────────────────────────────

def plot_model_comparison(df: pd.DataFrame, metric: str = "test_auc") -> None:
    """Horizontal bar chart comparing all models on a chosen metric."""
    if df.empty or metric not in df.columns:
        st.info("No data available for this chart.")
        return

    sorted_df = df.sort_values(metric, ascending=True)
    colors    = [_PALETTE if v == sorted_df[metric].max() else "#A8C6E8"
                 for v in sorted_df[metric]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(sorted_df["Model"], sorted_df[metric], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xlabel(metric.replace("test_", "").upper())
    ax.set_title(f"Model Comparison — {metric.replace('test_', '').capitalize()}", fontweight="bold")
    ax.set_xlim(0, min(sorted_df[metric].max() * 1.15, 1.0))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Feature Importance Bar ────────────────────────────────────────────────────

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    model_name: str = "Model",
) -> None:
    """Bar chart of top-N feature importances."""
    idx = np.argsort(importances)[-top_n:]
    top_names  = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.35)))
    ax.barh(top_names, top_values, color=_PALETTE, edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Churn Probability Histogram ───────────────────────────────────────────────

def plot_proba_histogram(
    y_proba: np.ndarray | pd.Series,
    title: str = "Predicted Churn Probability Distribution",
) -> None:
    """Histogram of predicted churn probabilities."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(y_proba, bins=30, color=_PALETTE, edgecolor="white", alpha=0.85)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="threshold = 0.5")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Count")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Multi-model ROC Overlay ───────────────────────────────────────────────────

def plot_roc_comparison(
    curves: list[dict],
) -> None:
    """
    Overlay ROC curves for multiple models on a single plot.

    Parameters
    ----------
    curves : list of dicts with keys 'name', 'fpr', 'tpr', 'auc'
    """
    if not curves:
        st.info("No curve data available.")
        return

    cmap    = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, c in enumerate(curves):
        ax.plot(
            c["fpr"], c["tpr"],
            color=cmap(i), lw=2,
            label=f"{c['name']}  (AUC = {c['auc']:.4f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curve Comparison", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Multi-model Feature Importance (tabbed) ───────────────────────────────────

def plot_multi_feature_importance(
    models_importance: list[dict],
    top_n: int = 10,
) -> None:
    """
    Render one tab per model, each with a top-N feature importance bar chart.

    Parameters
    ----------
    models_importance : list of dicts with keys 'name', 'importances', 'feature_names'
    top_n             : number of top features to show
    """
    if not models_importance:
        st.info("No feature importance data available for the selected models.")
        return

    tab_labels = [m["name"] for m in models_importance]
    tabs       = st.tabs(tab_labels)

    for tab, m in zip(tabs, models_importance):
        with tab:
            imp   = np.array(m["importances"])
            names = m["feature_names"]
            idx   = np.argsort(imp)[-top_n:]

            fig, ax = plt.subplots(figsize=(7, max(3.5, top_n * 0.32)))
            ax.barh(
                [names[i] for i in idx],
                imp[idx],
                color=_PALETTE,
                edgecolor="white",
            )
            ax.set_xlabel("Importance")
            ax.set_title(f"Top {top_n} Features — {m['name']}", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

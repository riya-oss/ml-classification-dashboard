"""
Machine Learning Classification Dashboard — Home Page

Entry point for the Streamlit multi-page app.
Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

from utils import MODEL_REGISTRY, load_all_model_metrics

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🤖 Machine Learning Classification Dashboard")
st.caption("IBM Telco Customer Churn · Binary Classification · 6 Models")

st.divider()

# ── Project overview ───────────────────────────────────────────────────────────
col_about, col_nav = st.columns([2, 1])

with col_about:
    st.subheader("Project Overview")
    st.markdown(
        """
        This dashboard provides an end-to-end view of a **binary churn-prediction** pipeline
        trained on the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
        dataset (~7,000 customers, 20 features).

        **Models trained:**
        - Logistic Regression
        - Decision Tree
        - K-Nearest Neighbours (KNN)
        - Gaussian Naive Bayes
        - Random Forest
        - XGBoost

        **Preprocessing pipeline:**  
        StandardScaler (numeric) → OneHotEncoder (categorical) · Class imbalance handled via `class_weight="balanced"`.

        **Target variable:**  `Churn Label` → **Yes (1) / No (0)**
        """
    )

with col_nav:
    st.subheader("Navigation")
    st.markdown(
        """
        | Page | What you can do |
        |------|----------------|
        | 📊 **Model Comparison** | Compare all 6 models side-by-side |
        | 🔮 **Predictions** | Upload CSV → select model → get predictions |
        | 🔬 **Model Insights** | Deep-dive: confusion matrix, ROC, feature importances |

        Use the **sidebar** to navigate between pages.
        """
    )

st.divider()

# ── Quick stats ────────────────────────────────────────────────────────────────
st.subheader("Quick Stats — Best Saved Metrics")

df = load_all_model_metrics()

if df.empty:
    st.warning(
        "No metrics found. Run `src/models/run_training.py` to train models and "
        "generate metric files."
    )
else:
    best_row    = df.iloc[0]
    best_model  = best_row["Model"]
    best_auc    = best_row.get("test_auc",       "N/A")
    best_acc    = best_row.get("test_accuracy",  "N/A")
    best_f1     = best_row.get("test_f1",        "N/A")
    best_recall = best_row.get("test_recall",    "N/A")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏆 Best Model",  best_model)
    c2.metric("📈 Best AUC",    f"{best_auc:.4f}"    if isinstance(best_auc,    float) else best_auc)
    c3.metric("📊 Accuracy",    f"{best_acc:.4f}"    if isinstance(best_acc,    float) else best_acc)
    c4.metric("⚖️ F1 Score",   f"{best_f1:.4f}"     if isinstance(best_f1,     float) else best_f1)
    c5.metric("🔍 Recall",      f"{best_recall:.4f}" if isinstance(best_recall, float) else best_recall)

    st.divider()

    st.subheader("Leaderboard (sorted by AUC)")
    from utils import METRIC_LABELS
    display_cols = ["Model"] + [k for k in METRIC_LABELS if k in df.columns]
    display_df   = df[display_cols].rename(columns=METRIC_LABELS)
    metric_value_cols = [v for v in METRIC_LABELS.values() if v in display_df.columns]
    st.dataframe(
        display_df.style.format({c: "{:.4f}" for c in metric_value_cols}),
        width="stretch",
        hide_index=True,
    )

st.divider()

# ── User journey guide ─────────────────────────────────────────────────────────
st.subheader("End-User Journey")
st.markdown(
    """
    ```
    Open App
         ↓
    View Project Overview  (this page)
         ↓
    Upload Test CSV        (Predictions page)
         ↓
    Select Model           (Predictions page)
         ↓
    Run Predictions        (Predictions page)
         ↓
    View Metrics           (Predictions page — if labels present)
         ↓
    View Confusion Matrix  (Predictions page  /  Model Insights)
         ↓
    View Classification Report
         ↓
    Download Results CSV
    ```
    """
)

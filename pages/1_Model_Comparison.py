"""Page 1 — Model Comparison: leaderboard, metrics chart, ROC overlay, feature importance, winner."""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parents[1]   # ml-assignment-2/
_APP_DIR = _ROOT / "streamlit_app"               # utils + components
for _p in [str(_APP_DIR), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

from components.plots import (
    plot_model_comparison,
    plot_multi_feature_importance,
    plot_roc_comparison,
    plot_roc_curve,
)
from utils import (
    METRIC_LABELS,
    MODEL_REGISTRY,
    get_test_predictions,
    load_all_model_metrics,
    load_latest_metrics,
    load_model,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Comparison")
st.caption("Compare classifiers side-by-side. Select one model for a focused view, or multiple for competitive analysis.")

# ── Load data ──────────────────────────────────────────────────────────────────
df_all = load_all_model_metrics()

if df_all.empty:
    st.warning("No metrics found. Run `src/models/run_training.py` to train models first.")
    st.stop()

# ── Model selector ─────────────────────────────────────────────────────────────
all_model_names = list(MODEL_REGISTRY.keys())
selected_models = st.multiselect(
    "Select models to compare",
    options=all_model_names,
    default=all_model_names,
    help="Choose one model for a focused view, or multiple for head-to-head comparison.",
)

if not selected_models:
    st.warning("Select at least one model to continue.")
    st.stop()

df       = df_all[df_all["Model"].isin(selected_models)].reset_index(drop=True)
is_multi = len(selected_models) > 1

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Performance Leaderboard
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🏅 Performance Leaderboard")
st.caption("Sorted by AUC (descending). Green cells mark the best value per metric.")

_MEDAL = {1: "🥇", 2: "🥈", 3: "🥉"}

leaderboard = df.reset_index(drop=True).copy()
leaderboard.insert(0, "Rank", [
    f"{_MEDAL.get(i + 1, '')} {i + 1}" for i in range(len(leaderboard))
])

display_cols      = ["Rank", "Model"] + [k for k in METRIC_LABELS if k in leaderboard.columns]
lb_display        = leaderboard[display_cols].rename(columns=METRIC_LABELS)
metric_cols_rn    = [METRIC_LABELS[k] for k in METRIC_LABELS if k in df.columns]

def _highlight_max(col):
    if col.name not in metric_cols_rn:
        return [""] * len(col)
    numeric = pd.to_numeric(col, errors="coerce")
    is_max  = numeric == numeric.max()
    return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_max]

styled_lb = (
    lb_display.style
    .apply(_highlight_max)
    .format({c: "{:.4f}" for c in metric_cols_rn if c in lb_display.columns})
)
st.dataframe(styled_lb, width="stretch", hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Metrics Comparison Chart
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📈 Metrics Comparison Chart")

col_sel, col_bar = st.columns([1, 3])
with col_sel:
    metric_choice = st.selectbox(
        "Metric",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda k: METRIC_LABELS[k],
        index=1,
    )
with col_bar:
    plot_model_comparison(df, metric=metric_choice)

# All-metrics grouped bar
st.subheader("All Metrics — Side-by-side")
metrics_to_plot = [k for k in METRIC_LABELS if k in df.columns]
x       = np.arange(len(metrics_to_plot))
n       = len(df)
w       = min(0.13, 0.7 / max(n, 1))
offsets = np.linspace(-(n - 1) * w / 2, (n - 1) * w / 2, n) if n > 1 else [0]
cmap    = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(11, 5))
for i, (_, row) in enumerate(df.iterrows()):
    vals = [row.get(m, 0) for m in metrics_to_plot]
    ax.bar(x + offsets[i], vals, width=w, label=row["Model"], color=cmap(i), alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels([METRIC_LABELS[m] for m in metrics_to_plot])
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Performance — All Test Metrics", fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📉 ROC Curves")

with st.spinner("Computing ROC curves on the held-out test split…"):
    curves = []
    for name in selected_models:
        y_test, _, y_proba = get_test_predictions(name)
        if y_test is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val      = round(roc_auc_score(y_test, y_proba), 4)
        curves.append({"name": name, "fpr": fpr, "tpr": tpr, "auc": auc_val})

if not curves:
    st.error("Could not compute ROC curves — ensure processed data is available.")
elif is_multi:
    plot_roc_comparison(curves)
else:
    y_test, _, y_proba = get_test_predictions(selected_models[0])
    plot_roc_curve(y_test, y_proba, model_name=selected_models[0])

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Feature Importances (multi-model only)
# ══════════════════════════════════════════════════════════════════════════════
if is_multi:
    st.subheader("🔍 Feature Importances — Top 10 per Model")
    st.caption("Tree models use Gini importance; Logistic Regression uses |coefficient| magnitude.")

    importance_data = []
    for name in selected_models:
        m_data    = load_latest_metrics(name)
        feat_names = m_data.get("feature_names", [])
        model_obj  = load_model(name)
        if model_obj is None or not feat_names:
            continue
        if hasattr(model_obj, "feature_importances_"):
            importance_data.append({
                "name":          name,
                "importances":   model_obj.feature_importances_.tolist(),
                "feature_names": feat_names,
            })
        elif hasattr(model_obj, "coef_"):
            importance_data.append({
                "name":          name,
                "importances":   np.abs(model_obj.coef_[0]).tolist(),
                "feature_names": feat_names,
            })

    if importance_data:
        plot_multi_feature_importance(importance_data, top_n=10)
    else:
        st.info("None of the selected models expose feature importances (e.g. KNN / Naive Bayes only).")

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Winner Announcement (multi-model only)
# ══════════════════════════════════════════════════════════════════════════════
if is_multi and not df.empty:
    st.subheader("🏆 Winner Announcement")

    best        = df.iloc[0]
    fmt         = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 14px; padding: 28px 36px; color: white;
            text-align: center; box-shadow: 0 4px 20px rgba(118,75,162,0.35);">
            <h1 style="margin:0; font-size:2.4rem;">🏆 {best['Model']}</h1>
            <p style="margin:8px 0 0; font-size:1.05rem; opacity:0.85;">
                Best overall model across {len(selected_models)} classifiers (ranked by AUC)
            </p>
            <div style="display:flex; justify-content:center; gap:40px;
                        margin-top:20px; flex-wrap:wrap;">
                <div><div style="font-size:1.6rem;font-weight:700;">{fmt(best.get("test_auc","N/A"))}</div>
                     <div style="font-size:.85rem;opacity:.8;">AUC</div></div>
                <div><div style="font-size:1.6rem;font-weight:700;">{fmt(best.get("test_accuracy","N/A"))}</div>
                     <div style="font-size:.85rem;opacity:.8;">Accuracy</div></div>
                <div><div style="font-size:1.6rem;font-weight:700;">{fmt(best.get("test_f1","N/A"))}</div>
                     <div style="font-size:.85rem;opacity:.8;">F1 Score</div></div>
                <div><div style="font-size:1.6rem;font-weight:700;">{fmt(best.get("test_recall","N/A"))}</div>
                     <div style="font-size:.85rem;opacity:.8;">Recall</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Overfitting Check
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⚠️ Train vs Test Gap (Overfitting Check)")
st.caption("A large positive gap means the model memorised training data.")

gap_rows = []
for _, row in df.iterrows():
    for metric in ["accuracy", "auc", "f1"]:
        tk, vk = f"train_{metric}", f"test_{metric}"
        if tk in row and vk in row:
            gap = round(float(row[tk]) - float(row[vk]), 4)
            gap_rows.append({
                "Model":  row["Model"],
                "Metric": metric.upper() if metric == "auc" else metric.capitalize(),
                "Train":  round(float(row[tk]), 4),
                "Test":   round(float(row[vk]), 4),
                "Gap (Train − Test)": f"{gap:+.4f}",
            })

if gap_rows:
    st.dataframe(pd.DataFrame(gap_rows), width="stretch", hide_index=True)

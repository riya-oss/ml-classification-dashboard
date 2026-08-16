"""Page 2 — Upload CSV, select model, run predictions, download results."""

from __future__ import annotations

import io
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

from components.metrics import kpi_cards
from components.plots import (
    display_classification_report,
    plot_confusion_matrix,
    plot_proba_histogram,
    plot_roc_curve,
)
from utils import MODEL_REGISTRY, REQUIRED_INPUT_COLS, run_inference

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictions",
    page_icon="🔮",
    layout="wide",
)

st.title("🔮 Predictions")
st.caption(
    "Upload a customer dataset (same schema as the Telco training data), "
    "pick a model, and run predictions."
)

# ── CSS for the test-results table ───────────────────────────────────────────
st.markdown(
    """
    <style>
    a { text-decoration: none; color: #464feb; }
    .test-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    .test-table th, .test-table td { border: 1px solid #e6e6e6; padding: 8px 14px; }
    .test-table th { background-color: #f5f5f5; font-weight: 600; }
    .test-table tr:hover td { background-color: #fafafa; }
    .pass { color: #28a745; font-weight: bold; }
    .fail { color: #dc3545; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.subheader("Step 1 — Upload Dataset")
uploaded = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    help="For assignment grading, upload only `test_data.csv` (same schema as the IBM Telco Customer Churn dataset).",
)

if uploaded is None:
    st.info(
        "No file uploaded yet. For assignment compliance, upload `test_data.csv` "
        "from the repository root."
    )
    st.stop()

if uploaded.name != "test_data.csv":
    st.warning(
        "Assignment note: expected upload file is `test_data.csv`. "
        "Continuing with the provided CSV."
    )

# ── Upload validation + Testing section ───────────────────────────────────────
st.subheader("🧪 Upload Validation")

_tests: list[tuple[str, bool, str]] = []  # (test name, passed, detail)

# Test 1 — file is a valid CSV (non-empty bytes)
_raw_bytes = uploaded.getvalue()
_t1_pass   = len(_raw_bytes) > 0
_tests.append(("File is not empty", _t1_pass, f"{len(_raw_bytes):,} bytes" if _t1_pass else "0 bytes — file is empty"))

# Attempt to parse
_parse_ok = False
df_raw    = None
_parse_err = ""
try:
    import io as _io
    df_raw    = pd.read_csv(_io.BytesIO(_raw_bytes))
    _parse_ok = True
except Exception as _e:
    _parse_err = str(_e)
_tests.append(("File parses as valid CSV", _parse_ok, "Parsed successfully" if _parse_ok else f"Parse error: {_parse_err}"))

# Test 3 — not completely empty after parse
_rows_ok = _parse_ok and len(df_raw) > 0
_tests.append(("File contains at least one row", _rows_ok, f"{len(df_raw):,} data rows" if _rows_ok else "No data rows found"))

# Test 4 — required columns present
if _parse_ok:
    _missing = [c for c in REQUIRED_INPUT_COLS if c not in df_raw.columns]
    _cols_ok = len(_missing) == 0
    _col_detail = (
        "All required columns present"
        if _cols_ok
        else f"{len(_missing)} missing: {', '.join(_missing[:5])}{'…' if len(_missing) > 5 else ''}"
    )
else:
    _cols_ok, _col_detail = False, "Cannot check — file did not parse"
_tests.append(("Required columns present", _cols_ok, _col_detail))

# Test 5 — no fully-null critical numeric columns
if _parse_ok and _rows_ok:
    _critical = ["Tenure Months", "Monthly Charges", "Total Charges"]
    _null_cols = [c for c in _critical if c in df_raw.columns and df_raw[c].isnull().all()]
    _null_ok   = len(_null_cols) == 0
    _null_detail = "No fully-null numeric columns" if _null_ok else f"All-null columns: {', '.join(_null_cols)}"
else:
    _null_ok, _null_detail = False if (_parse_ok and not _rows_ok) else True, "Skipped"
_tests.append(("No fully-null critical columns", _null_ok, _null_detail))

# Render test table
_rows_html = "".join(
    f"<tr><td>{name}</td>"
    f"<td class='{'pass' if passed else 'fail'}'>"
    f"{'✅ Pass' if passed else '❌ Fail'}</td>"
    f"<td>{detail}</td></tr>"
    for name, passed, detail in _tests
)
st.markdown(
    f"""
    <table class="test-table">
      <thead><tr><th>Test</th><th>Status</th><th>Detail</th></tr></thead>
      <tbody>{_rows_html}</tbody>
    </table>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

# Block further processing if critical tests failed
_critical_pass = _t1_pass and _parse_ok and _rows_ok
if not _critical_pass:
    st.error("⛔ Fix the issues above before running predictions.")
    st.stop()

if not _cols_ok:
    st.error(
        f"⚠️ {len(_missing)} required column(s) are missing. "
        "Predictions are blocked to avoid invalid evaluation. "
        f"Missing: `{'`, `'.join(_missing)}`"
    )
    st.stop()

# ── Step 2: Dataset Preview ────────────────────────────────────────────────────
st.subheader("Step 2 — Dataset Preview")
st.caption(f"{df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
st.dataframe(df_raw.head(100), width="stretch")

with st.expander("Column types & null counts"):
    info_df = pd.DataFrame({
        "Column":   df_raw.columns,
        "Dtype":    [str(d) for d in df_raw.dtypes],
        "Non-null": df_raw.notnull().sum().values,
        "Nulls":    df_raw.isnull().sum().values,
    })
    st.dataframe(info_df, width="stretch", hide_index=True)

# ── Step 3: Model selection ────────────────────────────────────────────────────
st.subheader("Step 3 — Select Model")
model_choice = st.selectbox(
    "Choose a classifier",
    options=list(MODEL_REGISTRY.keys()),
    index=0,
)

# ── Step 4: Run predictions ────────────────────────────────────────────────────
st.subheader("Step 4 — Run Predictions")
run_btn = st.button("▶ Run Predictions", type="primary", width="stretch")

if not run_btn:
    st.stop()

with st.spinner(f"Running {model_choice} predictions…"):
    results = run_inference(df_raw, model_choice)

if results is None:
    st.error("Prediction failed. Check the error messages above.")
    st.stop()

# ── Step 5: Results preview ────────────────────────────────────────────────────
st.success(f"Predictions complete — {len(results):,} rows processed.")
st.subheader("Step 5 — Prediction Results")

pred_cols = ["Churn Prediction", "Churn Probability", "Prediction Label"]
all_cols  = pred_cols + [c for c in results.columns if c not in pred_cols]
st.dataframe(results[all_cols].head(200), width="stretch")

# Churn distribution summary
churn_count    = int(results["Churn Prediction"].sum())
retain_count   = len(results) - churn_count
churn_pct      = churn_count / len(results) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Total Customers",  f"{len(results):,}")
c2.metric("Predicted Churned", f"{churn_count:,}",  f"{churn_pct:.1f}%")
c3.metric("Predicted Retained", f"{retain_count:,}", f"{100 - churn_pct:.1f}%")

plot_proba_histogram(results["Churn Probability"].values)

# ── Step 6: Evaluation (if target column present) ─────────────────────────────
TARGET_COL = "Churn Label"
has_labels = TARGET_COL in df_raw.columns

if has_labels:
    st.subheader("Step 6 — Evaluation Metrics")
    st.caption("Target column detected — computing evaluation metrics on this dataset.")

    y_true_raw = df_raw[TARGET_COL]
    # Normalize labels by value so pandas object, string, and numeric dtypes work.
    y_true_text = y_true_raw.astype("string").str.strip().str.lower()
    y_true = y_true_text.map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if y_true.isna().any():
        invalid_labels = sorted(y_true_raw[y_true.isna()].astype(str).unique())
        st.error(
            "Unsupported `Churn Label` values: "
            f"{', '.join(invalid_labels[:10])}. Use only Yes/No or 1/0."
        )
        st.stop()
    y_true = y_true.astype(int)

    y_pred  = results["Churn Prediction"].values
    y_proba = results["Churn Probability"].values

    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
    )

    live_metrics = {
        "test_accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "test_auc":       round(roc_auc_score(y_true, y_proba), 4),
        "test_precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "test_recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "test_f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "test_mcc":       round(matthews_corrcoef(y_true, y_pred), 4),
    }
    kpi_cards(live_metrics)

    st.subheader("Step 7 — Confusion Matrix")
    col_cm, col_roc = st.columns(2)
    with col_cm:
        plot_confusion_matrix(y_true, y_pred, title=f"Confusion Matrix — {model_choice}")
    with col_roc:
        plot_roc_curve(y_true, y_proba, model_name=model_choice)

    st.subheader("Step 8 — Classification Report")
    display_classification_report(y_true, y_pred)
else:
    st.info(
        "No `Churn Label` column found in the uploaded file — "
        "evaluation metrics and confusion matrix are not available. "
        "Add a `Churn Label` column (Yes/No or 1/0) to enable them."
    )

if has_labels:
    st.subheader("Step 9 — Cross-Model Results on Uploaded Test Data")
    st.caption(
        "Assignment check: results of different models on the uploaded test data."
    )

    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
    )

    compare_rows = []
    for model_name in MODEL_REGISTRY.keys():
        model_results = run_inference(df_raw.copy(), model_name)
        if model_results is None:
            continue

        m_pred = model_results["Churn Prediction"].values
        m_proba = model_results["Churn Probability"].values
        compare_rows.append({
            "Model": model_name,
            "Accuracy": round(accuracy_score(y_true, m_pred), 4),
            "AUC": round(roc_auc_score(y_true, m_proba), 4),
            "Precision": round(precision_score(y_true, m_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_true, m_pred, zero_division=0), 4),
            "F1": round(f1_score(y_true, m_pred, zero_division=0), 4),
            "MCC": round(matthews_corrcoef(y_true, m_pred), 4),
        })

    if compare_rows:
        compare_df = pd.DataFrame(compare_rows).sort_values("AUC", ascending=False)
        st.dataframe(compare_df, width="stretch", hide_index=True)
    else:
        st.warning("Could not compute cross-model comparison on uploaded test data.")

# ── Step 10: Download ──────────────────────────────────────────────────────────
st.subheader("Download Results")

csv_buffer = io.StringIO()
results.to_csv(csv_buffer, index=False)

st.download_button(
    label="⬇ Download Predictions CSV",
    data=csv_buffer.getvalue(),
    file_name=f"predictions_{model_choice.lower().replace(' ', '_')}.csv",
    mime="text/csv",
    width="stretch",
)

# Streamlit Application Architecture

## Entry Point

```
streamlit run app.py
```

`app.py` lives at the **project root**. Streamlit automatically discovers the `pages/` directory (also at root) and builds the sidebar navigation from it.

---

## Directory Layout

```
ml-classification-dashboard/            ← project root
│
├── app.py                              ← Home page + Streamlit entry point
│
├── pages/                              ← Streamlit multi-page routing
│   ├── 1_Model_Comparison.py           ← Page 1 (sidebar order from filename prefix)
│   ├── 2_Predictions.py                ← Page 2
│   └── 3_Model_Insights.py             ← Page 3
│
├── utils.py                            ← Loaders, inference, validation helpers
└── components/
       ├── metrics.py                      ← KPI cards, train/test table
       └── plots.py                        ← Matplotlib/Seaborn chart functions
```

---

## Path Resolution

Every file in `pages/` adds two directories to `sys.path` at startup:

```python
_ROOT = Path(__file__).resolve().parents[1]   # ml-classification-dashboard/
for _p in [str(_ROOT), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

This means all pages can do bare imports:

```python
from utils import load_all_model_metrics    # resolves to root utils.py
from components.plots import plot_roc_curve  # resolves to root components/plots.py
from features.preprocess import prepare     # resolves to src/features/preprocess.py
```

---

## Pages

### 🏠 Home (`app.py`)

| Element | Detail |
|---|---|
| Project overview | Markdown description of dataset, models, and pipeline |
| Quick-stats KPI row | Best Model · AUC · Accuracy · F1 · Recall from saved JSON metrics |
| Leaderboard table | All 6 models ranked by AUC, styled with `pandas.Styler` |
| User journey guide | Step-by-step flow diagram |

---

### 📊 Model Comparison (`pages/1_Model_Comparison.py`)

Controlled by a `st.multiselect` that filters every section below it.

| Section | Single model | Multiple models |
|---|---|---|
| **Leaderboard** | Shown (rank + medals) | Shown |
| **Metrics bar chart** | Shown (one bar) | Shown (grouped bars) |
| **ROC curve** | Single ROC plot | Overlay of all selected curves |
| **Feature importances** | Hidden | Tabbed top-10 per model |
| **Winner announcement** | Hidden | Gradient card for best AUC model |
| **Overfitting table** | Shown | Shown |

ROC curves and predictions are computed live on the held-out test split (1,409 rows) via `get_test_predictions()` which is `@st.cache_data` per model name.

---

### 🔮 Predictions (`pages/2_Predictions.py`)

Upload → Validate → Preview → Select Model → Predict → Evaluate → Download

**Upload validation (5 automated tests):**

| # | Test | Blocks on fail? |
|---|---|---|
| 1 | File is not empty | Yes |
| 2 | File parses as valid CSV | Yes |
| 3 | File contains at least one data row | Yes |
| 4 | All 19 required columns present | Warning only |
| 5 | No fully-null critical numeric columns | Warning only |

Results are rendered as a styled HTML table injected via `st.markdown(unsafe_allow_html=True)`.

**Evaluation (conditional):** If the uploaded file contains a `Churn Label` column, live metrics (Accuracy, AUC, Precision, Recall, F1, MCC), confusion matrix, ROC curve, and classification report are computed and displayed. Without the label column, only predictions + probability histogram are shown.

---

### 🔬 Model Insights (`pages/3_Model_Insights.py`)

Single-model deep-dive, driven by a `st.selectbox`.

| Section | Source |
|---|---|
| KPI cards (6 metrics) | Saved metrics JSON (`reports/metrics/`) |
| Train vs Test gap table | Saved metrics JSON |
| Hyperparameters expander | Saved metrics JSON |
| Live confusion matrix | Recomputed from test split via `get_test_predictions()` |
| Live ROC curve | Recomputed from test split |
| Classification report | Recomputed from test split |
| Feature importances | `model.feature_importances_` (trees) or `abs(model.coef_[0])` (LR) |

---

## Shared Backend (root modules)

### `utils.py`

| Symbol | Type | Purpose |
|---|---|---|
| `MODEL_REGISTRY` | `dict` | Maps display name → `(pkl_stem, metrics_prefix)` |
| `REQUIRED_INPUT_COLS` | `list[str]` | 19 columns that must be present in uploaded CSV |
| `METRIC_LABELS` | `dict` | Maps `test_accuracy` → `"Accuracy"` etc. |
| `load_model(name)` | `@st.cache_resource` | Loads `.pkl` from `models/` |
| `get_fitted_pipeline()` | `@st.cache_resource` | Loads or builds preprocessing pipeline |
| `load_processed_split()` | `@st.cache_data` | Returns `(X_train, X_test, y_train, y_test)` |
| `load_latest_metrics(name)` | `@st.cache_data` | Returns latest metrics JSON dict |
| `load_all_model_metrics()` | `@st.cache_data` | Returns leaderboard DataFrame |
| `get_test_predictions(name)` | `@st.cache_data` | Returns `(y_test, y_pred, y_proba)` |
| `run_inference(df, name)` | — | Full raw → preprocessed → predict pipeline |

### `components/metrics.py`

| Function | Renders |
|---|---|
| `kpi_cards(metrics)` | 6 `st.metric` cards in a 3-column grid |
| `train_vs_test_table(metrics)` | Train / Test / Gap comparison DataFrame |
| `model_params_expander(params)` | Collapsible hyperparameter table |

### `components/plots.py`

| Function | Chart type |
|---|---|
| `plot_confusion_matrix(y_true, y_pred)` | Seaborn heatmap |
| `plot_roc_curve(y_true, y_proba)` | Single Matplotlib ROC curve |
| `plot_roc_comparison(curves)` | Overlay of multiple ROC curves |
| `display_classification_report(y_true, y_pred)` | Precision / Recall / F1 styled DataFrame |
| `plot_model_comparison(df, metric)` | Horizontal bar chart |
| `plot_feature_importance(imp, names)` | Horizontal bar chart (single model) |
| `plot_multi_feature_importance(data)` | `st.tabs` — one tab per model |
| `plot_proba_histogram(y_proba)` | Probability distribution histogram |

---

## Caching Strategy

| Decorator | What is cached | Invalidated by |
|---|---|---|
| `@st.cache_resource` | Model objects, Pipeline object | App restart |
| `@st.cache_data` | Metrics DataFrames, predictions arrays, processed split | App restart or explicit `st.cache_data.clear()` |

`cache_resource` is used for large in-memory objects (sklearn models, pipelines) that are expensive to deserialise and are read-only. `cache_data` is used for serialisable return values (NumPy arrays, DataFrames).

---

## Data Flow (Predictions Page)

```
User uploads CSV
       │
       ▼
st.file_uploader → bytes in memory
       │
       ▼
5 validation checks (size, parse, rows, columns, nulls)
       │ pass
       ▼
pd.read_csv → df_raw (raw DataFrame)
       │
       ▼
run_inference(df_raw, model_name)
   ├── features.preprocess.prepare(df_raw)
   │       clean_raw → engineer_features → encode_binary → split_X_y
   │       returns X (features), y (target if present)
   │
   ├── get_fitted_pipeline().transform(X)
   │       StandardScaler + OneHotEncoder (fitted on training data)
   │
   └── model.predict(X_proc) + model.predict_proba(X_proc)
           returns y_pred, y_proba
       │
       ▼
results DataFrame (original columns + Churn Prediction + Churn Probability + Prediction Label)
       │
       ▼
If Churn Label column present → live metrics + confusion matrix + ROC + report
       │
       ▼
st.download_button → CSV download
```

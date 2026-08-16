# Telco Customer Churn — Machine Learning Classification

**BITS Pilani WILP | Machine Learning | Assignment 2**
**Student:** Riya Thakur | **Submission:** August 2026

---

## Problem Statement

Customer churn — voluntary cancellation or switching to a competitor — is one of the most costly problems in telecom. Acquiring a new customer costs 5–25× more than retaining an existing one. Proactively identifying at-risk customers lets retention teams intervene before the customer leaves.

**Objective:** Build, evaluate, and deploy a binary classification system that predicts whether a telecom customer will churn (`Yes`) or stay (`No`) using historical service-usage and demographic data.

**Business metric priority:** AUC → Recall → F1 (missing a churner costs more than a false alarm).

---

## Dataset Description

| Attribute | Value |
|---|---|
| **Name** | IBM Telco Customer Churn Dataset |
| **Source** | [Kaggle — Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) |
| **Records** | 7,043 customers |
| **Raw Features** | 33 columns |
| **Target** | `Churn Label` — `Yes` (churned) / `No` (retained) |
| **Class Balance** | ~73% No / ~27% Yes (imbalanced) |
| **Missing Values** | `Churn Reason` only (blank for non-churned rows) |

### Feature Groups

| Group | Examples |
|---|---|
| Demographics | Gender, Senior Citizen, Partner, Dependents |
| Account | Tenure Months, Contract, Payment Method, Paperless Billing |
| Services | Phone, Internet, Streaming TV/Movies, Online Security/Backup |
| Billing | Monthly Charges, Total Charges |
| Engineered | Services Count, Charge Per Tenure, Avg Monthly Spend |

### Preprocessing Pipeline

1. Drop identifiers (CustomerID, Lat/Long, Zip Code) and leakage columns (Churn Score, CLTV, Churn Reason)
2. Fix `Total Charges` dtype; coerce blanks to `0.0` for new customers
3. Engineer `Services Count`, `Charge Per Tenure`, `Avg Monthly Spend`
4. Binary-encode Yes/No columns to 1/0
5. **StandardScaler** on numeric features; **OneHotEncoder** on multi-level categoricals
6. Class imbalance handled via `class_weight="balanced"` in every classifier

**Final feature matrix:** 43 features | 5,634 train / 1,409 test (80/20 stratified, seed = 42)

---

## GitHub Repository

> **https://github.com/riya-oss/ml-classification-dashboard**

---

## Model Results

All metrics are on the held-out **test set** (20%, 1,409 rows), sorted by AUC descending.

| Rank | Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Random Forest** | 0.7658 | **0.8513** | 0.5400 | 0.7941 | **0.6429** | **0.4975** |
| 2 | Logistic Regression | 0.7438 | 0.8489 | 0.5114 | 0.7807 | 0.6180 | 0.4598 |
| 3 | XGBoost | 0.7658 | 0.8435 | **0.5447** | 0.7166 | 0.6189 | 0.4633 |
| 4 | Decision Tree | 0.7346 | 0.8202 | 0.5000 | 0.7754 | 0.6080 | 0.4443 |
| 5 | Naive Bayes | 0.7097 | 0.8128 | 0.4730 | **0.8209** | 0.6002 | 0.4345 |
| 6 | KNN | **0.7736** | 0.8035 | 0.5762 | 0.5561 | 0.5660 | 0.4130 |

### Train vs Test Gap

| Model | Train AUC | Test AUC | Gap |
|---|:---:|:---:|:---:|
| Logistic Regression | 0.8624 | 0.8489 | +0.0135 |
| Naive Bayes | 0.8277 | 0.8128 | +0.0149 |
| Random Forest | 0.9158 | 0.8513 | +0.0645 |
| Decision Tree | 0.9013 | 0.8202 | +0.0811 |
| XGBoost | 0.9744 | 0.8435 | +0.1309 |
| KNN | 1.0000 | 0.8035 | +0.1965 |

---

## Per-Model Observations

### Logistic Regression — AUC 0.8489
- Strong probabilistic ranking; linear boundary captures a significant portion of churn signal.
- **Recall 0.7807** — catches ~78% of actual churners; most operationally useful simple baseline.
- Minimal overfitting gap (0.0135) — generalises cleanly to unseen data.
- **Limitation:** Cannot capture non-linear interactions (fiber optic + month-to-month contract effect).

### Decision Tree — AUC 0.8202
- Only fully **white-box** model; every prediction traceable to explicit if/else rules.
- Clear overfitting despite `max_depth=10` and `min_samples_leaf=20` (8.1pp train-test gap).
- Consistently outperformed by ensemble counterparts on every metric.
- **Best for:** regulatory/explainability requirements where a single decision path is required.

### KNN — AUC 0.8035
- **Severe overfitting:** Train AUC 1.0000 → Test AUC 0.8035 (19.7pp collapse).
- Lowest recall (0.5561) — misses nearly half of real churners; worst business outcome.
- Curse of dimensionality: distance-based similarity degrades across 43-dimensional encoded space.
- **Not recommended for production** without dimensionality reduction.

### Naive Bayes — AUC 0.8128
- **Highest recall (0.8209)** — catches >82% of churners; best when false negatives are unacceptable.
- Lowest precision (0.4730) — ~1 in 2 alarms are false positives; high intervention budget required.
- Near-zero overfitting (0.0149 gap) — independence assumption acts as a strong regulariser.
- **Limitation:** Gaussian NB independence violated; Monthly Charges, Total Charges, Tenure are correlated.

### Random Forest — AUC 0.8513 - WINNER
- **Best AUC (0.8513)** — strongest probabilistic ranking across all six models.
- **Best F1 (0.6429) and MCC (0.4975)** — best balance across all four confusion matrix quadrants.
- Handles correlated features natively through random feature subsampling at each split.
- Moderate, acceptable overfitting (6.5pp gap) for a 100-tree ensemble.

### XGBoost — AUC 0.8435
- **Highest precision (0.5447)** — most confident when flagging a churner.
- Higher overfitting gap (13.1pp) than Random Forest; would benefit from tuning (min_child_weight, gamma).
- **Recommended when precision matters:** expensive retention offers sent only to high-confidence predictions.

---

## Overall Winner

> ### Random Forest — AUC 0.8513 | F1 0.6429 | MCC 0.4975

Random Forest is the recommended production model because:

1. **Highest AUC (0.8513)** — best probabilistic ranking of customers by churn risk.
2. **Best F1 (0.6429) and MCC (0.4975)** — best compound balance across all confusion matrix cells.
3. **Strong recall (0.7941)** — catches ~79.4% of churners; missing a churner is more costly than a false alarm.
4. **Robust to correlated features** — tenure, charges handled natively via feature subsampling.
5. **No distributional assumptions** — unlike Naive Bayes (independence) or Logistic Regression (linearity).

**Runner-up:** XGBoost when precision / budget constraints are primary.
**Not recommended:** KNN (severe overfit, non-scalable), Decision Tree (dominated by ensembles).

---

## Streamlit Application

> ### 🚀 Live App: https://ml-classification-dashboard-5yneh3ntlc24b9vwjx4rej.streamlit.app/

Deployment target (required by assignment): **Streamlit Community Cloud**

This is the final **public** Streamlit URL and it opens the interactive frontend without any sign-in prompt.

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/riya-oss/ml-classification-dashboard.git
cd ml-classification-dashboard

# 2. Create Conda environment and activate
conda create -n play_env python=3.12 -y
conda activate play_env

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Train models (skip if .pkl files already present in models/)
python src/models/run_training.py

# 5. Launch the app
streamlit run app.py
# Opens at http://localhost:8501
```

### App Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview, quick-stats leaderboard, user journey map |
| 📊 Model Comparison | Ranked leaderboard · bar charts · ROC overlay · feature importance tabs · winner card |
| 🔮 Predictions | Upload CSV → 5-test validation → select model → predict → confusion matrix → download |
| 🔬 Model Insights | Per-model KPIs · live confusion matrix · classification report · feature importance |

### Demo Upload File

A ready-to-use 100-row test CSV (with `Churn Label`) is included:

```
test_data.csv
```

Mirror location used by code and docs:

```
data/test_data.csv
```

Upload it on the **Predictions** page to see a full end-to-end run: validation → predictions → metrics → confusion matrix → ROC → classification report → download.

---

## Project Structure

```
ml-classification-dashboard/
├── app.py                          # Streamlit app entrypoint
├── README.md                       # Assignment README in the required format
├── requirements.txt                # Cloud/runtime Python dependencies
├── pyproject.toml                  # Minimal project metadata for deployment tooling
├── runtime.txt                     # Pins Python version for Streamlit Cloud
├── test_data.csv                   # Root-level 100-row test CSV required for submission
├── utils.py                        # Shared app helpers for model/metrics loading
├── components/
│   ├── __init__.py
│   ├── metrics.py                  # Reusable KPI cards and metric tables
│   └── plots.py                    # Reusable charts and confusion matrix/ROC plots
├── pages/
│   ├── 1_Model_Comparison.py       # Side-by-side model comparison page
│   ├── 2_Predictions.py            # CSV upload, predictions, and evaluation page
│   └── 3_Model_Insights.py         # Single-model deep-dive page
├── data/
│   ├── raw/                        # Source Telco dataset
│   ├── processed/                  # Processed training/inference dataset
│   └── test_data.csv               # 100-row demo file for Streamlit upload
├── docs/
│   ├── dataset_notes.md
│   ├── eda_summary.md
│   ├── model_analysis.md
│   └── streamlit_architecture.md
├── src/
│   ├── data/
│   │   ├── load_data.py            # Raw dataset loading helpers
│   │   └── validate_data.py        # Data validation helpers
│   ├── features/
│   │   ├── preprocess.py           # clean → engineer → encode
│   │   └── pipelines.py            # ColumnTransformer + train/test split
│   ├── ml_classification_dashboard/
│   │   └── __init__.py             # Package marker for deployment tooling
│   └── models/
│       ├── train.py                # fit, save, log metrics
│       ├── evaluate.py             # metrics, confusion matrix, load helpers
│       ├── predict.py              # load_model, predict, predict_proba
│       └── run_training.py         # trains all 6 models end-to-end
├── models/                         # Saved .pkl files (6 classifiers + pipeline)
├── reports/
│   ├── model_comparison.csv        # Final comparison table used in README
│   ├── feature_rankings.csv
│   ├── correlation_analysis.csv
│   ├── data_quality_report.csv
│   ├── churn_segment_analysis.csv
│   ├── figures/                    # EDA and model-analysis figures
│   └── metrics/                    # Per-model timestamped JSON evaluation logs
└── notebooks/
	└── 01_eda.ipynb                # Exploratory analysis notebook
```

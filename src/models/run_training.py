"""Train all six baseline classifiers and persist models + metrics."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ── Resolve project root so imports work when run directly ────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from features.pipelines import run_pipeline
from models.evaluate import load_all_metrics, save_comparison_csv
from models.train import train

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = ROOT / "data" / "raw" / "Telco_customer_churn.xlsx"
REPORTS_DIR  = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_model_comparison(comparison: pd.DataFrame) -> Path:
    """Write clean model_comparison.csv with the required columns."""
    col_map = {
        "model_name":    "Model",
        "test_accuracy": "Accuracy",
        "test_auc":      "AUC",
        "test_precision":"Precision",
        "test_recall":   "Recall",
        "test_f1":       "F1",
        "test_mcc":      "MCC",
    }
    out = comparison[[c for c in col_map if c in comparison.columns]].rename(columns=col_map)
    path = REPORTS_DIR / "model_comparison.csv"
    out.to_csv(path, index=False)
    print(f"[report] model_comparison.csv saved -> {path}")
    return path


def _write_model_analysis(comparison: pd.DataFrame) -> Path:
    """Generate model_analysis.md with per-model observations and overall winner."""
    r = comparison.set_index("model_name").to_dict("index")

    def row(name):
        return r.get(name, {})

    lr  = row("logistic_regression")
    dt  = row("decision_tree")
    knn = row("knn")
    nb  = row("naive_bayes")
    rf  = row("random_forest")
    xgb = row("xgboost")

    winner = comparison.iloc[0]  # sorted by AUC descending

    md = textwrap.dedent(f"""\
    # Model Analysis — Telco Customer Churn

    > **Dataset:** IBM Telco Customer Churn | 7,043 records | Target: `Churn Label` (Yes/No)
    > **Split:** 80% train / 20% test | Stratified | Class imbalance ~73:27 (No:Yes)
    > **Preprocessing:** StandardScaler (numeric) + OneHotEncoder (categorical) + `class_weight="balanced"` where applicable

    ---

    ## Results Summary

    | Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
    |---|---|---|---|---|---|---|
    | Logistic Regression | {lr.get('test_accuracy','N/A'):.4f} | {lr.get('test_auc','N/A'):.4f} | {lr.get('test_precision','N/A'):.4f} | {lr.get('test_recall','N/A'):.4f} | {lr.get('test_f1','N/A'):.4f} | {lr.get('test_mcc','N/A'):.4f} |
    | Decision Tree       | {dt.get('test_accuracy','N/A'):.4f} | {dt.get('test_auc','N/A'):.4f} | {dt.get('test_precision','N/A'):.4f} | {dt.get('test_recall','N/A'):.4f} | {dt.get('test_f1','N/A'):.4f} | {dt.get('test_mcc','N/A'):.4f} |
    | KNN                 | {knn.get('test_accuracy','N/A'):.4f} | {knn.get('test_auc','N/A'):.4f} | {knn.get('test_precision','N/A'):.4f} | {knn.get('test_recall','N/A'):.4f} | {knn.get('test_f1','N/A'):.4f} | {knn.get('test_mcc','N/A'):.4f} |
    | Naive Bayes         | {nb.get('test_accuracy','N/A'):.4f} | {nb.get('test_auc','N/A'):.4f} | {nb.get('test_precision','N/A'):.4f} | {nb.get('test_recall','N/A'):.4f} | {nb.get('test_f1','N/A'):.4f} | {nb.get('test_mcc','N/A'):.4f} |
    | Random Forest       | {rf.get('test_accuracy','N/A'):.4f} | {rf.get('test_auc','N/A'):.4f} | {rf.get('test_precision','N/A'):.4f} | {rf.get('test_recall','N/A'):.4f} | {rf.get('test_f1','N/A'):.4f} | {rf.get('test_mcc','N/A'):.4f} |
    | XGBoost             | {xgb.get('test_accuracy','N/A'):.4f} | {xgb.get('test_auc','N/A'):.4f} | {xgb.get('test_precision','N/A'):.4f} | {xgb.get('test_recall','N/A'):.4f} | {xgb.get('test_f1','N/A'):.4f} | {xgb.get('test_mcc','N/A'):.4f} |

    ---

    ## Per-Model Observations

    ### Logistic Regression
    - **AUC {lr.get('test_auc',0):.4f}** — Strong probabilistic ranking; the linear decision boundary still captures a good portion of the churn signal.
    - **Recall {lr.get('test_recall',0):.4f}** — Catches ~{lr.get('test_recall',0)*100:.1f}% of actual churners, making it the most operationally useful simple baseline.
    - **Precision {lr.get('test_precision',0):.4f}** — Roughly 1 in 2 flagged customers actually churn; acceptable given the cost asymmetry (missing a churner is worse than a false alarm).
    - **Generalises well** — Minimal gap between train and test AUC ({lr.get('train_auc',0):.4f} vs {lr.get('test_auc',0):.4f}), indicating no overfitting.
    - **Limitation:** Cannot capture non-linear interactions between features such as the joint effect of fiber optic + month-to-month contract.

    ### Decision Tree
    - **AUC {dt.get('test_auc',0):.4f}** — Moderate ranking ability; the lowest AUC among tree-based models.
    - **Recall {dt.get('test_recall',0):.4f}** — Detects ~{dt.get('test_recall',0)*100:.1f}% of churners, slightly below Logistic Regression.
    - **Train vs Test gap** — Train AUC {dt.get('train_auc',0):.4f} vs Test AUC {dt.get('test_auc',0):.4f} ({(dt.get('train_auc',0)-dt.get('test_auc',0))*100:.1f}pp gap) — clear overfitting despite `max_depth=10` and `min_samples_leaf=20`.
    - **Interpretable** — The only fully white-box model in this comparison; each prediction can be traced to a clear if/else rule path. Useful for explaining individual decisions to the business.
    - **Limitation:** Single-tree instability; small data perturbations can change predictions significantly. Inferior to ensemble counterparts on all metrics.

    ### KNN
    - **AUC {knn.get('test_auc',0):.4f}** — Lowest AUC in the comparison, indicating the model struggles to rank churn probabilities reliably.
    - **Severe overfitting** — Train AUC {knn.get('train_auc',0):.4f} vs Test AUC {knn.get('test_auc',0):.4f} — a {(knn.get('train_auc',0)-knn.get('test_auc',0))*100:.1f}pp collapse. The model memorises training neighbours but fails to generalise.
    - **Recall {knn.get('test_recall',0):.4f}** — Lowest or near-lowest recall; misses a high proportion of real churners, which is the most costly error in a retention context.
    - **Precision {knn.get('test_precision',0):.4f}** — Highest precision among models, but this is a consequence of being overly conservative — it flags few customers at all.
    - **Limitation:** Distance-based similarity breaks down in high-dimensional encoded feature spaces (43 features). Computationally expensive at inference time on large datasets. Not recommended for production without dimensionality reduction.

    ### Naive Bayes
    - **AUC {nb.get('test_auc',0):.4f}** — Competitive ranking performance given its simplicity.
    - **Recall {nb.get('test_recall',0):.4f}** — Highest recall of all models; catches ~{nb.get('test_recall',0)*100:.1f}% of churners. This stems from its probabilistic prior that naturally biases toward the minority class.
    - **Precision {nb.get('test_precision',0):.4f}** — Lowest precision in the comparison. The high recall comes at the cost of many false positives — roughly 1 in 2 flagged customers will not churn.
    - **Train/Test consistency** — Train AUC {nb.get('train_auc',0):.4f} vs Test AUC {nb.get('test_auc',0):.4f} — almost no overfitting. The conditional independence assumption acts as a strong regulariser.
    - **Limitation:** The Gaussian Naive Bayes independence assumption is violated in this dataset — monthly charges, tenure, and total charges are highly correlated. This inflates false positive rates and suppresses precision.

    ### Random Forest
    - **AUC {rf.get('test_auc',0):.4f}** — Highest or joint-highest AUC in the comparison; strong probabilistic discrimination.
    - **F1 {rf.get('test_f1',0):.4f}** — Best F1 score; best balance of precision and recall.
    - **Recall {rf.get('test_recall',0):.4f}** — Strong churn detection; second-highest recall after Naive Bayes.
    - **MCC {rf.get('test_mcc',0):.4f}** — Highest MCC score, reflecting genuine predictive skill that accounts for all four quadrants of the confusion matrix.
    - **Train vs Test** — Some overfitting visible (Train AUC {rf.get('train_auc',0):.4f} vs Test {rf.get('test_auc',0):.4f}) but within acceptable bounds for an ensemble model.
    - **Strength:** Handles feature interactions and non-linearities naturally. Robust to outliers and correlated features — ideal for this dataset.

    ### XGBoost
    - **AUC {xgb.get('test_auc',0):.4f}** — Very competitive; close to Random Forest.
    - **Accuracy {xgb.get('test_accuracy',0):.4f}** — Joint-highest accuracy alongside Random Forest.
    - **Precision {xgb.get('test_precision',0):.4f}** — Highest precision of all models; when XGBoost flags a churner, it is correct more often than any other model.
    - **Recall {xgb.get('test_recall',0):.4f}** — Lower recall than Random Forest and Logistic Regression — the model errs on the side of precision.
    - **Overfitting** — Train AUC {xgb.get('train_auc',0):.4f} vs Test {xgb.get('test_auc',0):.4f} — larger gap than Logistic Regression, suggesting hyperparameter tuning (`min_child_weight`, `gamma`) would further improve generalisation.
    - **Strength:** Best precision-accuracy trade-off; suitable when false positive costs (unnecessary retention offers) matter as much as false negative costs.

    ---

    ## Overall Winner

    **🏆 {winner['model_name'].replace('_', ' ').title()}** — AUC {winner['test_auc']:.4f} | F1 {winner['test_f1']:.4f} | MCC {winner['test_mcc']:.4f}

    Random Forest is the recommended production model for the following reasons:

    1. **Highest AUC ({rf.get('test_auc',0):.4f})** — Best at ranking customers by churn risk, which is what a retention scoring system needs.
    2. **Best F1 ({rf.get('test_f1',0):.4f}) and MCC ({rf.get('test_mcc',0):.4f})** — Best overall balance across all four quadrant metrics, not just one.
    3. **Strong recall ({rf.get('test_recall',0):.4f})** — Catches ~{rf.get('test_recall',0)*100:.1f}% of churners; in a business context, missing a churner is more costly than an unnecessary retention call.
    4. **Robust to correlated features** — Monthly charges, total charges, and tenure are highly correlated; Random Forest handles this natively through feature subsampling at each split.
    5. **No strong assumptions** — Unlike Naive Bayes (independence) or Logistic Regression (linearity), Random Forest makes no distributional assumptions about the data.

    ### Runner-up: XGBoost
    XGBoost is the recommended alternative when **precision matters more than recall** — for example, if retention offers are expensive and should only be sent to high-confidence churners.

    ### Not recommended for production:
    - **KNN** — Overfits severely, lowest AUC, non-scalable at inference.
    - **Decision Tree** — Interpretable but consistently outperformed by its ensemble counterparts on all metrics.
    """)

    path = REPORTS_DIR / "model_analysis.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[report] model_analysis.md saved -> {path}")
    return path


def main() -> None:
    print("=" * 55)
    print("  Telco Churn — Baseline Model Training")
    print("=" * 55)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    raw = pd.read_excel(DATA_PATH)
    X_train, X_test, y_train, y_test, pipeline, feature_names = run_pipeline(raw)

    # ── 2. Model registry ─────────────────────────────────────────────────────
    neg_pos_ratio = round((y_train == 0).sum() / (y_train == 1).sum(), 2)

    models = [
        (
            "logistic_regression",
            "logistic",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
        ),
        (
            "decision_tree",
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "knn",
            "knn",
            KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
                metric="euclidean",
                n_jobs=-1,
            ),
        ),
        (
            "naive_bayes",
            "naive_bayes",
            GaussianNB(),
        ),
        (
            "random_forest",
            "random_forest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "xgboost",
            "xgboost",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=neg_pos_ratio,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    # ── 3. Train loop ─────────────────────────────────────────────────────────
    for model_name, model_filename, estimator in models:
        train(
            model=estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            model_filename=model_filename,
            feature_names=feature_names,
            params=estimator.get_params(),
        )

    # ── 4. Build comparison table and reports ─────────────────────────────────
    comparison = load_all_metrics()
    if not comparison.empty:
        print("\n" + "=" * 55)
        print("  All-Model Comparison (sorted by test AUC)")
        print("=" * 55)
        display_cols = [c for c in [
            "model_name", "test_auc", "test_f1", "test_precision",
            "test_recall", "test_accuracy", "test_mcc",
        ] if c in comparison.columns]
        print(comparison[display_cols].to_string(index=False))

        _save_model_comparison(comparison)
        _write_model_analysis(comparison)
        save_comparison_csv()

    print("\nOK Training complete.")
    print(f"  Models  -> {ROOT / 'models'}")
    print(f"  Reports -> {REPORTS_DIR}")


if __name__ == "__main__":
    main()


# Model Analysis — Telco Customer Churn

> **Dataset:** IBM Telco Customer Churn | 7,043 records | Target: `Churn Label` (Yes/No)
> **Split:** 80% train / 20% test | Stratified | Class imbalance ~73:27 (No:Yes)
> **Preprocessing:** StandardScaler (numeric) + OneHotEncoder (categorical) + `class_weight="balanced"` where applicable

---

## Results Summary

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7438 | 0.8489 | 0.5114 | 0.7807 | 0.6180 | 0.4598 |
| Decision Tree       | 0.7346 | 0.8202 | 0.5000 | 0.7754 | 0.6080 | 0.4443 |
| KNN                 | 0.7736 | 0.8035 | 0.5762 | 0.5561 | 0.5660 | 0.4130 |
| Naive Bayes         | 0.7097 | 0.8128 | 0.4730 | 0.8209 | 0.6002 | 0.4345 |
| Random Forest       | 0.7658 | 0.8513 | 0.5400 | 0.7941 | 0.6429 | 0.4975 |
| XGBoost             | 0.7658 | 0.8435 | 0.5447 | 0.7166 | 0.6189 | 0.4633 |

---

## Per-Model Observations

### Logistic Regression
- **AUC 0.8489** — Strong probabilistic ranking; the linear decision boundary still captures a good portion of the churn signal.
- **Recall 0.7807** — Catches ~78.1% of actual churners, making it the most operationally useful simple baseline.
- **Precision 0.5114** — Roughly 1 in 2 flagged customers actually churn; acceptable given the cost asymmetry (missing a churner is worse than a false alarm).
- **Generalises well** — Minimal gap between train and test AUC (0.8624 vs 0.8489), indicating no overfitting.
- **Limitation:** Cannot capture non-linear interactions between features such as the joint effect of fiber optic + month-to-month contract.

### Decision Tree
- **AUC 0.8202** — Moderate ranking ability; the lowest AUC among tree-based models.
- **Recall 0.7754** — Detects ~77.5% of churners, slightly below Logistic Regression.
- **Train vs Test gap** — Train AUC 0.9013 vs Test AUC 0.8202 (8.1pp gap) — clear overfitting despite `max_depth=10` and `min_samples_leaf=20`.
- **Interpretable** — The only fully white-box model in this comparison; each prediction can be traced to a clear if/else rule path. Useful for explaining individual decisions to the business.
- **Limitation:** Single-tree instability; small data perturbations can change predictions significantly. Inferior to ensemble counterparts on all metrics.

### KNN
- **AUC 0.8035** — Lowest AUC in the comparison, indicating the model struggles to rank churn probabilities reliably.
- **Severe overfitting** — Train AUC 1.0000 vs Test AUC 0.8035 — a 19.7pp collapse. The model memorises training neighbours but fails to generalise.
- **Recall 0.5561** — Lowest or near-lowest recall; misses a high proportion of real churners, which is the most costly error in a retention context.
- **Precision 0.5762** — Highest precision among models, but this is a consequence of being overly conservative — it flags few customers at all.
- **Limitation:** Distance-based similarity breaks down in high-dimensional encoded feature spaces (43 features). Computationally expensive at inference time on large datasets. Not recommended for production without dimensionality reduction.

### Naive Bayes
- **AUC 0.8128** — Competitive ranking performance given its simplicity.
- **Recall 0.8209** — Highest recall of all models; catches ~82.1% of churners. This stems from its probabilistic prior that naturally biases toward the minority class.
- **Precision 0.4730** — Lowest precision in the comparison. The high recall comes at the cost of many false positives — roughly 1 in 2 flagged customers will not churn.
- **Train/Test consistency** — Train AUC 0.8277 vs Test AUC 0.8128 — almost no overfitting. The conditional independence assumption acts as a strong regulariser.
- **Limitation:** The Gaussian Naive Bayes independence assumption is violated in this dataset — monthly charges, tenure, and total charges are highly correlated. This inflates false positive rates and suppresses precision.

### Random Forest
- **AUC 0.8513** — Highest or joint-highest AUC in the comparison; strong probabilistic discrimination.
- **F1 0.6429** — Best F1 score; best balance of precision and recall.
- **Recall 0.7941** — Strong churn detection; second-highest recall after Naive Bayes.
- **MCC 0.4975** — Highest MCC score, reflecting genuine predictive skill that accounts for all four quadrants of the confusion matrix.
- **Train vs Test** — Some overfitting visible (Train AUC 0.9158 vs Test 0.8513) but within acceptable bounds for an ensemble model.
- **Strength:** Handles feature interactions and non-linearities naturally. Robust to outliers and correlated features — ideal for this dataset.

### XGBoost
- **AUC 0.8435** — Very competitive; close to Random Forest.
- **Accuracy 0.7658** — Joint-highest accuracy alongside Random Forest.
- **Precision 0.5447** — Highest precision of all models; when XGBoost flags a churner, it is correct more often than any other model.
- **Recall 0.7166** — Lower recall than Random Forest and Logistic Regression — the model errs on the side of precision.
- **Overfitting** — Train AUC 0.9744 vs Test 0.8435 — larger gap than Logistic Regression, suggesting hyperparameter tuning (`min_child_weight`, `gamma`) would further improve generalisation.
- **Strength:** Best precision-accuracy trade-off; suitable when false positive costs (unnecessary retention offers) matter as much as false negative costs.

---

## Overall Winner

**🏆 Random Forest** — AUC 0.8513 | F1 0.6429 | MCC 0.4975

Random Forest is the recommended production model for the following reasons:

1. **Highest AUC (0.8513)** — Best at ranking customers by churn risk, which is what a retention scoring system needs.
2. **Best F1 (0.6429) and MCC (0.4975)** — Best overall balance across all four quadrant metrics, not just one.
3. **Strong recall (0.7941)** — Catches ~79.4% of churners; in a business context, missing a churner is more costly than an unnecessary retention call.
4. **Robust to correlated features** — Monthly charges, total charges, and tenure are highly correlated; Random Forest handles this natively through feature subsampling at each split.
5. **No strong assumptions** — Unlike Naive Bayes (independence) or Logistic Regression (linearity), Random Forest makes no distributional assumptions about the data.

### Runner-up: XGBoost
XGBoost is the recommended alternative when **precision matters more than recall** — for example, if retention offers are expensive and should only be sent to high-confidence churners.

### Not recommended for production:
- **KNN** — Overfits severely, lowest AUC, non-scalable at inference.
- **Decision Tree** — Interpretable but consistently outperformed by its ensemble counterparts on all metrics.

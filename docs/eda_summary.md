# EDA Summary — Telco Customer Churn

## Dataset
- Records: 7,043 | Features: 34
- Target: `Churn Label`  (Yes=26.5%, No=73.5%)
- Class imbalance ratio: 2.8:1 (No:Yes)

## Quality
- Only `Churn Reason` has nulls (non-churned customers — expected, not a problem).
- `Total Charges` stored as object — converted to numeric (blanks = new customers with zero tenure).

## Key Findings

### Top 5 Features by Point-Biserial Correlation (statistical, no models)
  1. Contract
  2. Tenure Months
  3. Online Security
  4. Tech Support
  5. Dependents

### Target
- Overall churn rate: **26.5%**
- Month-to-month contract churn: **42.7%** (vs 26.5% overall)
- Fiber optic users churn at **41.9%**
- Senior citizens churn at **41.7%**

### Statistical Tests
- All numeric features are statistically significant (Mann-Whitney U, p < 0.05).
- Contract type, Internet Service, and Tenure are the strongest categorical predictors (Cramér's V).
- Feature ranking is based purely on Point-Biserial r — no classifiers used in EDA.

## Recommendations

1. **Target month-to-month customers** — highest churn risk; incentivise annual/biannual contracts.
2. **Investigate Fiber Optic satisfaction** — churn rate is disproportionately high; may indicate service quality issues.
3. **Tenure-based retention** — customers churning earliest (0-12 months) warrant an onboarding loyalty programme.
4. **Senior citizen segment** — elevated churn rate; consider dedicated support tier or simplified plans.
5. **Modeling strategy** — use `SMOTE` or `class_weight="balanced"` to handle the 2.8:1 imbalance.
6. **Feature engineering** — create `services_count` (sum of add-ons), `charge_per_tenure`, and `contract_encoded` as strong model inputs.
7. **Drop at modeling time** — `CustomerID`, `Lat Long`, `Churn Score`, `CLTV`, `Churn Reason` (target leakage / identifiers).

## Artifacts Saved
| File | Location |
|------|----------|
| data_quality_report.csv      | reports/ |
| correlation_analysis.csv     | reports/ |
| feature_rankings.csv         | reports/ |
| churn_segment_analysis.csv   | reports/ |
| eda_summary.md               | reports/ |
| target_distribution.png      | reports/figures/ |
| missingness_heatmap.png      | reports/figures/ |
| correlation_heatmap.png      | reports/figures/ |
| cramers_v_heatmap.png        | reports/figures/ |
| pca_projection.png           | reports/figures/ |
| umap_projection.png          | reports/figures/ |
| feature_importance.png       | reports/figures/ |
| churn_segment_analysis.png   | reports/figures/ |

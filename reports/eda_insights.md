# EDA Insights — Bank Customer Churn (Training Set)

> **Note:** All analysis below is based on the **training set only** (8,000 customers).
> The test set has not been touched — it is reserved for final evaluation.

## Dataset Summary

| Metric | Value |
|--------|-------|
| Training samples | 8,000 |
| Churned (Exited=1) | 1,630 (20.4%) |
| Stayed (Exited=0) | 6,370 (79.6%) |
| Features | 13 |

## Key Insights

1. **Class Imbalance:** Only 20.4% of training customers churned — the dataset is imbalanced. Models optimizing accuracy alone will miss churners. F1 score is the right metric, and class-weight balancing or SMOTE should be considered.

2. **Age is the strongest churn predictor:** Middle-aged and older customers (41-60) churn at dramatically higher rates than younger customers. Age shows the highest positive correlation with churn among numeric features.

3. **Germany has the highest churn rate** (33.0%), roughly double that of France (15.9%) and Spain (16.6%). German customers may need targeted retention strategies.

4. **Customers with 3-4 products churn at extreme rates** (82%-100%) vs. 1-2 products (27.8%-7.6%). However, very few customers have 3+ products — these may be special cases.

5. **Inactive members churn more** (26.5%) vs. active members (14.6%). Engagement programs for inactive members could help reduce churn.

6. **Balance and gender patterns:** Customers with non-zero balances churn at a higher rate (24.2%) than zero-balance customers (13.5%). Additionally, female customers churn more than male customers across all geographies, with German females showing the highest churn segment.

## Figures

All visualizations are saved in `reports/figures/`:

| # | Figure | Description |
|---|--------|-------------|
| 01 | `01_class_balance.png` | Churn class distribution (bar + pie) |
| 02 | `02_univariate_numeric.png` | Histograms/KDE for all numeric features |
| 03-08 | `03-08_churn_by_*.png` | Churn rate by each categorical feature |
| 09 | `09_correlation_heatmap.png` | Correlation matrix of numeric features |
| 10 | `10_churn_correlation_ranking.png` | Feature correlations ranked by churn |
| 11 | `11_segment_analysis.png` | Multi-panel segment analysis |

## Implications for Modeling (Phase 3)

- **Class imbalance handling** is critical — use `class_weight='balanced'` or SMOTE inside CV folds
- **Age, Geography, NumOfProducts, IsActiveMember** are the strongest churn signals — prioritize these in feature engineering
- **Satisfaction Score and Card Type** show minimal correlation with churn — may add noise rather than signal
- **Engineered features** to consider: age buckets, balance-to-salary ratio, Germany flag, zero-balance flag, inactive × multi-product interaction

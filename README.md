# Customer Churn Prediction

Predicting bank customer churn using machine learning. This is a team project where we analyze the Bank Customer Churn dataset from Kaggle and build models to identify customers who are likely to leave.

## Dataset

- **Source:** Bank Customer Churn Dataset (Kaggle)
- **Size:** 10,000 customers, 18 features
- **Target:** `Exited` (1 = churned, 0 = stayed)
- **Split:** 80/20 train/test with `random_state=42` (stratified)

## Project Structure

```
├── data/                  # dataset files (raw + train/test split)
├── notebooks/             # EDA and analysis notebooks
├── src/                   # python scripts for data processing
│   ├── split_data.py      # creates the canonical train/test split
│   ├── eda_churn.py       # Phase 1 EDA — all visualizations & insights
│   ├── feature_engineering.py   # Phase 2 — custom sklearn feature transformers
│   └── preprocessing_pipeline.py # Phase 2 — full Pipeline + ColumnTransformer
├── models/                # saved pipeline artifacts (.joblib)
├── reports/
│   ├── figures/           # all EDA charts (11 PNGs)
│   └── eda_insights.md    # key insight takeaways from EDA
├── requirements.txt       # project dependencies
└── Customer_Churn_Roadmap.md  # team roadmap and task breakdown
```

## Setup

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Place the raw dataset as `data/raw-bank-churn.csv` (download from Kaggle).

3. Run the split script to generate train/test files:
```bash
python src/split_data.py --input data/raw-bank-churn.csv --output-dir data
```

This creates `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` in the `data/` folder.

4. Run the preprocessing pipeline to generate processed features:
```bash
python src/preprocessing_pipeline.py --verify
```

This creates `X_train_processed.csv`, `X_test_processed.csv` in `data/` and saves the fitted pipeline to `models/`.

## Columns Dropped (Leakage Prevention)

We drop these columns before any modeling:

| Column | Reason |
|--------|--------|
| `RowNumber` | Just a row index, no predictive value |
| `CustomerId` | Unique identifier, not a feature |
| `Surname` | Customer name, irrelevant to churn |
| `Complain` | **Target leakage** — directly correlates with `Exited` |

## Engineered Features (Phase 2)

| Feature | Source Columns | Rationale (from EDA) |
|---------|---------------|---------------------|
| `AgeBucket` | Age | Age is the #1 churn predictor; binned into 5 ordinal groups |
| `BalanceToSalaryRatio` | Balance, EstimatedSalary | Captures relative financial position |
| `IsGermany` | Geography | Germany has ~2x the churn rate of France/Spain |
| `ZeroBalance` | Balance | Zero vs non-zero balance shows different churn patterns |
| `InactiveProducts` | IsActiveMember, NumOfProducts | Interaction: inactive + multi-product = high risk |
| `HighValueAtRisk` | Age, IsActiveMember, Geography, Balance | Composite flag for highest-risk segment |

## Current Progress

- [x] Project structure set up (`/data`, `/notebooks`, `/src`, `/reports`)
- [x] Canonical train/test split created (80/20, stratified, random_state=42)
- [x] Leakage columns identified and dropped
- [x] EDA and visualizations (Person B) — 11 figures + insights report
- [x] Feature engineering and preprocessing pipeline (Person A - Phase 2)
- [ ] Model training and comparison (Person C - Phase 3)

## Team

- **Person A (Data & Pipeline Lead):** Samarth — preprocessing, feature engineering, train/test split
- **Person B (EDA Lead):** Kartik (techykartik07) — exploratory analysis, visualizations
- **Person C (Modeling Lead):** TBD — model training, evaluation, final F1 score


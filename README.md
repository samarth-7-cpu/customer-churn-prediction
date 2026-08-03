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
│   ├── preprocessing_pipeline.py # Phase 2 — full Pipeline + ColumnTransformer
│   └── model_training.py  # Phase 3 — GridSearchCV model training & evaluation
├── models/                # saved pipeline + model artifacts (.joblib)
├── reports/
│   ├── figures/           # all EDA charts (11 PNGs)
│   ├── eda_insights.md    # key insight takeaways from EDA
│   └── model_comparison.csv # Phase 3 model F1 scores
├── requirements.txt       # project dependencies
└── README.md
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

5. Train and evaluate all models:
```bash
python src/model_training.py --verify
```

This runs GridSearchCV (5-fold, F1-scored) for Logistic Regression, Random Forest, and XGBoost, then saves the best model as `models/final_model.joblib` and the comparison table as `reports/model_comparison.csv`.

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

## Model Results (Phase 3)

Three models were tuned via GridSearchCV (5-fold CV, F1-scored) and evaluated on the held-out test set:

| Model | CV F1 (Train) | Test F1 | Best Hyperparameters |
|-------|:-------------:|:-------:|---------------------|
| **Random Forest** 🥇 | 0.617 | **0.635** | `max_depth=None, min_samples_leaf=3, n_estimators=200` |
| **XGBoost** 🥈 | 0.617 | **0.632** | `learning_rate=0.05, max_depth=3, n_estimators=200, scale_pos_weight=3` |
| **Logistic Regression** 🥉 | 0.491 | **0.504** | `C=0.1, penalty=l2, solver=lbfgs` |

- **Best model:** Random Forest (test F1 = 0.635)
- Class imbalance handled via `class_weight='balanced'` (LR, RF) and `scale_pos_weight` (XGB)
- CV and test F1 are closely aligned — no significant overfitting

## Current Progress

- [x] Project structure set up (`/data`, `/notebooks`, `/src`, `/reports`)
- [x] Canonical train/test split created (80/20, stratified, random_state=42)
- [x] Leakage columns identified and dropped
- [x] EDA and visualizations (Person B) — 11 figures + insights report
- [x] Feature engineering and preprocessing pipeline (Person A — Phase 2)
- [x] Model training, tuning, and comparison (Person C — Phase 3)

## Team

- **Person A (Data & Pipeline Lead):** Samarth — preprocessing, feature engineering, train/test split
- **Person B (EDA Lead):** Kartik (techykartik07) — exploratory analysis, visualizations
- **Person C (Modeling Lead):** Hitesh (hiteshchandra2703-cloud) — model training, evaluation, final F1 score

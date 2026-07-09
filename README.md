# Customer Churn Prediction

This repository is for the Bank Customer Churn prediction project described in `Customer_Churn_Roadmap.md`.

## Project scope

- Dataset: Bank Customer Churn (Kaggle)
- Split: 80/20 train/test, `random_state=42`
- Target metric: F1 score on `y_test`
- Person A focus: preprocessing, feature engineering, canonical split, train/test artifacts

## Repository structure

- `data/` — raw dataset and generated train/test files
- `notebooks/` — exploratory analysis, model development notebooks
- `src/` — reusable scripts for splitting data and preparing preprocessing pipelines
- `reports/` — presentation slides, writeups, findings

## Initial setup (Person A)

1. Place the raw dataset file into `data/`, e.g. `data/raw_bank_churn.csv`.
2. Run the split script to create canonical train/test artifacts:

```powershell
python src/split_data.py --input data/raw_bank_churn.csv --output-dir data
```

3. Use the saved train/test files for all preprocessing and modeling work.
4. Do not use `X_test`/`y_test` for EDA or feature engineering.

## GitHub remote

This workspace is initialized as a local git repository. To link your GitHub repo, add the remote after you create it:

```powershell
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Files created

- `src/split_data.py` — script to generate the canonical split and drop leakage columns
- `src/preprocessing.py` — pipeline builder for train/test transforms

## Notes

- `RowNumber`, `CustomerId`, `Surname`, and `Complain` are dropped before training to avoid leakage.
- The final `f1_score` is computed only once on `X_test` / `y_test` after all training and tuning are complete.

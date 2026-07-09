"""
split_data.py
-------------
Creates the canonical 80/20 train-test split for the Bank Customer Churn dataset.
Uses stratified splitting to maintain class balance in both sets.

Drops leakage / identifier columns before splitting:
  - RowNumber, CustomerId, Surname  →  identifiers, no predictive value
  - Complain  →  target leakage (directly correlates with Exited)

Usage:
    python src/split_data.py --input data/raw-bank-churn.csv --output-dir data
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# columns to drop before any modeling work
DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname", "Complain"]
TARGET_COLUMN = "Exited"


def split_and_save(raw_path: Path, output_dir: Path, test_size=0.2, random_state=42):
    """Load raw CSV, drop leakage columns, and save stratified train/test split."""

    raw = pd.read_csv(raw_path)
    print(f"Loaded dataset: {raw.shape[0]} rows, {raw.shape[1]} columns")

    # check that all required columns exist
    missing = [col for col in DROP_COLUMNS + [TARGET_COLUMN] if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw dataset: {missing}")

    # separate features and target, dropping leakage columns
    X = raw.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y = raw[TARGET_COLUMN]

    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Churn rate: {y.mean():.2%}")

    # stratified split so both sets have roughly the same churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"\nSaved split files to {output_dir}/")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create canonical train/test split for customer churn data."
    )
    parser.add_argument("--input", required=True, help="Path to raw dataset CSV")
    parser.add_argument("--output-dir", default="data", help="Directory to save split files")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    split_and_save(Path(args.input), Path(args.output_dir), args.test_size, args.random_state)

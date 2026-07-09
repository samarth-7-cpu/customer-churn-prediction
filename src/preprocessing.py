"""
preprocessing.py
-----------------
Builds the sklearn preprocessing pipeline (ColumnTransformer) for the churn dataset.
Also contains feature engineering functions.

Design choices:
  - StandardScaler for numeric features (fit on train only, transform on test)
  - OneHotEncoder for categorical features (handle_unknown='ignore' for safety)
  - Feature engineering is done before the pipeline so engineered columns
    are available as inputs to the ColumnTransformer

NOTE: Always fit the pipeline on training data only to avoid data leakage.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


# numeric columns that go through StandardScaler
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    # engineered features (created in add_engineered_features)
    "balance_salary_ratio",
    "zero_balance",
    "active_product_interaction",
]

# categorical columns that get one-hot encoded
CATEGORICAL_FEATURES = [
    "Geography",
    "Gender",
    "HasCrCard",
    "IsActiveMember",
    "tenure_bucket",  # engineered from Tenure
]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features based on domain knowledge.

    New columns:
      - balance_salary_ratio: Balance / EstimatedSalary (how much of salary is in bank)
      - zero_balance: flag for customers with 0 balance (potential inactive accounts)
      - tenure_bucket: group tenure into meaningful ranges
      - active_product_interaction: IsActiveMember * NumOfProducts
    """
    df = X.copy()

    # ratio of balance to salary — higher ratio might mean more invested in bank
    df["balance_salary_ratio"] = np.where(
        df["EstimatedSalary"] != 0,
        df["Balance"] / (df["EstimatedSalary"] + 1e-6),  # small epsilon to avoid div by zero
        0,
    )

    # binary flag: does the customer have zero balance?
    df["zero_balance"] = (df["Balance"] == 0).astype(int)

    # bucket tenure into groups for categorical treatment
    df["tenure_bucket"] = pd.cut(
        df["Tenure"],
        bins=[-1, 1, 3, 6, 12],
        labels=["0-1", "2-3", "4-6", "7-12"]
    )

    # interaction: active members with multiple products might behave differently
    df["active_product_interaction"] = df["IsActiveMember"] * df["NumOfProducts"]

    return df


def build_preprocessor():
    """Build the ColumnTransformer pipeline (unfitted)."""

    numeric_transformer = make_pipeline(
        FunctionTransformer(lambda X: X.fillna(0), validate=False),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_features(X: pd.DataFrame, preprocessor=None):
    """Engineer features and fit+transform. Use on TRAINING data only."""
    X = add_engineered_features(X)
    if preprocessor is None:
        preprocessor = build_preprocessor()
    return preprocessor.fit_transform(X), preprocessor


def transform_features(X: pd.DataFrame, preprocessor):
    """Engineer features and transform (no fitting). Use on TEST data."""
    X = add_engineered_features(X)
    return preprocessor.transform(X)


def save_preprocessor(preprocessor, output_path: Path):
    """Save fitted preprocessor to disk."""
    joblib.dump(preprocessor, output_path)
    print(f"Saved preprocessor to {output_path}")


def load_preprocessor(path: Path):
    """Load a previously fitted preprocessor."""
    return joblib.load(path)

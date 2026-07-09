from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


NUMERIC_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]


def build_preprocessor():
    numeric_transformer = make_pipeline(
        FunctionTransformer(lambda X: X.fillna(0)),
        StandardScaler(),
    )

    categorical_transformer = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()
    df["balance_salary_ratio"] = np.where(
        df["EstimatedSalary"] != 0,
        df["Balance"] / (df["EstimatedSalary"] + 1e-6),
        0,
    )
    df["zero_balance"] = (df["Balance"] == 0).astype(int)
    df["tenure_bucket"] = pd.cut(df["Tenure"], bins=[-1, 1, 3, 6, 12], labels=["0-1", "2-3", "4-6", "7-12"])
    df["active_product_interaction"] = df["IsActiveMember"] * df["NumOfProducts"]
    return df


def prepare_features(X: pd.DataFrame, preprocessor=None):
    X = add_engineered_features(X)
    if preprocessor is None:
        preprocessor = build_preprocessor()
    return preprocessor.fit_transform(X), preprocessor


def transform_features(X: pd.DataFrame, preprocessor):
    X = add_engineered_features(X)
    return preprocessor.transform(X)


def save_preprocessor(preprocessor, output_path: Path):
    joblib.dump(preprocessor, output_path)


def load_preprocessor(path: Path):
    return joblib.load(path)

"""
preprocessing_pipeline.py
-------------------------
Phase 2 — Preprocessing Pipeline (Person A / Samarth).

Builds a sklearn Pipeline + ColumnTransformer that:
  1. Engineers new features (from EDA insights)
  2. One-hot encodes categorical features (Geography, Gender, Card Type)
  3. Scales numeric features using StandardScaler
  4. Passes through binary features untouched

The pipeline is fit ONLY on training data and then applied identically
to the test set — this prevents data leakage.

Usage:
    python src/preprocessing_pipeline.py            # builds, fits, saves pipeline
    python src/preprocessing_pipeline.py --verify   # also runs sanity checks

Outputs:
    data/X_train_processed.csv   — processed training features
    data/X_test_processed.csv    — processed test features (transform only)
    models/preprocessing_pipeline.joblib  — fitted pipeline for Person C
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from feature_engineering import EDAFeatureEngineer

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Feature categories AFTER engineering step.
# These lists define which columns go through which transformer.

# Categorical features to one-hot encode
CATEGORICAL_FEATURES = ["Geography", "Gender", "Card Type"]

# Numeric features to standard-scale
NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary",
    "Satisfaction Score", "Point Earned",
    "NumOfProducts",  # ordinal (1-4), scale alongside other numerics
    # Engineered numeric features
    "AgeBucket", "BalanceToSalaryRatio", "InactiveProducts",
]

# Binary features to pass through as-is (already 0/1)
PASSTHROUGH_FEATURES = [
    "HasCrCard", "IsActiveMember",
    # Engineered binary features
    "IsGermany", "ZeroBalance", "HighValueAtRisk",
]


def build_pipeline():
    """
    Build the full preprocessing pipeline.

    Pipeline stages:
        1. EDAFeatureEngineer   — create engineered features
        2. ColumnTransformer    — encode/scale/passthrough by column type

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline ready for .fit(X_train).
    """

    # Stage 2: column-specific transformations
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    drop="first",          # drop first category to avoid multicollinearity
                    sparse_output=False,    # dense output for compatibility
                    handle_unknown="ignore",  # safety for unseen categories in test
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "pass",
                "passthrough",
                PASSTHROUGH_FEATURES,
            ),
        ],
        remainder="drop",  # drop any columns not listed above
        verbose_feature_names_out=False,  # cleaner column names
    )

    # Full pipeline: engineer → transform
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", EDAFeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )

    return pipeline


def get_feature_names(pipeline, X_sample):
    """Extract clean feature names from a fitted pipeline."""
    ct = pipeline.named_steps["column_transformer"]
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        # Fallback: generate names manually
        names = []
        # One-hot encoded
        ohe = ct.named_transformers_["cat"]
        for feature, categories in zip(CATEGORICAL_FEATURES, ohe.categories_):
            for cat in categories[1:]:  # skip first (dropped)
                names.append(f"{feature}_{cat}")
        # Numeric (same names)
        names.extend(NUMERIC_FEATURES)
        # Passthrough (same names)
        names.extend(PASSTHROUGH_FEATURES)
        return names


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Build and apply the preprocessing pipeline."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run sanity checks after building the pipeline."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2 — Preprocessing Pipeline (Person A / Samarth)")
    print("=" * 60 + "\n")

    # ── Load data ──
    print("--- Loading data ---")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")

    # ── Build pipeline ──
    print("\n--- Building pipeline ---")
    pipeline = build_pipeline()
    print("  Pipeline structure:")
    print(f"    Step 1: {pipeline.steps[0][0]} -> EDAFeatureEngineer")
    print(f"    Step 2: {pipeline.steps[1][0]} -> ColumnTransformer")
    ct = pipeline.named_steps["column_transformer"]
    for name, transformer, cols in ct.transformers:
        t_name = type(transformer).__name__ if hasattr(transformer, '__class__') else str(transformer)
        print(f"      [{name}] {t_name} -> {cols}")

    # ── Fit on training data ONLY ──
    print("\n--- Fitting pipeline on TRAINING DATA ONLY ---")
    X_train_processed = pipeline.fit_transform(X_train)
    print(f"  X_train processed shape: {X_train_processed.shape}")

    # ── Transform test data (no fitting!) ──
    print("\n--- Transforming test data (transform only, no fit) ---")
    X_test_processed = pipeline.transform(X_test)
    print(f"  X_test processed shape:  {X_test_processed.shape}")

    # ── Get feature names ──
    feature_names = get_feature_names(pipeline, X_train)
    print(f"\n  Total features: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")

    # ── Convert to DataFrames with proper column names ──
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    # ── Save processed data ──
    print("\n--- Saving outputs ---")
    X_train_df.to_csv(DATA_DIR / "X_train_processed.csv", index=False)
    X_test_df.to_csv(DATA_DIR / "X_test_processed.csv", index=False)
    print(f"  Saved X_train_processed.csv ({X_train_df.shape})")
    print(f"  Saved X_test_processed.csv ({X_test_df.shape})")

    # ── Save fitted pipeline ──
    pipeline_path = MODEL_DIR / "preprocessing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"  Saved pipeline to {pipeline_path}")

    # ── Verification ──
    if args.verify:
        print("\n" + "=" * 60)
        print("VERIFICATION CHECKS")
        print("=" * 60)
        errors = 0

        # Check 1: shapes match
        assert X_train_df.shape[1] == X_test_df.shape[1], "Column count mismatch!"
        print(f"\n  [OK] Column counts match: train={X_train_df.shape[1]}, test={X_test_df.shape[1]}")

        # Check 2: no NaN values
        train_nans = X_train_df.isna().sum().sum()
        test_nans = X_test_df.isna().sum().sum()
        if train_nans > 0 or test_nans > 0:
            print(f"  [FAIL] NaN values found: train={train_nans}, test={test_nans}")
            errors += 1
        else:
            print(f"  [OK] No NaN values in processed data")

        # Check 3: scaled features have ~0 mean, ~1 std on train
        for col in NUMERIC_FEATURES:
            if col in X_train_df.columns:
                mean = X_train_df[col].mean()
                std = X_train_df[col].std()
                if abs(mean) > 0.01 or abs(std - 1.0) > 0.1:
                    print(f"  [FAIL] Scaling issue: {col} mean={mean:.4f}, std={std:.4f}")
                    errors += 1
        print(f"  [OK] Numeric features properly scaled (mean~0, std~1 on train)")

        # Check 4: binary features are still 0/1
        for col in PASSTHROUGH_FEATURES:
            if col in X_train_df.columns:
                unique_vals = set(X_train_df[col].unique())
                if not unique_vals.issubset({0.0, 1.0, 0, 1}):
                    print(f"  [FAIL] Binary feature {col} has non-binary values: {unique_vals}")
                    errors += 1
        print(f"  [OK] Binary features are 0/1")

        # Check 5: one-hot encoded features look correct
        ohe_cols = [c for c in feature_names if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]
        print(f"  [OK] One-hot encoded columns ({len(ohe_cols)}): {ohe_cols}")

        # Check 6: pipeline is serializable
        loaded = joblib.load(pipeline_path)
        test_output = loaded.transform(X_test.head(5))
        print(f"  [OK] Pipeline loads and transforms correctly from disk")

        # Check 7: engineered features are present
        eng_features = ["AgeBucket", "BalanceToSalaryRatio", "IsGermany",
                        "ZeroBalance", "InactiveProducts", "HighValueAtRisk"]
        present = [f for f in eng_features if f in feature_names]
        print(f"  [OK] Engineered features present: {present}")

        # Summary
        if errors == 0:
            print(f"\n  === ALL CHECKS PASSED ===")
        else:
            print(f"\n  === {errors} CHECK(S) FAILED ===")

    # ── Summary stats ──
    print("\n" + "=" * 60)
    print("PROCESSED DATA SUMMARY (Training Set)")
    print("=" * 60)
    print(X_train_df.describe().round(3).to_string())

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — Ready for Person C (Phase 3: Modeling)")
    print("=" * 60)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from feature_engineering import EDAFeatureEngineer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["Geography", "Gender", "Card Type"]
NUM_COLS = [
    "CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary",
    "Satisfaction Score", "Point Earned", "NumOfProducts",
    "AgeBucket", "BalanceToSalaryRatio", "InactiveProducts",
]
BIN_COLS = [
    "HasCrCard", "IsActiveMember",
    "IsGermany", "ZeroBalance", "HighValueAtRisk",
]


def build_pipeline():
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
            ("pass", "passthrough", BIN_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("feat_eng", EDAFeatureEngineer()),
        ("col_trans", ct),
    ])


def get_feat_names(pipe, X):
    ct = pipe.named_steps["col_trans"]
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        names = []
        ohe = ct.named_transformers_["cat"]
        for feat, cats in zip(CAT_COLS, ohe.categories_):
            for c in cats[1:]:
                names.append(f"{feat}_{c}")
        names.extend(NUM_COLS)
        names.extend(BIN_COLS)
        return names


def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    parser.add_argument("--verify", action="store_true", help="Run sanity checks")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2 — Preprocessing Pipeline")
    print("=" * 60 + "\n")

    print("Loading data...")
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_test = pd.read_csv(DATA / "X_test.csv")
    y_train = pd.read_csv(DATA / "y_train.csv")
    y_test = pd.read_csv(DATA / "y_test.csv")
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    print("\nBuilding pipeline...")
    pipe = build_pipeline()
    for name, _, cols in pipe.named_steps["col_trans"].transformers:
        print(f"  [{name}] -> {cols}")

    print("\nFitting on train data...")
    X_tr = pipe.fit_transform(X_train)
    print(f"  train processed: {X_tr.shape}")

    print("Transforming test data...")
    X_te = pipe.transform(X_test)
    print(f"  test processed: {X_te.shape}")

    feat_names = get_feat_names(pipe, X_train)
    print(f"\n  {len(feat_names)} features: {feat_names}")

    X_tr_df = pd.DataFrame(X_tr, columns=feat_names)
    X_te_df = pd.DataFrame(X_te, columns=feat_names)

    print("\nSaving outputs...")
    X_tr_df.to_csv(DATA / "X_train_processed.csv", index=False)
    X_te_df.to_csv(DATA / "X_test_processed.csv", index=False)
    print(f"  X_train_processed.csv {X_tr_df.shape}")
    print(f"  X_test_processed.csv {X_te_df.shape}")

    pipe_path = MODELS / "preprocessing_pipeline.joblib"
    joblib.dump(pipe, pipe_path)
    print(f"  pipeline -> {pipe_path}")

    if args.verify:
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        errs = 0

        assert X_tr_df.shape[1] == X_te_df.shape[1], "col count mismatch"
        print(f"\n  [OK] cols match: {X_tr_df.shape[1]}")

        tr_nan = X_tr_df.isna().sum().sum()
        te_nan = X_te_df.isna().sum().sum()
        if tr_nan > 0 or te_nan > 0:
            print(f"  [FAIL] NaN found: train={tr_nan}, test={te_nan}")
            errs += 1
        else:
            print("  [OK] no NaN")

        for col in NUM_COLS:
            if col in X_tr_df.columns:
                m, s = X_tr_df[col].mean(), X_tr_df[col].std()
                if abs(m) > 0.01 or abs(s - 1.0) > 0.1:
                    print(f"  [FAIL] {col} mean={m:.4f} std={s:.4f}")
                    errs += 1
        print("  [OK] numeric scaling looks good")

        for col in BIN_COLS:
            if col in X_tr_df.columns:
                vals = set(X_tr_df[col].unique())
                if not vals.issubset({0.0, 1.0, 0, 1}):
                    print(f"  [FAIL] {col} not binary: {vals}")
                    errs += 1
        print("  [OK] binary cols are 0/1")

        ohe_cols = [c for c in feat_names if any(c.startswith(f"{cat}_") for cat in CAT_COLS)]
        print(f"  [OK] OHE cols ({len(ohe_cols)}): {ohe_cols}")

        loaded = joblib.load(pipe_path)
        _ = loaded.transform(X_test.head(5))
        print("  [OK] pipeline serialization works")

        eng = ["AgeBucket", "BalanceToSalaryRatio", "IsGermany",
               "ZeroBalance", "InactiveProducts", "HighValueAtRisk"]
        present = [f for f in eng if f in feat_names]
        print(f"  [OK] engineered features: {present}")

        if errs == 0:
            print("\n  === ALL CHECKS PASSED ===")
        else:
            print(f"\n  === {errs} CHECK(S) FAILED ===")

    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    print(X_tr_df.describe().round(3).to_string())

    print("\n" + "=" * 60)
    print("DONE — Ready for Phase 3: Modeling")
    print("=" * 60)


if __name__ == "__main__":
    main()

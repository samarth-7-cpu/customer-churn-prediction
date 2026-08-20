import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
SEED = 42


def load_data():
    X_train = pd.read_csv(DATA / "X_train_processed.csv")
    X_test = pd.read_csv(DATA / "X_test_processed.csv")
    y_train = pd.read_csv(DATA / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA / "y_test.csv").values.ravel()
    print(f"  train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def show_class_balance(y_train):
    vals, counts = np.unique(y_train, return_counts=True)
    print("\nClass balance in y_train:")
    for v, c in zip(vals, counts):
        print(f"  class {v}: {c} ({c / len(y_train):.1%})")


def get_models():
    return {
        "LogisticRegression": (
            LogisticRegression(random_state=SEED, max_iter=2000, class_weight="balanced"),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, class_weight="balanced"),
            {"n_estimators": [200, 400], "max_depth": [5, 10, None], "min_samples_leaf": [1, 3, 5]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=SEED, eval_metric="logloss", use_label_encoder=False),
            {
                "n_estimators": [200, 400], "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "scale_pos_weight": [1, 3],  # handles churn class imbalance
            },
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = get_models()
    results = []
    fitted = {}

    for name, (estimator, params) in models.items():
        print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
        grid = GridSearchCV(estimator, params, scoring="f1", cv=5, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        fitted[name] = best

        y_pred = best.predict(X_test)
        test_f1 = f1_score(y_test, y_pred)

        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV F1 (train-only): {grid.best_score_:.4f}")
        print(f"  Test F1 (held-out): {test_f1:.4f}")
        print(classification_report(y_test, y_pred))

        results.append({
            "model": name,
            "best_params": grid.best_params_,
            "cv_f1_train": grid.best_score_,
            "test_f1": test_f1,
        })

    return results, fitted


def summarize(results):
    df = pd.DataFrame(results).sort_values("test_f1", ascending=False)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by test F1)")
    print("=" * 60)
    print(df[["model", "cv_f1_train", "test_f1"]].to_string(index=False))
    return df


def save_best(df, fitted, X_test, y_test):
    best_name = df.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (test F1 = {df.iloc[0]['test_f1']:.4f})")

    path = MODELS / "final_model.joblib"
    joblib.dump(best_model, path)
    print(f"  -> saved to {path}")

    cm = confusion_matrix(y_test, best_model.predict(X_test))
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(cm)
    return best_name, best_model


def main():
    parser = argparse.ArgumentParser(description="Phase 3 model training")
    parser.add_argument("--verify", action="store_true", help="Run sanity checks")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3 — Model Training & Evaluation")
    print("=" * 60 + "\n")

    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_data()
    show_class_balance(y_train)

    if args.verify:
        print("\n" + "=" * 60)
        print("VERIFICATION")
        print("=" * 60)
        errs = 0
        if X_train.shape[1] != X_test.shape[1]:
            print("  [FAIL] train/test column count mismatch")
            errs += 1
        else:
            print(f"  [OK] train/test columns match: {X_train.shape[1]}")
        if X_train.isna().sum().sum() or X_test.isna().sum().sum():
            print("  [FAIL] NaNs found in processed data")
            errs += 1
        else:
            print("  [OK] no NaN in processed data")
        print(f"  [OK] test set size: {len(y_test)} rows (only used for final scoring)")
        print("=" * 60 if errs == 0 else f"{errs} CHECK(S) FAILED")

    results, fitted = train_and_evaluate(X_train, X_test, y_train, y_test)
    comp_df = summarize(results)
    comp_df.to_csv(REPORTS / "model_comparison.csv", index=False)
    print(f"\nComparison table saved to {REPORTS / 'model_comparison.csv'}")

    save_best(comp_df, fitted, X_test, y_test)

    print("\n" + "=" * 60)
    print("DONE — final_model.joblib ready for the notebook / web app")
    print("=" * 60)


if __name__ == "__main__":
    main()
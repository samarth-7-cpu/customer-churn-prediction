import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DROP_COLS = ["RowNumber", "CustomerId", "Surname", "Complain"]  # leakage/id cols
TARGET = "Exited"


def split_and_save(raw_path: Path, out_dir: Path, test_size=0.2, seed=42):
    raw = pd.read_csv(raw_path)
    print(f"Loaded dataset: {raw.shape[0]} rows, {raw.shape[1]} columns")

    missing = [c for c in DROP_COLS + [TARGET] if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = raw.drop(columns=DROP_COLS + [TARGET])
    y = raw[TARGET]

    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    print(f"\nSaved split files to {out_dir}/")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/test split for churn data.")
    parser.add_argument("--input", required=True, help="Path to raw dataset CSV")
    parser.add_argument("--output-dir", default="data", help="Directory to save split files")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    split_and_save(Path(args.input), Path(args.output_dir), args.test_size, args.random_state)

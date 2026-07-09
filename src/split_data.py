import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


DROP_COLUMNS = ["RowNumber", "CustomerId", "Surname", "Complain"]
TARGET_COLUMN = "Exited"


def split_and_save(raw_path: Path, output_dir: Path, test_size: float = 0.2, random_state: int = 42):
    raw = pd.read_csv(raw_path)
    missing = [col for col in DROP_COLUMNS + [TARGET_COLUMN] if col not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw dataset: {missing}")

    X = raw.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y = raw[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print(f"Saved canonical split to {output_dir}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create canonical train/test split for customer churn data.")
    parser.add_argument("--input", required=True, help="Path to raw dataset CSV")
    parser.add_argument("--output-dir", default="data", help="Directory to save split files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    split_and_save(Path(args.input), Path(args.output_dir), args.test_size, args.random_state)

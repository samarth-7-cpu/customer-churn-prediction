"""
fit_pipeline.py
---------------
Fits the preprocessing pipeline on the training data and saves it.
The fitted pipeline can then be loaded and applied to test data later.

Usage:
    python src/fit_pipeline.py --data-dir data --output data/preprocessor.joblib
"""

from pathlib import Path

import pandas as pd
from src.preprocessing import build_preprocessor, add_engineered_features, save_preprocessor


def main(data_dir: Path, output_path: Path):
    """Fit the preprocessing pipeline on X_train and save to disk."""

    X_train = pd.read_csv(data_dir / "X_train.csv")
    print(f"Loaded X_train: {X_train.shape}")

    # add engineered features before fitting
    X_train = add_engineered_features(X_train)

    # build and fit the preprocessor
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # save for later use
    save_preprocessor(preprocessor, output_path)
    print("Pipeline fitted and saved successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit preprocessing pipeline on canonical training data."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing X_train.csv")
    parser.add_argument("--output", default="data/preprocessor.joblib",
                        help="Path to save fitted preprocessor")
    args = parser.parse_args()

    main(Path(args.data_dir), Path(args.output))

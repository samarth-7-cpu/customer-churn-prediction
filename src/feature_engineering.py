"""
feature_engineering.py
---------------------
Phase 2 — Custom feature engineering transformers (Person A / Samarth).

Implements sklearn-compatible transformers for engineered features that were
identified as important during Person B's EDA (Phase 1):

  - AgeBucket          : binned age groups (EDA showed age is #1 churn predictor)
  - BalanceToSalaryRatio: ratio feature (balance patterns matter)
  - IsGermany          : binary flag (Germany has ~2x churn rate)
  - ZeroBalance        : binary flag (zero vs non-zero balance split)
  - InactiveProducts   : interaction term (inactive × NumProducts)
  - HighValueAtRisk    : composite flag (old + inactive + Germany)

All transformers are stateless (no fitting required) so there is zero risk
of train→test data leakage from these features.

Usage:
    from feature_engineering import EDAFeatureEngineer
    transformer = EDAFeatureEngineer()
    X_new = transformer.fit_transform(X_train)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EDAFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that creates engineered features
    informed by Phase 1 EDA insights.

    Features created:
        AgeBucket              — ordinal encoding of age ranges
        BalanceToSalaryRatio   — Balance / EstimatedSalary (clipped)
        IsGermany              — 1 if Geography == 'Germany'
        ZeroBalance            — 1 if Balance == 0
        InactiveProducts       — (1 - IsActiveMember) * NumOfProducts
        HighValueAtRisk        — 1 if Age >= 40 AND inactive AND (Germany OR Balance > 100K)

    Parameters
    ----------
    age_bins : list
        Bin edges for AgeBucket. Default: [0, 30, 40, 50, 60, 100]
    drop_originals : bool
        If True, drop the original columns that were used to create
        engineered features. Default: False (keep originals for the
        model to decide importance).
    """

    def __init__(self, age_bins=None, drop_originals=False):
        self.age_bins = age_bins or [0, 30, 40, 50, 60, 100]
        self.drop_originals = drop_originals

    def fit(self, X, y=None):
        """No fitting required — all transformations are stateless."""
        return self

    def transform(self, X, y=None):
        """Create engineered features from raw columns."""
        # Work on a copy to avoid mutating the input
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # --- 1. AgeBucket (ordinal: 0–4) ---
        # EDA Insight #2: Age is the strongest churn predictor.
        # 18-30 → 0, 31-40 → 1, 41-50 → 2, 51-60 → 3, 60+ → 4
        if "Age" in df.columns:
            df["AgeBucket"] = pd.cut(
                df["Age"],
                bins=self.age_bins,
                labels=list(range(len(self.age_bins) - 1)),
                include_lowest=True,
            ).astype(int)

        # --- 2. BalanceToSalaryRatio ---
        # EDA Insight #6: Balance patterns differ between churners/stayers.
        # Ratio captures relative financial position. Clipped to avoid
        # extreme values when salary is very small.
        if "Balance" in df.columns and "EstimatedSalary" in df.columns:
            df["BalanceToSalaryRatio"] = (
                df["Balance"] / df["EstimatedSalary"].clip(lower=1.0)
            ).clip(upper=10.0)

        # --- 3. IsGermany ---
        # EDA Insight #3: Germany has ~2x churn rate (33% vs ~16%).
        if "Geography" in df.columns:
            df["IsGermany"] = (df["Geography"] == "Germany").astype(int)

        # --- 4. ZeroBalance ---
        # EDA Insight #6: Zero-balance customers churn at 13.5% vs 24.2%.
        if "Balance" in df.columns:
            df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

        # --- 5. InactiveProducts (interaction term) ---
        # EDA Insights #4 + #5: Both NumOfProducts and IsActiveMember are
        # strong churn signals. Inactive customers with multiple products
        # may be at the highest risk.
        if "IsActiveMember" in df.columns and "NumOfProducts" in df.columns:
            df["InactiveProducts"] = (
                (1 - df["IsActiveMember"]) * df["NumOfProducts"]
            )

        # --- 6. HighValueAtRisk (composite segment flag) ---
        # Combines the top EDA signals into a single "red flag" indicator:
        # older, inactive, AND either in Germany or high balance.
        has_age = "Age" in df.columns
        has_active = "IsActiveMember" in df.columns
        has_geo = "Geography" in df.columns
        has_balance = "Balance" in df.columns
        if has_age and has_active and (has_geo or has_balance):
            is_older = df["Age"] >= 40
            is_inactive = df["IsActiveMember"] == 0
            is_germany = (df["Geography"] == "Germany") if has_geo else False
            is_high_bal = (df["Balance"] > 100_000) if has_balance else False
            df["HighValueAtRisk"] = (
                is_older & is_inactive & (is_germany | is_high_bal)
            ).astype(int)

        return df

    def get_feature_names_out(self, input_features=None):
        """Return feature names for the transformer output."""
        engineered = [
            "AgeBucket", "BalanceToSalaryRatio", "IsGermany",
            "ZeroBalance", "InactiveProducts", "HighValueAtRisk",
        ]
        if input_features is not None:
            return list(input_features) + engineered
        return engineered

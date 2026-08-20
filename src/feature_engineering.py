import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EDAFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, age_bins=None, drop_originals=False):
        self.age_bins = age_bins or [0, 30, 40, 50, 60, 100]
        self.drop_originals = drop_originals

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if "Age" in df.columns:
            df["AgeBucket"] = pd.cut(
                df["Age"], bins=self.age_bins,
                labels=range(len(self.age_bins) - 1),
                include_lowest=True
            ).astype(int)

        if "Balance" in df.columns and "EstimatedSalary" in df.columns:
            df["BalanceToSalaryRatio"] = (
                df["Balance"] / df["EstimatedSalary"].clip(lower=1.0)
            ).clip(upper=10.0)

        if "Geography" in df.columns:
            df["IsGermany"] = (df["Geography"] == "Germany").astype(int)

        if "Balance" in df.columns:
            df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

        if "IsActiveMember" in df.columns and "NumOfProducts" in df.columns:
            df["InactiveProducts"] = (1 - df["IsActiveMember"]) * df["NumOfProducts"]

        # composite risk flag: old + inactive + (germany or high balance)
        has_age = "Age" in df.columns
        has_act = "IsActiveMember" in df.columns
        has_geo = "Geography" in df.columns
        has_bal = "Balance" in df.columns
        if has_age and has_act and (has_geo or has_bal):
            old = df["Age"] >= 40
            inactive = df["IsActiveMember"] == 0
            de = (df["Geography"] == "Germany") if has_geo else False
            hi_bal = (df["Balance"] > 100_000) if has_bal else False
            df["HighValueAtRisk"] = (old & inactive & (de | hi_bal)).astype(int)

        return df

    def get_feature_names_out(self, input_features=None):
        eng = [
            "AgeBucket", "BalanceToSalaryRatio", "IsGermany",
            "ZeroBalance", "InactiveProducts", "HighValueAtRisk",
        ]
        if input_features is not None:
            return list(input_features) + eng
        return eng

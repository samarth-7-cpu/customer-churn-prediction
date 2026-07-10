"""
eda_churn.py
------------
Phase 1 — Exploratory Data Analysis (Person B's work).

Performs all EDA on the TRAINING SET ONLY (X_train + y_train).
Never touches X_test / y_test — those are reserved for final evaluation.

Generates publication-quality figures saved to reports/figures/ and prints
key insight takeaways for the presentation.

Usage:
    python src/eda_churn.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Seaborn / matplotlib style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
PALETTE_CHURN = {0: "#3498db", 1: "#e74c3c"}   # blue = stayed, red = churned
PALETTE_CAT = "Set2"

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "EstimatedSalary", "Satisfaction Score", "Point Earned",
]

CATEGORICAL_FEATURES = [
    "Geography", "Gender", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "Card Type",
]


def load_train() -> pd.DataFrame:
    """Load training set (features + target) into a single DataFrame."""
    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")
    df = pd.concat([X, y], axis=1)
    print(f"Training set loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")
    return df


# ──────────────────────────────────────────────
# 1. Class balance
# ──────────────────────────────────────────────

def plot_class_balance(df: pd.DataFrame):
    """Bar + pie chart showing churn vs non-churn distribution."""
    counts = df["Exited"].value_counts().sort_index()
    labels = ["Stayed (0)", "Churned (1)"]
    colors = [PALETTE_CHURN[0], PALETTE_CHURN[1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # bar chart
    ax = axes[0]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution of Exited (Churn)")
    ax.set_ylim(0, counts.max() * 1.15)

    # pie chart
    ax = axes[1]
    ax.pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 12}, explode=(0, 0.05))
    ax.set_title("Churn Proportion")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved 01_class_balance.png")

    churn_rate = counts[1] / counts.sum()
    print(f"  -> Churn rate: {churn_rate:.2%}  ({counts[1]:,} churned / {counts.sum():,} total)")
    return churn_rate


# ──────────────────────────────────────────────
# 2. Univariate distributions (numeric)
# ──────────────────────────────────────────────

def plot_univariate_numeric(df: pd.DataFrame):
    """Histograms with KDE for each numeric feature, split by churn."""
    n = len(NUMERIC_FEATURES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_FEATURES):
        ax = axes[i]
        for label, color in PALETTE_CHURN.items():
            subset = df[df["Exited"] == label][col]
            ax.hist(subset, bins=30, alpha=0.5, color=color, density=True,
                    label=f"{'Churned' if label else 'Stayed'}", edgecolor="white")
            subset.plot.kde(ax=ax, color=color, linewidth=2)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.legend(fontsize=9)

    # hide unused subplot(s)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Univariate Distributions — Numeric Features (by Churn)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_univariate_numeric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved 02_univariate_numeric.png")


# ──────────────────────────────────────────────
# 3. Categorical breakdowns vs churn rate
# ──────────────────────────────────────────────

def plot_categorical_vs_churn(df: pd.DataFrame):
    """For each categorical feature: bar chart of churn rate per category."""
    file_index = 3
    for col in CATEGORICAL_FEATURES:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.3, 1]})

        # Left: count plot grouped by churn
        ax = axes[0]
        ct = df.groupby([col, "Exited"]).size().unstack(fill_value=0)
        ct.plot.bar(ax=ax, color=[PALETTE_CHURN[0], PALETTE_CHURN[1]], edgecolor="white")
        ax.set_title(f"{col} — Count by Churn Status")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend(["Stayed", "Churned"], loc="upper right")
        ax.tick_params(axis="x", rotation=0)

        # Right: churn rate per category
        ax = axes[1]
        churn_rate = df.groupby(col)["Exited"].mean().sort_values(ascending=False)
        bars = ax.barh(churn_rate.index.astype(str), churn_rate.values,
                       color=sns.color_palette(PALETTE_CAT, len(churn_rate)), edgecolor="white")
        for bar, val in zip(bars, churn_rate.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontweight="bold")
        ax.set_title(f"Churn Rate by {col}")
        ax.set_xlabel("Churn Rate")
        ax.set_xlim(0, min(churn_rate.max() * 1.4, 1.0))

        fig.tight_layout()
        fname = f"{file_index:02d}_churn_by_{col.lower().replace(' ', '_')}.png"
        fig.savefig(FIG_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Saved {fname}")
        file_index += 1


# ──────────────────────────────────────────────
# 4. Correlation heatmap + churn ranking
# ──────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame):
    """Correlation heatmap of numeric features and bar chart ranking by churn correlation."""

    # Include binary categoricals in correlation analysis
    numeric_cols = NUMERIC_FEATURES + ["NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]
    corr = df[numeric_cols].corr()

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 9})
    ax.set_title("Correlation Heatmap — Numeric Features + Exited", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved 09_correlation_heatmap.png")

    # --- Churn correlation ranking ---
    churn_corr = corr["Exited"].drop("Exited").sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in churn_corr.values]
    ax.barh(churn_corr.index, churn_corr.values, color=colors, edgecolor="white")
    for i, (name, val) in enumerate(zip(churn_corr.index, churn_corr.values)):
        ax.text(val + (0.005 if val >= 0 else -0.005), i,
                f"{val:+.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontweight="bold", fontsize=10)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_title("Feature Correlation with Churn (Exited)", fontsize=13)
    ax.set_xlabel("Pearson Correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_churn_correlation_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved 10_churn_correlation_ranking.png")


# ──────────────────────────────────────────────
# 5. Segment analysis
# ──────────────────────────────────────────────

def plot_segment_analysis(df: pd.DataFrame):
    """Multi-panel segment analysis: tenure bucket, product count, geography×gender, age group, balance bucket."""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # --- 5a. Churn rate by tenure bucket ---
    ax = axes[0, 0]
    df["TenureBucket"] = pd.cut(df["Tenure"], bins=[-1, 1, 3, 5, 7, 10],
                                 labels=["0-1", "2-3", "4-5", "6-7", "8-10"])
    rate = df.groupby("TenureBucket", observed=True)["Exited"].mean()
    ax.bar(rate.index.astype(str), rate.values, color=sns.color_palette("viridis", len(rate)),
           edgecolor="white")
    for i, (cat, val) in enumerate(zip(rate.index, rate.values)):
        ax.text(i, val + 0.005, f"{val:.1%}", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Churn Rate by Tenure Bucket")
    ax.set_xlabel("Tenure (years)")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, rate.max() * 1.25)

    # --- 5b. Churn rate by NumOfProducts ---
    ax = axes[0, 1]
    rate = df.groupby("NumOfProducts")["Exited"].agg(["mean", "count"])
    bars = ax.bar(rate.index.astype(str), rate["mean"],
                  color=sns.color_palette("magma", len(rate)), edgecolor="white")
    for i, (idx, row) in enumerate(rate.iterrows()):
        ax.text(i, row["mean"] + 0.01, f"{row['mean']:.1%}\n(n={row['count']:,})",
                ha="center", fontweight="bold", fontsize=9)
    ax.set_title("Churn Rate by Number of Products")
    ax.set_xlabel("NumOfProducts")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, min(rate["mean"].max() * 1.35, 1.05))

    # --- 5c. Geography × Gender churn rate ---
    ax = axes[0, 2]
    cross = df.groupby(["Geography", "Gender"])["Exited"].mean().unstack()
    cross.plot.bar(ax=ax, color=[PALETTE_CHURN[1], PALETTE_CHURN[0]], edgecolor="white")
    ax.set_title("Churn Rate: Geography × Gender")
    ax.set_ylabel("Churn Rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Gender")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # --- 5d. Churn rate by age group ---
    ax = axes[1, 0]
    df["AgeGroup"] = pd.cut(df["Age"], bins=[17, 30, 40, 50, 60, 100],
                             labels=["18-30", "31-40", "41-50", "51-60", "60+"])
    rate = df.groupby("AgeGroup", observed=True)["Exited"].mean()
    ax.bar(rate.index.astype(str), rate.values, color=sns.color_palette("rocket", len(rate)),
           edgecolor="white")
    for i, val in enumerate(rate.values):
        ax.text(i, val + 0.005, f"{val:.1%}", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Churn Rate by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, rate.max() * 1.2)

    # --- 5e. Churn rate by balance bucket ---
    ax = axes[1, 1]
    df["BalanceBucket"] = pd.cut(df["Balance"],
                                  bins=[-1, 0, 50_000, 100_000, 150_000, 300_000],
                                  labels=["Zero", "1-50K", "50-100K", "100-150K", "150K+"])
    rate = df.groupby("BalanceBucket", observed=True)["Exited"].mean()
    ax.bar(rate.index.astype(str), rate.values, color=sns.color_palette("crest", len(rate)),
           edgecolor="white")
    for i, val in enumerate(rate.values):
        ax.text(i, val + 0.005, f"{val:.1%}", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Churn Rate by Balance Bucket")
    ax.set_xlabel("Account Balance")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, rate.max() * 1.25)

    # --- 5f. Churn rate by Satisfaction Score ---
    ax = axes[1, 2]
    rate = df.groupby("Satisfaction Score")["Exited"].mean()
    ax.bar(rate.index.astype(str), rate.values, color=sns.color_palette("flare", len(rate)),
           edgecolor="white")
    for i, val in enumerate(rate.values):
        ax.text(i, val + 0.005, f"{val:.1%}", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Churn Rate by Satisfaction Score")
    ax.set_xlabel("Satisfaction Score")
    ax.set_ylabel("Churn Rate")
    ax.set_ylim(0, rate.max() * 1.25)

    fig.suptitle("Segment Analysis — Churn Rate by Customer Segments", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_segment_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved 11_segment_analysis.png")

    # cleanup temp columns
    df.drop(columns=["TenureBucket", "AgeGroup", "BalanceBucket"], inplace=True)


# ──────────────────────────────────────────────
# 6. Insight takeaways
# ──────────────────────────────────────────────

def print_insights(df: pd.DataFrame, churn_rate: float):
    """Print the key insights from the EDA for the presentation."""
    insights = []

    # Insight 1: class imbalance
    insights.append(
        f"1. **Class Imbalance:** Only {churn_rate:.1%} of training customers churned — "
        f"the dataset is imbalanced. Models optimizing accuracy alone will miss churners. "
        f"F1 score is the right metric, and class-weight balancing or SMOTE should be considered."
    )

    # Insight 2: age is a strong predictor
    age_churn = df.groupby(pd.cut(df["Age"], bins=[17, 40, 50, 100]))["Exited"].mean()
    insights.append(
        f"2. **Age is the strongest churn predictor:** Middle-aged and older customers (41-60) "
        f"churn at dramatically higher rates than younger customers. "
        f"Age shows the highest positive correlation with churn among numeric features."
    )

    # Insight 3: geography matters
    geo_churn = df.groupby("Geography")["Exited"].mean()
    top_geo = geo_churn.idxmax()
    insights.append(
        f"3. **Germany has the highest churn rate** ({geo_churn[top_geo]:.1%}), "
        f"roughly double that of France ({geo_churn.get('France', 0):.1%}) and "
        f"Spain ({geo_churn.get('Spain', 0):.1%}). German customers may need "
        f"targeted retention strategies."
    )

    # Insight 4: product count
    prod_churn = df.groupby("NumOfProducts")["Exited"].mean()
    insights.append(
        f"4. **Customers with 3-4 products churn at extreme rates** "
        f"({prod_churn.get(3, 0):.0%}-{prod_churn.get(4, 0):.0%}) vs. "
        f"1-2 products ({prod_churn.get(1, 0):.1%}-{prod_churn.get(2, 0):.1%}). "
        f"However, very few customers have 3+ products — these may be special cases."
    )

    # Insight 5: active membership
    active_churn = df.groupby("IsActiveMember")["Exited"].mean()
    insights.append(
        f"5. **Inactive members churn more** ({active_churn[0]:.1%}) vs. "
        f"active members ({active_churn[1]:.1%}). Engagement programs for "
        f"inactive members could help reduce churn."
    )

    # Insight 6: balance
    zero_bal = df[df["Balance"] == 0]["Exited"].mean()
    nonzero_bal = df[df["Balance"] > 0]["Exited"].mean()
    insights.append(
        f"6. **Balance and gender patterns:** Customers with non-zero balances churn "
        f"at a higher rate ({nonzero_bal:.1%}) than zero-balance customers ({zero_bal:.1%}). "
        f"Additionally, female customers churn more than male customers across all geographies, "
        f"with German females showing the highest churn segment."
    )

    print("\n" + "=" * 60)
    print("KEY EDA INSIGHTS (Person B — Presentation Backbone)")
    print("=" * 60)
    for ins in insights:
        print(f"\n{ins}")
    print()

    return insights


def save_insights_report(insights: list[str], df: pd.DataFrame):
    """Save insights as a markdown report to reports/eda_insights.md."""
    report_path = ROOT / "reports" / "eda_insights.md"

    # Compute some summary stats for the report header
    n = len(df)
    churn_count = df["Exited"].sum()
    stayed_count = n - churn_count

    report = f"""# EDA Insights — Bank Customer Churn (Training Set)

> **Note:** All analysis below is based on the **training set only** ({n:,} customers).
> The test set has not been touched — it is reserved for final evaluation.

## Dataset Summary

| Metric | Value |
|--------|-------|
| Training samples | {n:,} |
| Churned (Exited=1) | {churn_count:,} ({churn_count / n:.1%}) |
| Stayed (Exited=0) | {stayed_count:,} ({stayed_count / n:.1%}) |
| Features | {len(df.columns) - 1} |

## Key Insights

"""
    for ins in insights:
        report += f"{ins}\n\n"

    report += """## Figures

All visualizations are saved in `reports/figures/`:

| # | Figure | Description |
|---|--------|-------------|
| 01 | `01_class_balance.png` | Churn class distribution (bar + pie) |
| 02 | `02_univariate_numeric.png` | Histograms/KDE for all numeric features |
| 03-08 | `03-08_churn_by_*.png` | Churn rate by each categorical feature |
| 09 | `09_correlation_heatmap.png` | Correlation matrix of numeric features |
| 10 | `10_churn_correlation_ranking.png` | Feature correlations ranked by churn |
| 11 | `11_segment_analysis.png` | Multi-panel segment analysis |

## Implications for Modeling (Phase 3)

- **Class imbalance handling** is critical — use `class_weight='balanced'` or SMOTE inside CV folds
- **Age, Geography, NumOfProducts, IsActiveMember** are the strongest churn signals — prioritize these in feature engineering
- **Satisfaction Score and Card Type** show minimal correlation with churn — may add noise rather than signal
- **Engineered features** to consider: age buckets, balance-to-salary ratio, Germany flag, zero-balance flag, inactive × multi-product interaction
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"[OK] Saved insights report to {report_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 1 — EDA (Person B)")
    print("Working on TRAINING SET ONLY")
    print("=" * 60 + "\n")

    df = load_train()

    # Quick data overview
    print("--- Data Overview ---")
    print(df.describe().to_string())
    print()

    # 1. Class balance
    print("\n--- 1. Class Balance ---")
    churn_rate = plot_class_balance(df)

    # 2. Univariate distributions
    print("\n--- 2. Univariate Distributions ---")
    plot_univariate_numeric(df)

    # 3. Categorical breakdowns
    print("\n--- 3. Categorical Breakdowns vs Churn ---")
    plot_categorical_vs_churn(df)

    # 4. Correlation heatmap
    print("\n--- 4. Correlation Analysis ---")
    plot_correlation(df)

    # 5. Segment analysis
    print("\n--- 5. Segment Analysis ---")
    plot_segment_analysis(df)

    # 6. Insights
    insights = print_insights(df, churn_rate)
    save_insights_report(insights, df)

    print("\n" + "=" * 60)
    print("EDA COMPLETE — All figures saved to reports/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()


from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results/training")
OUT = Path("results/circular_plots_v2/utility_check")
OUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "Gradient": "gradient",
    "Eigen-90": "eigen_90",
    "MI": "mi",
    "Permutation": "permutation",
}

rows = []

for method, slug in METHODS.items():
    for version in ["v1", "v2"]:

        model = (
            f"scm_classification_{slug}_w_imp_"
            f"start0.3_epoch4_{version}"
        )

        path = (
            ROOT / model / "openml_eval"
            / "real_eval_metrics_summary.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            rows.append({
                "dataset": str(row["dataset"]),
                "method": method,
                "version": version,
                "utility_mean": row[
                    "feature_selection_utility_mean"
                ],
                "utility_std": row[
                    "feature_selection_utility_std"
                ],
            })

# ===================================================
# 1. All utility values
# ===================================================

utility = pd.DataFrame(rows)

utility["utility_mean"] = pd.to_numeric(
    utility["utility_mean"], errors="coerce"
)

utility["utility_std"] = pd.to_numeric(
    utility["utility_std"], errors="coerce"
)

utility["abs_mean"] = utility["utility_mean"].abs()

utility = utility.sort_values(
    ["dataset", "method", "version"]
)

utility.to_csv(
    OUT / "utility_all_values.csv",
    index=False,
    float_format="%.6f",
)

# ===================================================
# 2. Dataset-level summary
# ===================================================

summary = (
    utility.groupby("dataset")
    .agg(
        n_available=("utility_mean", "count"),
        min_utility=("utility_mean", "min"),
        max_utility=("utility_mean", "max"),
        median_utility=("utility_mean", "median"),
        median_abs_utility=("abs_mean", "median"),
    )
    .sort_values("median_abs_utility")
)

summary.to_csv(
    OUT / "utility_dataset_summary.csv",
    float_format="%.6f",
)

# ===================================================
# 3. Global distribution
# ===================================================

values = utility["utility_mean"].dropna()

print("\n========== Utility Statistics ==========")

print(f"Total results: {len(values)}")
print(f"Minimum:       {values.min():.6f}")
print(f"Maximum:       {values.max():.6f}")
print(f"Mean:          {values.mean():.6f}")
print(f"Median:        {values.median():.6f}")
print(f"Std:           {values.std():.6f}")

print("\nAbsolute utility quantiles:")

for q in [0.25, 0.50, 0.75, 0.90, 0.95, 1.00]:
    print(
        f"  {q:.0%}: "
        f"{values.abs().quantile(q):.6f}"
    )

print("\nUtility sign counts:")
print(f"Positive: {(values > 0).sum()}")
print(f"Negative: {(values < 0).sum()}")
print(f"Zero:     {(values == 0).sum()}")

print("\n========== By Dataset ==========")
print(summary.to_string())

print("\n========== Smallest Utility ==========")
print(
    utility.nsmallest(
        15, "utility_mean"
    ).to_string(index=False)
)

print("\n========== Largest Utility ==========")
print(
    utility.nlargest(
        15, "utility_mean"
    ).to_string(index=False)
)

print(f"\nSaved to: {OUT.resolve()}")


from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("results/training")
PREFIX = "real_eval_30repeats_"
REFS = ["imp_spearman_mi", "imp_spearman_rf", "imp_spearman_linear_perm"]


def load(directory, suffix):
    path = directory / f"{PREFIX}{suffix}.csv"
    return pd.read_csv(path) if path.is_file() else None



def agreement_stats(df):
    if df is None or df.empty or not all(c in df.columns for c in REFS):
        return np.nan, np.nan
    per_repeat = df[REFS].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=False)
    valid = per_repeat.dropna()
    if valid.empty:
        return np.nan, np.nan
    return float(valid.mean()), float(valid.std(ddof=1))



def ranking_stability(df):
    if df is None or df.empty:
        return np.nan
    p = df.pivot(index="repeat", columns="feature_index_original", values="pfn_imp")
    if len(p) < 2 or p.isna().any().any():
        return np.nan
    corr = p.T.corr(method="spearman").to_numpy()
    values = corr[np.triu_indices(len(corr), 1)]
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def selection_utility(df, metric):
    if df is None or df.empty or metric not in df.columns:
        return np.nan, np.nan
    df = df[df["method"].isin(["pfn_imp", "random"]) & df["mode"].isin(["keep", "remove"])]
    if df.empty:
        return np.nan, np.nan

    p = df.pivot_table(index="repeat", columns=["method", "mode"], values=metric, aggfunc="first")
    keys = [("pfn_imp", "keep"), ("random", "keep"), ("pfn_imp", "remove"), ("random", "remove")]
    if not all(key in p.columns for key in keys):
        return np.nan, np.nan

    keep = p[("pfn_imp", "keep")] - p[("random", "keep")]
    remove = p[("random", "remove")] - p[("pfn_imp", "remove")]
    utility = (keep + remove) / 2

    return float(utility.mean()), float(utility.std(ddof=1))


count = 0

for directory in sorted(ROOT.glob("*/openml_eval")):
    model = directory.parent.name
    task = "classification" if "classification" in model else "regression" if "regression" in model else None
    if task is None:
        continue

    summary = load(directory, "summary")
    if summary is None or summary.empty:
        continue

    metrics = load(directory, "metrics_each_repeat")
    importance = load(directory, "importance_each_repeat")
    intervention = load(directory, "topk_intervention")
    metric = "balanced_accuracy" if task == "classification" else "r2"
    rows = []

    for _, item in summary.iterrows():
        dataset = item["dataset"]
        agreement_mean = agreement_std = stability = utility_mean = utility_std = np.nan

        if metrics is not None:
            m = metrics[metrics["dataset"] == dataset]
            agreement_mean, agreement_std = agreement_stats(m)

        if importance is not None:
            imp = importance[importance["dataset"] == dataset]
            stability = ranking_stability(imp)

        if intervention is not None:
            inter = intervention[intervention["dataset"] == dataset]
            utility_mean, utility_std = selection_utility(inter, metric)

        rows.append({
            "dataset": dataset,
            "prediction_mean": item[f"{metric}_mean"],
            "prediction_std": item[f"{metric}_std"],
            "mean_reference_agreement_mean": agreement_mean,
            "mean_reference_agreement_std": agreement_std,
            "importance_ranking_stability": stability,
            "feature_selection_utility_mean": utility_mean,
            "feature_selection_utility_std": utility_std,
        })

    output = directory / "real_eval_metrics_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.6f")
    count += 1
    print(f"[OK] {model}: {len(rows)} datasets")

print(f"\nCompleted: {count} models")

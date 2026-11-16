import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/evaluation")
OUT_DIR = Path("results/figures")

PRIORS = ["baseline", "depth", "sparsity", "mechanism", "noise", "missing", "features"]

METHODS = {
    "gradient": "Gradient",
    #"eigen_top1": "Eigen Top-1",
    "eigen_90": "Eigen-90",
    "mi": "MI",
    #"marginal": "Marginal",
    "permutation": "Permutation",
}

TASKS = ["classification", "regression"]
TOPKS = [25, 50, 75]


def collect_topk_metrics(prior, start):
    path = RESULTS_DIR / f"extended_eval_{prior}.csv"
    data = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["epochs"]) != 4:
                continue
            if not np.isclose(float(row["importance_start_frac"]), start):
                continue

            task = row["task_kind"]
            method = row["importance_method"]
            if task not in TASKS or method not in METHODS:
                continue
            for pct in TOPKS:
                key = (task, method, pct)
                if key in data:
                    raise ValueError(f"Duplicate entry: {key} in {path}")
                std_value = row[f"random_topk_pred_loss_std_{pct}"]
                data[key] = {
                    "gt": float(row[f"gt_topk_pred_loss_{pct}"]),
                    "pred": float(row[f"pred_topk_pred_loss_{pct}"]),
                    "random": float(row[f"random_topk_pred_loss_{pct}"]),
                    "random_std": float(std_value) if std_value else 0.0,
                }
    if not data:
        raise ValueError(f"No matching data in {path}, start={start}")
    return data


def plot_predictive_utility(prior, start, view):
    data = collect_topk_metrics(prior, start)

    methods = list(METHODS.keys())
    labels = list(METHODS.values())
    x = np.arange(len(methods))
    width = 0.25

    if view == "main":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        panels = [(axes[i], task, 25) for i, task in enumerate(TASKS)]
        selected_topks = [25]

    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        panels = [
            (axes[i, j], task, pct)
            for i, task in enumerate(TASKS) for j, pct in enumerate(TOPKS)
        ]
        selected_topks = TOPKS

    task_limits = {}

    for task in TASKS:
        values = []
        for method in methods:
            for pct in selected_topks:
                v = data[(task, method, pct)]
                values.extend([v["gt"], v["pred"], v["random"] + v["random_std"],])

        pad = 0.08 * (max(values) - min(values))
        task_limits[task] = (max(0, min(values) - pad), max(values) + pad,)

    for ax, task, pct in panels:
        gt_values = [data[(task, method, pct)]["gt"] for method in methods]
        pred_values = [data[(task, method, pct)]["pred"] for method in methods]
        random_values = [data[(task, method, pct)]["random"] for method in methods]
        random_stds = [data[(task, method, pct)]["random_std"] for method in methods]
        ax.bar(x - width, gt_values, width, label="GT Top-k")
        ax.bar(x, pred_values, width, label="Pred Top-k")
        ax.bar(x + width, random_values, width,  yerr=random_stds, capsize=3, label="Random Top-k")
        ax.set_title(f"{task.capitalize()} — Top-{pct}%")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Top-k Prediction Loss")
        ax.set_ylim(*task_limits[task])
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    # Shared legend
    first_ax = panels[0][0]
    handles, legend_labels = first_ax.get_legend_handles_labels()

    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=10, frameon=False,)
    if view == "main":
        fig.suptitle(f"Predictive Utility — Baseline — Start = {start:g}", fontsize=14,)
        filename = (f"predictive_utility_main_start{start:g}_epoch4.png")
    else:
        fig.suptitle(f"Predictive Utility — {prior.capitalize()} — Start = {start:g}", fontsize=14,)
        filename = (f"predictive_utility_{prior}_start{start:g}_epoch4.png")

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["main", "appendix", "both"], default="both")
    parser.add_argument("--start", type=float, choices=[0.0, 0.15, 0.3], default=0.3)
    parser.add_argument("--prior", choices=PRIORS + ["all"], default="all")
    args = parser.parse_args()

    if args.view in ("main", "both"):
        plot_predictive_utility(prior="baseline", start=args.start, view="main",)

    if args.view in ("appendix", "both"):
        priors = PRIORS if args.prior == "all" else [args.prior]
        for prior in priors:
            plot_predictive_utility(prior=prior, start=args.start, view="appendix")
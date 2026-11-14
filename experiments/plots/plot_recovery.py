
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
    "eigen_90": "Eigen-90",
    "mi": "MI",
    "permutation": "Permutation",
}

STARTS = [(0.0, "Start = 0"), (0.15, "Start = 0.15"), (0.3, "Start = 0.3")]
TASKS = [("classification", "Classification"), ("regression", "Regression")]

METRICS = [
    ("importance_pearson", "Pearson Correlation"),
    ("importance_topk_overlap_25", "Top-25% Overlap"),
    ("importance_topk_overlap_50", "Top-50% Overlap"),
]


def collect_metrics(prior):
    path = RESULTS_DIR / f"extended_eval_{prior}.csv"
    data = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "task_kind",
            "importance_method",
            "importance_start_frac",
            "epochs",
            *(metric for metric, _ in METRICS),
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        for row in reader:
            if float(row["epochs"]) != 4:
                continue

            task = row["task_kind"]
            method = row["importance_method"]
            start = float(row["importance_start_frac"])

            if task not in dict(TASKS):
                continue
            if method not in METHODS:
                continue
            if start not in [s for s, _ in STARTS]:
                continue

            key = (task, method, start)

            if key in data:
                raise ValueError(f"Duplicate entry: {key} in {path}")

            data[key] = {metric: float(row[metric]) for metric, _ in METRICS}

    if not data:
        raise ValueError(f"No matching data in {path}")

    return data


def plot_recovery(prior):
    data = collect_metrics(prior)

    methods = list(METHODS.keys())
    labels = list(METHODS.values())

    x = np.arange(len(methods))
    width = 0.22

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for row_idx, (metric, metric_label) in enumerate(METRICS):
        for col_idx, (task, task_label) in enumerate(TASKS):

            ax = axes[row_idx, col_idx]

            for i, (start, start_label) in enumerate(STARTS):
                values = [
                    data.get((task, method, start), {}).get(metric, np.nan)
                    for method in methods
                ]
                ax.bar(x + (i - 1) * width, values, width, label=start_label,)

            ax.set_title(f"{task_label} — {metric_label}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(metric_label)

            if metric == "importance_pearson":
                values = [v[metric] for (t, _, _), v in data.items() if t == task]
                ax.set_ylim(-1 if values and min(values) < 0 else 0, 1,)
            else:
                ax.set_ylim(0, 1)

            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=10, frameon=False,)
    fig.suptitle(f"Feature-Importance Recovery — {prior.capitalize()}", fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"recovery_{prior}_epoch4.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", choices=PRIORS + ["all"], default="all")
    args = parser.parse_args()
    priors = PRIORS if args.prior == "all" else [args.prior]

    for prior in priors:
        plot_recovery(prior)

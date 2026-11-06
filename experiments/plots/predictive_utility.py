import os
import csv
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "results/figures"

METHODS = {
    "gradient": "Gradient",
    #"eigen_top1": "Eigen Top-1",
    "eigen_90": "Eigen-90",
    "mi": "MI",
    #"marginal": "Marginal",
    "permutation": "Permutation",
}

STARTS = [(0.0, "Start = 0"), (0.15, "Start = 0.15"), (0.3, "Start = 0.3")]



def collect_topk_metrics(prior, pct):
    data = {"classification": {}, "regression": {}}

    path = f"results/evaluation/extended_eval_{prior}.csv"

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            task = row["task_kind"]
            method = row["importance_method"]
            start = float(row["importance_start_frac"])

            data[task][(method, start)] = {
                "gt": float(row[f"gt_topk_pred_loss_{pct}"]),
                "pred": float(row[f"pred_topk_pred_loss_{pct}"]),
                "random": float(row[f"random_topk_pred_loss_{pct}"]),
                "random_std": float(row[f"random_topk_pred_loss_std_{pct}"]),
            }

    return data


def plot_predictive_utility(prior, pct):
    data = collect_topk_metrics(prior, pct)
    methods = list(METHODS.keys())
    labels = [METHODS[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.25

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    panels = [
        (axes[i, j], task, start, f"{task.capitalize()} — {label}")
        for i, task in enumerate(("classification", "regression"))
        for j, (start, label) in enumerate(STARTS)
    ]

    task_limits = {}
    for task in ("classification", "regression"):
        values = []
        for method in methods:
            for start, _ in STARTS:
                values.append(data[task][(method, start)]["gt"])
                values.append(data[task][(method, start)]["pred"])
                values.append(data[task][(method, start)]["random"] + data[task][(method, start)]["random_std"])
        pad = 0.08 * (max(values) - min(values))
        task_limits[task] = (max(0, min(values) - pad), max(values) + pad)

    for ax, task, start, title in panels:
        gt_values = [data[task][(method, start)]["gt"] for method in methods]
        pred_values = [data[task][(method, start)]["pred"] for method in methods]
        random_values = [data[task][(method, start)]["random"] for method in methods]
        random_stds = [data[task][(method, start)]["random_std"] for method in methods]

        ax.bar(x - width, gt_values, width, label="GT Top-k")
        ax.bar(x, pred_values, width, label="Pred Top-k")
        ax.bar(x + width, random_values, width, yerr=random_stds, capsize=3, label="Random Top-k")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(*task_limits[task])
        ax.grid(axis="y", alpha=0.3)

    axes[0, 0].set_ylabel("Top-k Prediction Loss")
    axes[1, 0].set_ylabel("Top-k Prediction Loss")
    axes[0, 1].set_ylabel("Top-k Prediction Loss")
    axes[1, 1].set_ylabel("Top-k Prediction Loss")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=10, frameon=False)

    fig.suptitle(f"Predictive Utility — {prior.capitalize()} — Top {pct}%", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"predictive_utility_{prior}_top{pct}_epoch4.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for prior in [
        "baseline",
        "depth",
        "sparsity",
        "mechanism",
        "noise",
        "missing",
        "features",
    ]:
        for pct in (25, 50, 75):
            plot_predictive_utility(prior, pct)
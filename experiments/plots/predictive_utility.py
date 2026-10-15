import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results/training"
RANDOM_TOPK_PATH = "results/evaluation/random_topk_results.csv"
OUT_DIR = "results/figures"

METHODS = {
    "gradient": "Gradient",
    "eigen_top1": "Eigen Top-1",
    "eigen_90": "Eigen-90",
    "mi": "MI",
    "marginal": "Marginal",
    "permutation": "Permutation",
}

STARTS = [("start0", "Start = 0"), ("start0.3", "Start = 0.3")]

def read_random_topk_metrics(path):
    data = {}

    with open(path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (row["task"], row["method"], row["start"])
            data[key] = {
                "random": float(row["random_topk_pred_loss"]),
                "random_std": float(row["random_topk_pred_loss_std"]),
            }

    return data

def read_best_topk_metrics(path):
    best_step = None
    best_pattern = re.compile(r"\[best\]\s+step\s+(\d+)")
    val_pattern = re.compile(r"\[val\]\s+step\s+(\d+)")
    gt_pattern = re.compile(r"\|\s+gt_topk_pred_loss\s+([\d.eE+-]+)")
    pred_pattern = re.compile(r"\|\s+pred_topk_pred_loss\s+([\d.eE+-]+)")

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        m = best_pattern.search(line)
        if m:
            best_step = int(m.group(1))

    if best_step is None:
        raise ValueError(f"No [best] entry found: {path}")

    for line in lines:
        m = val_pattern.search(line)
        if m and int(m.group(1)) == best_step:
            gt_match = gt_pattern.search(line)
            pred_match = pred_pattern.search(line)
            if gt_match is None or pred_match is None:
                raise ValueError(f"Top-k metrics not found at best step {best_step}: {path}")
            return float(gt_match.group(1)), float(pred_match.group(1))

    raise ValueError(f"No [val] entry found for best step {best_step}: {path}")


def collect_topk_metrics():
    data = {"classification": {}, "regression": {}}
    random_data = read_random_topk_metrics(RANDOM_TOPK_PATH)
    for task in data:
        for method in METHODS:
            for start, _ in STARTS:
                path = os.path.join(RESULTS_DIR, f"scm_{task}_{method}_w_imp_{start}", "train_log.txt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing: {path}")

                gt_loss, pred_loss = read_best_topk_metrics(path)
                key = (task, method, start)
                if key not in random_data:
                    raise ValueError(f"Random Top-k result not found: {key}")
                data[task][(method, start)] = {
                    "gt": gt_loss,
                    "pred": pred_loss,
                    "random": random_data[key]["random"],
                    "random_std": random_data[key]["random_std"],
                }
    return data

def plot_predictive_utility():
    data = collect_topk_metrics()
    methods = list(METHODS.keys())
    labels = [METHODS[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    panels = [
        (axes[0, 0], "classification", "start0", "Classification — Start = 0"),
        (axes[0, 1], "classification", "start0.3", "Classification — Start = 0.3"),
        (axes[1, 0], "regression", "start0", "Regression — Start = 0"),
        (axes[1, 1], "regression", "start0.3", "Regression — Start = 0.3"),
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

    fig.suptitle("Predictive Utility of Selected Features", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "predictive_utility_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_predictive_utility()
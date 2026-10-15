import os
import re
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results/training"
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


def read_importance_loss(path):
    steps, losses = [], []
    pattern = re.compile(r"\[val\]\s+step\s+(\d+).*?\|\s+importance_loss\s+([\d.eE+-]+)")
    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses


def read_importance_start(path):
    pattern = re.compile(r"\[importance supervision ON\]\s+step=(\d+)")
    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return int(m.group(1))
    return None


def read_best_recovery_metrics(path):
    best_step = None
    best_pattern = re.compile(r"\[best\]\s+step\s+(\d+)")
    val_pattern = re.compile(r"\[val\]\s+step\s+(\d+)")
    pearson_pattern = re.compile(r"\|\s+importance_pearson\s+([\d.eE+-]+)")
    topk_pattern = re.compile(r"\|\s+importance_topk_overlap\s+([\d.eE+-]+)")

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
            pearson_match = pearson_pattern.search(line)
            topk_match = topk_pattern.search(line)
            if pearson_match is None or topk_match is None:
                raise ValueError(f"Recovery metrics not found at best step {best_step}: {path}")
            return float(pearson_match.group(1)), float(topk_match.group(1))

    raise ValueError(f"No [val] entry found for best step {best_step}: {path}")


def plot_importance_loss(task):
    data = {}
    all_steps = []
    all_losses = []

    for start, _ in STARTS:
        data[start] = {}
        for method, label in METHODS.items():
            path = os.path.join(RESULTS_DIR, f"scm_{task}_{method}_w_imp_{start}", "train_log.txt")
            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue

            steps, losses = read_importance_loss(path)
            if not steps:
                print(f"No importance-loss data: {path}")
                continue

            data[start][method] = (steps, losses)
            all_steps.extend(steps)
            all_losses.extend(losses)

    if not all_steps:
        raise ValueError(f"No importance-loss data found for {task}")

    loss_pad = 0.03 * (max(all_losses) - min(all_losses))
    loss_ylim = (max(0, min(all_losses) - loss_pad), max(all_losses) + loss_pad)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, (start, panel_title) in zip(axes, STARTS):
        for method, label in METHODS.items():
            if method not in data[start]:
                continue

            steps, losses = data[start][method]
            ax.plot(steps, losses, marker="o", markersize=3.5, linewidth=1.8, label=label)

        if start == "start0.3":
            reference_path = os.path.join(RESULTS_DIR, f"scm_{task}_gradient_w_imp_{start}", "train_log.txt")
            start_step = read_importance_start(reference_path)
            if start_step is not None:
                ax.axvline(start_step, color="gray", linestyle=":", linewidth=1.5)

        ax.set_title(panel_title)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Validation Importance Loss")
        ax.set_xlim(0, max(all_steps))
        ax.set_ylim(*loss_ylim)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8)

    title = "Classification" if task == "classification" else "Regression"
    fig.suptitle(f"{title} — Importance-Loss Dynamics", fontsize=14)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{task}_importance_loss_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def collect_recovery_metrics():
    data = {"classification": {}, "regression": {}}

    for task in data:
        for method in METHODS:
            for start, _ in STARTS:
                path = os.path.join(RESULTS_DIR, f"scm_{task}_{method}_w_imp_{start}", "train_log.txt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing: {path}")

                pearson, topk = read_best_recovery_metrics(path)
                data[task][(method, start)] = {"pearson": pearson, "topk": topk}

    return data


def plot_final_recovery():
    data = collect_recovery_metrics()
    methods = list(METHODS.keys())
    labels = [METHODS[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)

    panels = [
        (axes[0, 0], "classification", "pearson", "Classification — Pearson Correlation"),
        (axes[0, 1], "regression", "pearson", "Regression — Pearson Correlation"),
        (axes[1, 0], "classification", "topk", "Classification — Top-k Overlap"),
        (axes[1, 1], "regression", "topk", "Regression — Top-k Overlap"),
    ]

    for ax, task, metric, title in panels:
        start0 = [data[task][(method, "start0")][metric] for method in methods]
        start03 = [data[task][(method, "start0.3")][metric] for method in methods]

        ax.bar(x - width / 2, start0, width, label="Start = 0")
        ax.bar(x + width / 2, start03, width, label="Start = 0.3")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    axes[0, 0].set_ylabel("Pearson Correlation")
    axes[1, 0].set_ylabel("Top-k Overlap")

    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="lower center", ncol=2, fontsize=10, frameon=False)

    fig.suptitle("Feature-Importance Recovery", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "importance_recovery_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # plot_importance_loss("classification")
    # plot_importance_loss("regression")
    plot_final_recovery()
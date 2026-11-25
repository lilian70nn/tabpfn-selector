
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


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
    ("importance_pearson", "Pearson correlation"),
    ("importance_topk_overlap_25", "Top-25% overlap"),
    ("importance_topk_overlap_50", "Top-50% overlap"),
]

METHOD_COLORS = {
    "gradient": "#345E87",
    "eigen_90": "#58988C",
    "mi": "#C47B5C",
    "permutation": "#806B9F",
}

START_COLORS = {
    0.0: "#527C9C",
    0.15: "#C58C66",
    0.3: "#57988A",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def collect_metrics(prior):
    path = RESULTS_DIR / f"extended_eval_{prior}.csv"
    data = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"task_kind", "importance_method", "importance_start_frac", "epochs", *(m for m, _ in METRICS)}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        for row in reader:
            if float(row["epochs"]) != 4:
                continue

            task, method = row["task_kind"], row["importance_method"]
            start_value = float(row["importance_start_frac"])

            if task not in dict(TASKS) or method not in METHODS:
                continue

            start = next((s for s, _ in STARTS if np.isclose(start_value, s)), None)
            if start is None:
                continue

            key = (task, method, start)
            if key in data:
                raise ValueError(f"Duplicate entry: {key} in {path}")

            data[key] = {metric: float(row[metric]) for metric, _ in METRICS}

    if not data:
        raise ValueError(f"No matching data in {path}")

    return data


def style_axis(ax, row_idx, col_idx, x, labels, pearson_min, rotation=0):
    ax.set_ylim(pearson_min if row_idx == 0 else 0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.5 if row_idx == 0 and pearson_min < 0 else 0.25))

    ax.grid(axis="y", color="#E3E6E9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B9BEC4")
    ax.spines["bottom"].set_color("#B9BEC4")
    ax.tick_params(axis="y", colors="#555555", length=3, width=0.7)
    ax.tick_params(axis="x", colors="#444444", length=0, pad=8)

    if col_idx == 0:
        ax.set_ylabel(METRICS[row_idx][1], labelpad=10)

    ax.set_xticks(x)
    if row_idx == len(METRICS) - 1:
        ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def get_pearson_min(data):
    values = [v["importance_pearson"] for v in data.values()]
    return -1 if any(np.isfinite(v) and v < 0 for v in values) else 0


def save_figure(fig, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_recovery(prior):
    data = collect_metrics(prior)
    methods = list(METHODS)
    x = np.arange(len(methods))
    width = 0.23
    pearson_min = get_pearson_min(data)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8.5))

    for row_idx, (metric, _) in enumerate(METRICS):
        for col_idx, (task, task_label) in enumerate(TASKS):
            ax = axes[row_idx, col_idx]

            for i, (start, _) in enumerate(STARTS):
                values = [data.get((task, method, start), {}).get(metric, np.nan) for method in methods]
                ax.bar(x + (i - 1) * width, values, width, color=START_COLORS[start], edgecolor="white", linewidth=0.5)

            style_axis(ax, row_idx, col_idx, x, list(METHODS.values()), pearson_min)
            ax.set_xlim(-0.55, len(methods) - 0.45)

            if row_idx == 0:
                ax.set_title(task_label, fontweight="semibold", pad=13)

    handles = [Patch(facecolor=START_COLORS[s], label=label) for s, label in STARTS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3, frameon=False, columnspacing=2.5)
    fig.suptitle(f"Feature-Importance Recovery — {prior.capitalize()}", fontsize=14, fontweight="semibold", y=0.98)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.13, hspace=0.22, wspace=0.20)

    save_figure(fig, f"recovery_{prior}_epoch4.pdf")


def plot_recovery_across_settings(start=0.3):
    all_data = {prior: collect_metrics(prior) for prior in PRIORS}
    x = np.arange(len(PRIORS))
    width = 0.19

    all_rows = [v for prior_data in all_data.values() for v in prior_data.values()]
    pearson_min = -1 if any(v["importance_pearson"] < 0 for v in all_rows) else 0

    fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))

    for row_idx, (metric, _) in enumerate(METRICS):
        for col_idx, (task, task_label) in enumerate(TASKS):
            ax = axes[row_idx, col_idx]

            for i, (method, _) in enumerate(METHODS.items()):
                values = [all_data[prior].get((task, method, start), {}).get(metric, np.nan) for prior in PRIORS]
                offset = (i - (len(METHODS) - 1) / 2) * width
                ax.bar(x + offset, values, width, color=METHOD_COLORS[method], edgecolor="white", linewidth=0.4)

            labels = [p.capitalize() for p in PRIORS]
            style_axis(ax, row_idx, col_idx, x, labels, pearson_min, rotation=28)
            ax.set_xlim(-0.6, len(PRIORS) - 0.4)

            if row_idx == 0:
                ax.set_title(task_label, fontweight="semibold", pad=13)

    handles = [Patch(facecolor=METHOD_COLORS[m], label=label) for m, label in METHODS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False, columnspacing=2.0)
    fig.suptitle(f"Feature-Importance Recovery Across Settings — Start = {start:g}", fontsize=14, fontweight="semibold", y=0.98)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.89, bottom=0.19, hspace=0.23, wspace=0.20)

    save_figure(fig, f"recovery_all_settings_start{start:g}_epoch4.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["all", "per-prior", "across-settings"], default="all")
    parser.add_argument("--prior", choices=PRIORS + ["all"], default="all")
    args = parser.parse_args()

    if args.view in ("all", "per-prior"):
        priors = PRIORS if args.prior == "all" else [args.prior]
        for prior in priors:
            plot_recovery(prior)

    if args.view in ("all", "across-settings"):
        plot_recovery_across_settings(start=0.3)

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

RESULTS_DIR = Path("results/evaluation")
OUT_DIR = Path("results/figures")

PRIORS = ["baseline", "depth", "sparsity", "mechanism", "noise", "missing", "features"]
METHODS = {"gradient": "Gradient", "eigen_90": "Eigen-90", "mi": "MI", "permutation": "Permutation"}
TASKS = ["classification", "regression"]
TOPKS = [25, 50, 75]

SERIES = {
    "gt": ("GT Top-k", "#416C91"),
    "pred": ("Pred Top-k", "#5A9B91"),
    "random": ("Random Top-k", "#C98562"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "savefig.facecolor": "white",
})


def collect_topk_metrics(prior, start):
    path = RESULTS_DIR / f"extended_eval_{prior}.csv"
    data = {}

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if float(row["epochs"]) != 4 or not np.isclose(float(row["importance_start_frac"]), start):
                continue

            task, method = row["task_kind"], row["importance_method"]
            if task not in TASKS or method not in METHODS:
                continue

            for pct in TOPKS:
                key = (task, method, pct)
                if key in data:
                    raise ValueError(f"Duplicate entry: {key} in {path}")

                std = row[f"random_topk_pred_loss_std_{pct}"]
                data[key] = {
                    "gt": float(row[f"gt_topk_pred_loss_{pct}"]),
                    "pred": float(row[f"pred_topk_pred_loss_{pct}"]),
                    "random": float(row[f"random_topk_pred_loss_{pct}"]),
                    "random_std": float(std) if std else 0.0,
                }

    if not data:
        raise ValueError(f"No matching data in {path}, start={start}")
    return data


def get_task_limit(data, task, pcts):
    values = []
    for method in METHODS:
        for pct in pcts:
            v = data[(task, method, pct)]
            values.extend([v["gt"], v["pred"], v["random"] + v["random_std"]])
    return max(values) * 1.10


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7EAED", linewidth=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CED4D9")
    ax.spines["bottom"].set_color("#CED4D9")
    ax.tick_params(axis="both", length=0, pad=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def draw_panel(ax, data, task, pct, index, ymax):
    methods = list(METHODS)
    x = np.arange(len(methods))
    width = 0.23

    for offset, (key, (label, color)) in enumerate(SERIES.items()):
        values = [data[(task, method, pct)][key] for method in methods]
        errors = [data[(task, method, pct)]["random_std"] for method in methods] if key == "random" else None
        ax.bar(x + (offset - 1) * width, values, width=width, color=color, label=label,
               yerr=errors, capsize=2, error_kw={"ecolor": "#505861", "elinewidth": 0.9, "capthick": 0.9}, zorder=3)

    ax.set_title(f"({chr(97 + index)})  {task.capitalize()} · Top-{pct}%", loc="left", fontweight="semibold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(list(METHODS.values()))
    ax.set_xlim(-0.55, len(methods) - 0.45)
    ax.set_ylim(0, ymax)
    style_axis(ax)


def plot_predictive_utility(prior, start, view):
    data = collect_topk_metrics(prior, start)
    pcts = [25, 50] if view == "main" else TOPKS
    ncols = len(pcts)

    figsize = (10.5, 6.5) if view == "main" else (14.2, 6.5)
    fig, axes = plt.subplots(2, ncols, figsize=figsize)

    for i, task in enumerate(TASKS):
        ymax = get_task_limit(data, task, pcts)
        for j, pct in enumerate(pcts):
            ax = axes[i, j]
            draw_panel(ax, data, task, pct, i * ncols + j, ymax)
            if j == 0:
                ax.set_ylabel("Prediction loss")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=3, frameon=False, fontsize=10, columnspacing=2.5, handlelength=1.5)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.15, wspace=0.23, hspace=0.42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"predictive_utility_main_start{start:g}_epoch4.pdf" if view == "main" else f"predictive_utility_{prior}_start{start:g}_epoch4.pdf"
    out_path = OUT_DIR / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", choices=["main", "appendix", "both"], default="main")
    parser.add_argument("--start", type=float, choices=[0.0, 0.15, 0.3], default=0.3)
    parser.add_argument("--prior", choices=PRIORS + ["all"], default="all")
    args = parser.parse_args()

    if args.view in ("main", "both"):
        plot_predictive_utility("baseline", args.start, "main")

    if args.view in ("appendix", "both"):
        priors = PRIORS if args.prior == "all" else [args.prior]
        for prior in priors:
            plot_predictive_utility(prior, args.start, "appendix")
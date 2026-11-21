
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROBE_DIR = Path("results/probing")
EVAL_DIR = Path("results/evaluation")
OUT_DIR = Path("results/figures")

PRIORS = ["baseline", "depth", "sparsity", "mechanism", "noise", "missing", "features"]
TASKS = ["classification", "regression"]
METHODS = {
    "gradient": "Gradient",
    "eigen_90": "Eigen-90",
    "mi": "MI",
    "permutation": "Permutation",
}
GROUPS = {
    "A": "(a) Pred-only · Linear",
    "B": "(b) Pred-only · MLP",
    "C": "(c) Joint · Linear",
    "D": "(d) Joint · MLP (Ref.)",
}
METRICS = [
    ("importance_pearson", "Importance Pearson"),
    ("importance_topk_overlap_25", "Top-25% Overlap"),
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_results():
    data = {}

    for task in TASKS:
        for method in METHODS:
            folders = {
                "A": f"scm_{task}_imp_weight0_epoch4_{task}_{method}_linear_epoch2",
                "B": f"scm_{task}_imp_weight0_epoch4_{task}_{method}_mlp_epoch2",
                "C": f"scm_{task}_{method}_w_imp_start0.3_epoch4_{task}_{method}_linear_epoch2",
            }

            for group, folder in folders.items():
                path = PROBE_DIR / folder / "ood_results.csv"
                for row in read_csv(path):
                    data[(row["prior"], task, method, group)] = row

    for prior in PRIORS:
        path = EVAL_DIR / f"extended_eval_{prior}.csv"

        for row in read_csv(path):
            if row["task_kind"] not in TASKS:
                continue
            if row["importance_method"] not in METHODS:
                continue
            if float(row["epochs"]) != 4:
                continue
            if not np.isclose(float(row["importance_start_frac"]), 0.3):
                continue

            key = (prior, row["task_kind"], row["importance_method"], "D")
            data[key] = row

    return data



COLORS = {
    "A": "#507A9B",
    "B": "#A9C6DA",
    "C": "#4C8A79",
    "D": "#A5CCBD",
}

GROUPS = {
    "A": "Prediction-only + Linear",
    "B": "Prediction-only + MLP",
    "C": "Joint + Linear",
    "D": "Joint + MLP (reference)",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
})


def plot_probing(prior, data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    methods = list(METHODS)
    x = np.arange(len(methods))
    width = 0.19

    for i, (metric, ylabel) in enumerate(METRICS):
        for j, task in enumerate(TASKS):
            ax = axes[i, j]

            for g, (group, label) in enumerate(GROUPS.items()):
                values = [
                    float(data[(prior, task, method, group)][metric])
                    for method in methods
                ]

                ax.bar(
                    x + (g - 1.5) * width,
                    values,
                    width,
                    color=COLORS[group],
                    edgecolor="#5F6971" if group in ("B", "D") else "white",
                    linewidth=0.6,
                    hatch="//" if group in ("B", "D") else None,
                    label=label,
                )

            if i == 0:
                ax.set_title(task.capitalize(), fontweight="semibold")

            if j == 0:
                ax.set_ylabel(ylabel)

            ax.set_xticks(x)
            ax.set_xticklabels(list(METHODS.values()))
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, 0.2))

            ax.grid(axis="y", color="#E5E8EB", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#AAB0B5")
            ax.spines["bottom"].set_color("#AAB0B5")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        f"Importance Representation Probing — {prior.capitalize()}",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"importance_probing_{prior}_start0.3.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ood_summary(data):
    ood_priors = PRIORS[1:]
    methods = list(METHODS)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), constrained_layout=True)

    for i, (metric, ylabel) in enumerate(METRICS):
        matrices = []

        for task in TASKS:
            matrix = np.array([
                [
                    float(data[(prior, task, method, "C")][metric])
                    - float(data[(prior, task, method, "A")][metric])
                    for prior in ood_priors
                ]
                for method in methods
            ])
            matrices.append(matrix)

        limit = max(np.max(np.abs(m)) for m in matrices)
        limit = max(limit, 0.01)

        for j, task in enumerate(TASKS):
            ax = axes[i, j]
            matrix = matrices[j]

            im = ax.imshow(
                matrix,
                cmap="BrBG",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
            )

            ax.set_xticks(range(len(ood_priors)))
            ax.set_xticklabels(
                [p.capitalize() for p in ood_priors],
                rotation=25,
                ha="right",
            )
            ax.set_yticks(range(len(methods)))
            ax.set_yticklabels(list(METHODS.values()))

            if i == 0:
                ax.set_title(task.capitalize(), fontweight="semibold")

            for r in range(len(methods)):
                for c in range(len(ood_priors)):
                    value = matrix[r, c]
                    ax.text(
                        c, r, f"{value:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if abs(value) > 0.65 * limit else "#263238",
                    )

            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.colorbar(
            im,
            ax=axes[i, :],
            shrink=0.8,
            pad=0.02,
            label=f"Δ {ylabel}",
        )

    fig.suptitle(
        "OOD Probing — Joint Linear minus Prediction-only Linear",
        fontsize=14,
        fontweight="semibold",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "importance_probing_ood_summary_start0.3.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    data = load_results()

    for prior in PRIORS:
        plot_probing(prior, data)

    plot_ood_summary(data)


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
    "A": "wo_imp + Linear",
    "B": "wo_imp + MLP",
    "C": "w_imp + Linear",
    "D": "w_imp + MLP",
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


def plot_probing(prior, data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    methods = list(METHODS)
    x = np.arange(len(methods))
    width = 0.2

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
                    label=label,
                )

            ax.set_title(task.capitalize())
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(list(METHODS.values()))
            ax.set_ylim((-1, 1) if i == 0 else (0, 1))
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

    fig.suptitle(f"Importance Probing — {prior.capitalize()}", fontsize=14)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"importance_probing_{prior}_start0.3.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    data = load_results()

    for prior in PRIORS:
        plot_probing(prior, data)

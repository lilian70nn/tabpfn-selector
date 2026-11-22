
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


RESULTS_DIR = Path("results/training")
OUT_DIR = Path("results/figures")

METHODS = {
    "gradient": "Gradient",
    "eigen_top1": "Eigen Top-1",
    "eigen_90": "Eigen-90",
    "mi": "MI",
    "marginal": "Marginal",
    "permutation": "Permutation",
}

STARTS = [
    ("start0", "Start = 0"),
    ("start0.15", "Start = 0.15"),
    ("start0.3", "Start = 0.3"),
]

# Consistent across all figures
COLORS = {
    "gradient": "#326B8C",
    "eigen_top1": "#C16A44",
    "eigen_90": "#7665A7",
    "mi": "#39846D",
    "marginal": "#B44F82",
    "permutation": "#B8902F",
}

MARKERS = {
    "gradient": "o",
    "eigen_top1": "s",
    "eigen_90": "^",
    "mi": "D",
    "marginal": "v",
    "permutation": "P",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def get_log_path(task, backbone, method=None, start=None):
    if method is None:
        folder = f"scm_{task}_imp_weight0_epoch4_{backbone}"
    else:
        folder = (
            f"scm_{task}_{method}_w_imp_"
            f"{start}_epoch4_{backbone}"
        )

    return RESULTS_DIR / folder / "train_log.txt"


def read_loss(path, loss_type):
    steps, losses = [], []

    loss_name = (
        "pred_loss"
        if loss_type == "prediction"
        else "importance_loss"
    )

    pattern = re.compile(
        rf"\[val\]\s+step\s+(\d+)\b.*?"
        rf"\b{loss_name}\s+({FLOAT_PATTERN})"
    )

    if not path.is_file():
        return steps, losses

    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    return steps, losses


def read_importance_start(path):
    pattern = re.compile(
        r"\[importance supervision ON\]"
        r"\s+step\s*=\s*(\d+)",
        re.IGNORECASE,
    )

    if not path.is_file():
        return None

    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return int(match.group(1))

    return None


def load_results(task, loss_type):
    """Load both V1 and V2 to keep axis limits consistent."""
    data = {}

    for backbone in ("v1", "v2"):

        if loss_type == "prediction":
            path = get_log_path(task, backbone)
            steps, losses = read_loss(path, loss_type)

            if steps:
                data[(backbone, "baseline")] = (steps, losses)

        for start, _ in STARTS:
            for method in METHODS:
                path = get_log_path(
                    task, backbone, method, start
                )

                steps, losses = read_loss(path, loss_type)

                if steps:
                    data[(backbone, start, method)] = (
                        steps, losses
                    )

    return data


def get_axis_limits(data, loss_type):
    if not data:
        raise ValueError("No validation-loss data found.")

    all_steps = [
        step
        for steps, _ in data.values()
        for step in steps
    ]

    all_losses = [
        loss
        for _, losses in data.values()
        for loss in losses
    ]

    loss_min = min(all_losses)
    loss_max = max(all_losses)

    pad = max(
        0.04 * (loss_max - loss_min),
        0.02 * abs(loss_max),
        1e-4,
    )

    ymin = loss_min - pad
    ymax = loss_max + pad

    if loss_type == "importance":
        ymin = max(0, ymin)

    return (ymin, ymax), max(all_steps)


def plot_loss(task, loss_type, backbone):
    data = load_results(task, loss_type)
    ylim, xmax = get_axis_limits(data, loss_type)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(13.6, 4.5),
        sharey=True,
    )

    onset_drawn = False

    for ax, (start, panel_title) in zip(axes, STARTS):

        # Importance definitions
        for method, label in METHODS.items():
            key = (backbone, start, method)

            if key not in data:
                continue

            steps, losses = data[key]

            ax.plot(
                steps,
                losses,
                color=COLORS[method],
                linewidth=1.9,
                marker=MARKERS[method],
                markersize=3.2,
                markevery=max(1, len(steps) // 10),
                markerfacecolor="white",
                markeredgewidth=0.9,
                solid_capstyle="round",
                label=label,
                zorder=3,
            )

        # Prediction-only baseline
        if loss_type == "prediction":
            baseline = data.get((backbone, "baseline"))

            if baseline is not None:
                base_steps, base_losses = baseline

                ax.plot(
                    base_steps,
                    base_losses,
                    color="#333333",
                    linewidth=1.9,
                    linestyle="--",
                    label="Importance Weight = 0",
                    zorder=2,
                )

        # Delayed importance supervision
        if start in ("start0.15", "start0.3"):
            reference_path = get_log_path(
                task, backbone, "gradient", start
            )

            start_step = read_importance_start(reference_path)

            if start_step is not None:
                ax.axvline(
                    start_step,
                    color="#9A9A9A",
                    linestyle=":",
                    linewidth=1.4,
                    zorder=1,
                )
                onset_drawn = True

        # Panel formatting
        ax.set_title(
            panel_title,
            fontsize=11,
            fontweight="semibold",
            pad=12,
        )

        ax.set_xlabel("Training step", labelpad=8)
        ax.set_xlim(0, xmax)
        ax.set_ylim(*ylim)

        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, integer=True)
        )

        ax.grid(
            axis="y",
            color="#E4E7EB",
            linewidth=0.8,
        )
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#B5B5B5")
        ax.spines["bottom"].set_color("#B5B5B5")

        ax.tick_params(
            axis="both",
            colors="#555555",
            length=3,
            width=0.7,
        )

    ylabel = (
        "Validation prediction loss"
        if loss_type == "prediction"
        else "Validation importance loss"
    )
    axes[0].set_ylabel(ylabel, labelpad=9)

    # One shared legend
    legend_handles = [
        Line2D(
            [0], [0],
            color=COLORS[method],
            linewidth=1.9,
            marker=MARKERS[method],
            markersize=4,
            markerfacecolor="white",
            label=label,
        )
        for method, label in METHODS.items()
    ]

    if loss_type == "prediction":
        legend_handles.append(
            Line2D(
                [0], [0],
                color="#333333",
                linewidth=1.9,
                linestyle="--",
                label="Importance Weight = 0",
            )
        )

    if onset_drawn:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="#9A9A9A",
                linewidth=1.4,
                linestyle=":",
                label="Importance enabled",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )

    title = task.capitalize()
    loss_title = loss_type.capitalize()

    fig.suptitle(
        f"{title} — {loss_title} Loss ({backbone.upper()})",
        fontsize=13,
        fontweight="semibold",
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.84,
        bottom=0.24,
        wspace=0.12,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{task}_{loss_type}_"
        f"loss_comparison_epoch4_{backbone}.pdf"
    )

    out_path = OUT_DIR / filename

    fig.savefig(
        out_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )

    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-kind",
        choices=["classification", "regression"],
        required=True,
    )
    parser.add_argument(
        "--loss",
        choices=["prediction", "importance"],
        required=True,
    )
    parser.add_argument(
        "--backbone",
        choices=["v1", "v2"],
        required=True,
    )

    args = parser.parse_args()

    plot_loss(
        task=args.task_kind,
        loss_type=args.loss,
        backbone=args.backbone,
    )

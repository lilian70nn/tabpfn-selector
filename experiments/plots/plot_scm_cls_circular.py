
from pathlib import Path
from math import ceil
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path("results/training")
OUT = Path("results/circular_plots")
OUT.mkdir(parents=True, exist_ok=True)

TASK = "classification"
START = "0.3"
EPOCH = 4

METHODS = {
    "Gradient": "gradient",
    "Eigen-90": "eigen_90",
    "MI": "mi",
    "Permutation": "permutation",
}

VERSIONS = ["v1", "v2"]

COLORS = {
    "Gradient": "#477C9D",
    "Eigen-90": "#C58B69",
    "MI": "#5B9B8B",
    "Permutation": "#927FA5",
}

METRICS = [
    {
        "name": "Prediction",
        "mean": "prediction_mean",
        "std": "prediction_std",
    },
    {
        "name": "Agreement",
        "mean": "best_reference_agreement_mean",
        "std": "best_reference_agreement_std",
    },
    {
        "name": "Stability",
        "mean": "importance_ranking_stability",
        "std": None,
    },
    {
        "name": "Utility",
        "mean": "feature_selection_utility_mean",
        "std": "feature_selection_utility_std",
    },
]

# Layout
NCOLS = 5
CELL_SIZE = 3.15

# Polar geometry
R_IN = 0.31
R_OUT = 0.84
SECTOR_GAP = np.deg2rad(5)

# Output
DPI = 250

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


# ============================================================
# 2. Load summary CSVs
# ============================================================

def load_results():
    tables = {}

    for method, slug in METHODS.items():
        for version in VERSIONS:

            model = (
                f"scm_{TASK}_{slug}_w_imp_"
                f"start{START}_epoch{EPOCH}_{version}"
            )

            path = (
                ROOT
                / model
                / "openml_eval"
                / "real_eval_metrics_summary.csv"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing CSV: {path}"
                )

            df = pd.read_csv(path)

            if "dataset" not in df.columns:
                raise ValueError(
                    f"Missing dataset column: {path}"
                )

            df["dataset"] = df["dataset"].astype(str)

            if df["dataset"].duplicated().any():
                raise ValueError(
                    f"Duplicated datasets: {path}"
                )

            tables[(method, version)] = (
                df.set_index("dataset")
            )

            print(
                f"[OK] {method:12s} {version}: "
                f"{len(df)} datasets"
            )

    return tables


# ============================================================
# 3. Data utilities
# ============================================================

def get_dataset_names(tables):
    # Union: retain datasets missing in V1
    datasets = set()

    for df in tables.values():
        datasets.update(df.index.tolist())

    return sorted(datasets, key=str.lower)


def get_value(tables, method, version, dataset, column):
    if column is None:
        return np.nan

    df = tables[(method, version)]

    if dataset not in df.index:
        return np.nan

    if column not in df.columns:
        return np.nan

    value = df.at[dataset, column]

    return (
        float(value)
        if pd.notna(value)
        else np.nan
    )


def lighten(color, amount=0.55):
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def compute_scales(tables):
    """
    Fixed metric scales across all datasets.

    Prediction: [0, 1]
    Agreement: [-1, 1] (Spearman)
    Stability: [-1, 1] (Spearman)
    Utility: symmetric around 0, based on data.
    """

    utility_values = []

    for df in tables.values():

        mean_col = "feature_selection_utility_mean"
        std_col = "feature_selection_utility_std"

        if mean_col not in df.columns:
            continue

        means = pd.to_numeric(
            df[mean_col], errors="coerce"
        )

        if std_col in df.columns:
            stds = pd.to_numeric(
                df[std_col], errors="coerce"
            ).fillna(0)
        else:
            stds = pd.Series(0, index=df.index)

        utility_values.extend(
            (means.abs() + stds).dropna().tolist()
        )

    maximum = max(utility_values, default=0.01)

    # Symmetric scale with a small margin
    utility_limit = max(
        0.01,
        ceil(maximum * 1.10 * 100) / 100
    )

    return {
        "Prediction": (0.0, 1.0),
        "Agreement": (-1.0, 1.0),
        "Stability": (-1.0, 1.0),
        "Utility": (-utility_limit, utility_limit),
    }


# ============================================================
# 4. Polar drawing helpers
# ============================================================

def radial_position(value, vmin, vmax):
    fraction = (value - vmin) / (vmax - vmin)
    return R_IN + fraction * (R_OUT - R_IN)


def draw_arc(ax, radius, start, end, **kwargs):
    theta = np.linspace(start, end, 100)

    ax.plot(
        theta,
        np.full_like(theta, radius),
        **kwargs
    )


def draw_errorbar(
    ax, theta, mean, std,
    vmin, vmax, bar_width
):
    if not np.isfinite(std) or std < 0:
        return

    low = np.clip(mean - std, vmin, vmax)
    high = np.clip(mean + std, vmin, vmax)

    r_low = radial_position(low, vmin, vmax)
    r_high = radial_position(high, vmin, vmax)

    # Radial error line
    ax.plot(
        [theta, theta],
        [r_low, r_high],
        color="#252A2E",
        lw=0.65,
        zorder=8,
        solid_capstyle="round",
    )

    # Small tangential caps
    cap = bar_width * 0.26

    for r in (r_low, r_high):
        ax.plot(
            [theta - cap, theta + cap],
            [r, r],
            color="#252A2E",
            lw=0.65,
            zorder=8,
        )


def draw_sector_guides(
    ax, sector_index, vmin, vmax
):
    sector_size = np.pi / 2

    start = sector_index * sector_size
    end = start + sector_size

    a = start + SECTOR_GAP * 0.6
    b = end - SECTOR_GAP * 0.6

    # Concentric guides
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        radius = R_IN + fraction * (R_OUT - R_IN)

        draw_arc(
            ax, radius, a, b,
            color="#B5BDC1",
            lw=0.45,
            alpha=0.60,
            zorder=1,
        )

    # Zero reference for metrics allowing negatives
    if vmin < 0 < vmax:
        zero_r = radial_position(0, vmin, vmax)

        draw_arc(
            ax, zero_r, a, b,
            color="#69747B",
            lw=0.8,
            ls=(0, (2.5, 2)),
            alpha=0.9,
            zorder=2,
        )

    # Outer sector boundary
    draw_arc(
        ax, 0.90, a, b,
        color="#515C62",
        lw=0.95,
        zorder=3,
    )

    # Sector separators
    for angle in (a, b):
        ax.plot(
            [angle, angle],
            [R_IN - 0.015, 0.90],
            color="#899398",
            lw=0.50,
            zorder=2,
        )


def draw_bar(
    ax, theta, mean, std,
    vmin, vmax, width, color
):
    if not np.isfinite(mean):
        # Missing results: keep this angular slot empty
        return

    mean = np.clip(mean, vmin, vmax)

    r_mean = radial_position(mean, vmin, vmax)
    r_zero = radial_position(
        np.clip(0.0, vmin, vmax),
        vmin, vmax
    )

    bottom = min(r_mean, r_zero)
    height = abs(r_mean - r_zero)

    if height > 0:
        ax.bar(
            theta,
            height,
            width=width,
            bottom=bottom,
            color=color,
            edgecolor="#586168",
            linewidth=0.32,
            align="center",
            zorder=5,
        )

    if np.isfinite(std):
        draw_errorbar(
            ax, theta, mean, std,
            vmin, vmax, width
        )


# ============================================================
# 5. One circular plot per Dataset
# ============================================================

def draw_dataset(ax, dataset, tables, scales):

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.13)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")

    sector_size = np.pi / 2

    for s, metric in enumerate(METRICS):

        name = metric["name"]
        mean_col = metric["mean"]
        std_col = metric["std"]

        vmin, vmax = scales[name]

        draw_sector_guides(
            ax, s, vmin, vmax
        )

        # Eight positions in each sector
        usable = sector_size - 2 * SECTOR_GAP
        step = usable / 8
        bar_width = step * 0.78

        sector_start = (
            s * sector_size + SECTOR_GAP
        )

        for method_index, method in enumerate(METHODS):

            for version_index, version in enumerate(VERSIONS):

                slot = method_index * 2 + version_index

                theta = (
                    sector_start
                    + (slot + 0.5) * step
                )

                mean = get_value(
                    tables, method, version,
                    dataset, mean_col
                )

                std = get_value(
                    tables, method, version,
                    dataset, std_col
                )

                base_color = COLORS[method]

                # V1 = light, V2 = saturated
                if version == "v1":
                    color = lighten(base_color, 0.55)
                else:
                    color = base_color

                draw_bar(
                    ax,
                    theta=theta,
                    mean=mean,
                    std=std,
                    vmin=vmin,
                    vmax=vmax,
                    width=bar_width,
                    color=color,
                )

        # Metric label outside each quadrant
        mid_angle = (
            s * sector_size + sector_size / 2
        )

        ax.text(
            mid_angle,
            1.005,
            name,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="semibold",
            color="#343D43",
        )

    # Inner circular boundary
    theta_circle = np.linspace(
        0, 2 * np.pi, 360
    )

    ax.plot(
        theta_circle,
        np.full_like(theta_circle, R_IN),
        color="#69747B",
        lw=0.9,
        zorder=9,
    )

    # Dataset name in the center
    wrapped = textwrap.wrap(
        str(dataset),
        width=13,
        break_long_words=True,
        break_on_hyphens=True,
    )

    if len(wrapped) > 3:
        wrapped = wrapped[:3]
        wrapped[-1] += "…"

    ax.text(
        0.5,
        0.5,
        "\n".join(wrapped),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#28333A",
        linespacing=1.1,
        zorder=10,
    )


# ============================================================
# 6. All Classification datasets in one figure
# ============================================================

def main():

    tables = load_results()
    datasets = get_dataset_names(tables)
    scales = compute_scales(tables)

    n = len(datasets)
    nrows = ceil(n / NCOLS)

    print(f"\nTotal unique datasets: {n}")

    for name, (vmin, vmax) in scales.items():
        print(f"{name:12s}: [{vmin:.3f}, {vmax:.3f}]")

    fig, axes = plt.subplots(
        nrows,
        NCOLS,
        figsize=(
            NCOLS * CELL_SIZE,
            nrows * CELL_SIZE + 1.3,
        ),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )

    axes = axes.ravel()

    for i, dataset in enumerate(datasets):
        draw_dataset(
            axes[i], dataset, tables, scales
        )

    # Hide unused subplot slots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "SCM Classification Benchmark",
        fontsize=19,
        fontweight="bold",
        y=0.988,
        color="#25323A",
    )

    fig.text(
        0.5,
        0.970,
        "Start = 0.3  |  OpenML datasets  |  Mean ± Std",
        ha="center",
        fontsize=10,
        color="#67737B",
    )

    legend_handles = [
        Patch(
            facecolor=COLORS[method],
            edgecolor="none",
            label=method,
        )
        for method in METHODS
    ]

    legend_handles += [
        Patch(
            facecolor="#CBD4D8",
            edgecolor="#74828A",
            label="V1 (light)",
        ),
        Patch(
            facecolor="#546B79",
            edgecolor="#546B79",
            label="V2 (dark)",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=6,
        frameon=False,
        fontsize=9,
        columnspacing=2.0,
        handlelength=1.5,
    )

    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        top=0.915,
        bottom=0.07,
        wspace=0.14,
        hspace=0.15,
    )

    scale_text = "   |   ".join(
        f"{name}: {vmin:g} to {vmax:g}"
        for name, (vmin, vmax) in scales.items()
    )

    fig.text(
        0.5,
        0.043,
        "Metric-specific radial scales: " + scale_text,
        ha="center",
        fontsize=8.5,
        color="#515C62",
    )

    fig.text(
        0.5,
        0.024,
        "Each method has two adjacent bars: V1 (light), "
        "V2 (dark). Missing results are left blank. "
        "Error bars show ±1 repeat-level Std.",
        ha="center",
        fontsize=8,
        color="#68747A",
    )

    output_base = (
        OUT / "scm_classification_circular_start0.3"
    )

    fig.savefig(
        f"{output_base}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        f"{output_base}.png",
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {output_base}.pdf")
    print(f"Saved: {output_base}.png")


if __name__ == "__main__":
    main()

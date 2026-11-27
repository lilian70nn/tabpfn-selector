
from pathlib import Path
from math import ceil
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path("results/training")

START = "0.3"
EPOCH = 4

METHODS = {
    "Gradient": "gradient",
    "Eigen-90": "eigen_90",
    "MI": "mi",
    "Permutation": "permutation",
}

COLORS = {
    "Gradient": "#6B9CB3",
    "Eigen-90": "#C58F76",
    "MI": "#75AA9E",
    "Permutation": "#9B8AAC",
}


# Layout
NCOLS = 4
CELL_SIZE = 4.3

# Detailed example: show bar-end numerical labels
SHOW_VALUES_IN_OVERVIEW = False
SHOW_VALUES_IN_DETAIL = True

# Utility scaling:
# "robust": show the typical range, mark clipped values
# "full": show the entire range without clipping means
UTILITY_SCALE_MODE = "robust"
UTILITY_QUANTILE = 0.95



# Polar geometry
R_IN = 0.31
R_OUT = 0.88
SECTOR_GAP = np.deg2rad(4.0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


# ============================================================
# LOAD DATA
# ============================================================


def compute_keep_advantage(intervention_df):

    required = {
        "dataset",
        "repeat",
        "method",
        "mode",
        "k_frac",
        "balanced_accuracy",
    }

    missing = required - set(intervention_df.columns)
    if missing:
        raise ValueError(
            f"Missing intervention columns: {missing}"
        )

    df = intervention_df.copy()
    df["dataset"] = df["dataset"].astype(str)
    df["k_frac"] = pd.to_numeric(
        df["k_frac"], errors="coerce"
    )
    df["balanced_accuracy"] = pd.to_numeric(
        df["balanced_accuracy"], errors="coerce"
    )

    # Only Top 20% keep experiments
    df = df[
        (df["mode"] == "keep")
        & df["method"].isin(["pfn_imp", "random"])
        & np.isclose(df["k_frac"], 0.2)
    ].copy()

    # Each dataset/repeat/method should have one value
    keys = ["dataset", "repeat", "method"]

    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate Keep results for the same "
            "dataset/repeat/method."
        )

    pivot = df.pivot(
        index=["dataset", "repeat"],
        columns="method",
        values="balanced_accuracy",
    )

    if not {"pfn_imp", "random"}.issubset(pivot.columns):
        raise ValueError(
            "Missing pfn_imp or random Keep results."
        )

    # Paired difference for each repeat
    advantage = (
        pivot["pfn_imp"] - pivot["random"]
    ).dropna()

    advantage = advantage.rename(
        "keep_advantage"
    ).reset_index()

    # Mean ± sample Std across repeats
    stats = (
        advantage.groupby("dataset")["keep_advantage"]
        .agg(["mean", "std", "count"])
        .rename(columns={
            "mean": "keep_advantage_mean",
            "std": "keep_advantage_std",
            "count": "keep_n_pairs",
        })
    )

    return stats



def load_tables():
    tables = {}

    for method, slug in METHODS.items():
        for version in ("v1", "v2"):

            folder = (
                f"scm_classification_{slug}_w_imp_"
                f"start{START}_epoch{EPOCH}_{version}"
            )

            directory = ROOT / folder / "openml_eval"

            summary_path = (
                directory / "real_eval_metrics_summary.csv"
            )

            intervention_path = (
                directory /
                "real_eval_30repeats_topk_intervention.csv"
            )

            if not summary_path.exists():
                raise FileNotFoundError(summary_path)

            if not intervention_path.exists():
                raise FileNotFoundError(intervention_path)

            # Existing three metrics
            df = pd.read_csv(summary_path)
            df["dataset"] = df["dataset"].astype(str)

            if df["dataset"].duplicated().any():
                raise ValueError(
                    f"Duplicated datasets: {summary_path}"
                )

            df = df.set_index("dataset")

            # New fourth metric
            intervention = pd.read_csv(intervention_path)

            keep_stats = compute_keep_advantage(
                intervention
            )

            # Join by dataset name
            df = df.join(keep_stats, how="left")

            # Warn about incomplete repeat pairs
            incomplete = df[
                df["keep_n_pairs"].notna()
                & (df["keep_n_pairs"] != 30)
            ]

            if not incomplete.empty:
                print(
                    f"[WARN] {method} {version}: "
                    f"{len(incomplete)} datasets have "
                    "fewer than 30 paired repeats"
                )

            tables[(method, version)] = df

            print(
                f"[OK] {method:12s} {version}: "
                f"{len(df)} datasets"
            )

    return tables


def get_value(tables, dataset, method, version, column):
    if column is None:
        return np.nan

    df = tables[(method, version)]

    if dataset not in df.index or column not in df.columns:
        return np.nan

    val = df.at[dataset, column]

    return float(val) if pd.notna(val) else np.nan


def all_values(tables, column):
    values = []

    for df in tables.values():
        if column in df.columns:
            v = pd.to_numeric(df[column], errors="coerce")
            values.extend(v.dropna().tolist())

    return np.asarray(values, dtype=float)


# ============================================================
# METRIC-SPECIFIC SCALES
# ============================================================

def nice_ceiling(x):
    """Round positive bounds to readable tick ranges."""
    x = float(x)

    if not np.isfinite(x) or x <= 0:
        return 0.01

    power = 10 ** np.floor(np.log10(x))
    fraction = x / power

    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if fraction <= step + 1e-12:
            return float(step * power)

    return float(10 * power)



def make_scales(tables):
    scales = {}

    for name, mean_col, std_col in METRICS:

        if SCALE_OVERRIDES[name] is not None:
            scales[name] = SCALE_OVERRIDES[name]
            continue

        values = all_values(tables, mean_col)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            raise ValueError(
                f"No valid values for metric: {name}"
            )

        minimum = float(np.min(values))
        maximum = float(np.max(values))

        # Keep Advantage uses a signed scale
        if name == "Keep Adv.":

            lower = (
                -nice_ceiling(abs(minimum) * 1.05)
                if minimum < 0
                else 0.0
            )

            upper = (
                nice_ceiling(maximum * 1.05)
                if maximum > 0
                else 0.01
            )

            scales[name] = (lower, upper)
            continue

        # Other metrics
        if minimum < 0:
            lower = -nice_ceiling(
                abs(minimum) * 1.05
            )
        else:
            lower = 0.0

        upper = nice_ceiling(
            max(maximum * 1.05, 0.01)
        )

        upper = min(upper, 1.0)
        lower = max(lower, -1.0)

        scales[name] = (lower, upper)

    return scales



def fmt_number(x):
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.3g}".replace("-", "−")


def to_radius(value, limits):
    low, high = limits
    fraction = (value - low) / (high - low)
    return R_IN + fraction * (R_OUT - R_IN)


# ============================================================
# DRAWING HELPERS
# ============================================================

def pastel(color, factor=0.57):
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1 - rgb) * factor)


def arc(ax, theta0, theta1, radius, **kwargs):
    theta = np.linspace(theta0, theta1, 100)
    ax.plot(
        theta,
        np.full(theta.shape, radius),
        **kwargs
    )


def draw_sector(ax, index, name, limits):

    start = index * np.pi / 2 + SECTOR_GAP
    end = (index + 1) * np.pi / 2 - SECTOR_GAP
    midpoint = (start + end) / 2

    # Background: each sector has its own field
    ax.bar(
        midpoint,
        R_OUT - R_IN,
        width=end - start,
        bottom=R_IN,
        color="#FBFCFB",
        edgecolor="none",
        zorder=0,
    )

    # Radial grid and numerical tick labels
    tick_values = np.linspace(
        limits[0], limits[1], 5
    )

    tick_theta = start + np.deg2rad(7)

    for tick in tick_values:
        r = to_radius(tick, limits)

        arc(
            ax, start, end, r,
            color="#BFC7C8",
            linewidth=0.48,
            linestyle=(0, (2, 2)),
            zorder=1,
        )

        ax.text(
            tick_theta,
            r,
            fmt_number(tick),
            fontsize=5.9,
            color="#667078",
            ha="center",
            va="center",
            bbox={
                "facecolor": "#FBFCFB",
                "edgecolor": "none",
                "pad": 0.3,
            },
            zorder=9,
        )

    # Zero ring for signed metrics
    low, high = limits

    if low < 0 < high:
        zero_radius = to_radius(0, limits)

        arc(
            ax, start, end, zero_radius,
            color="#4F5960",
            linewidth=0.95,
            linestyle=(0, (3, 2)),
            zorder=2,
        )

    # Sector boundaries
    for theta in (start, end):
        ax.plot(
            [theta, theta],
            [R_IN, R_OUT + 0.035],
            color="#626C70",
            linewidth=0.75,
            zorder=3,
        )

    # Outer boundary
    arc(
        ax, start, end, R_OUT + 0.035,
        color="#5A6569",
        linewidth=0.9,
        zorder=3,
    )

    # Separate outer metric-title band
    ax.bar(
        midpoint,
        0.061,
        width=end - start,
        bottom=0.967,
        color="#F6F8F7",
        edgecolor="#586368",
        linewidth=0.9,
        zorder=3,
    )

    # Tangential label, like the supplied reference figure
    angle_deg = np.degrees(midpoint)

    # Normalize rotation to [-180, 180]
    rotation = (-angle_deg + 180) % 360 - 180

    # Keep all labels upright and readable
    if rotation > 90:
        rotation -= 180
    elif rotation < -90:
        rotation += 180

    ax.text(
        midpoint,
        0.998,
        name,
        fontsize=8.1,
        fontweight="semibold",
        ha="center",
        va="center",
        rotation=rotation,
        rotation_mode="anchor",
        color="#273136",
        zorder=10,
    )

    return start, end


def draw_error_bar(ax, theta, mean, std, limits, width):
    if not np.isfinite(std) or std < 0:
        return

    low, high = limits

    lower = mean - std
    upper = mean + std

    clipped_lower = lower < low
    clipped_upper = upper > high

    r1 = to_radius(np.clip(lower, low, high), limits)
    r2 = to_radius(np.clip(upper, low, high), limits)

    ax.plot(
        [theta, theta],
        [r1, r2],
        color="#30393D",
        linewidth=0.70,
        zorder=8,
    )

    cap_width = width * 0.25

    for r in (r1, r2):
        ax.plot(
            [theta - cap_width, theta + cap_width],
            [r, r],
            color="#30393D",
            linewidth=0.70,
            zorder=8,
        )

    # Flag intervals extending beyond the display range.
    if clipped_upper:
        ax.scatter(
            [theta], [R_OUT + 0.016],
            s=9, marker="^",
            color="#30393D",
            zorder=11,
        )

    if clipped_lower:
        ax.scatter(
            [theta], [R_IN - 0.016],
            s=9, marker="v",
            color="#30393D",
            zorder=11,
        )


def draw_one_bar(
    ax, theta, width, mean, std,
    limits, color, show_value
):
    if not np.isfinite(mean):
        # Missing V1/V2: preserve slot, draw nothing
        return

    low, high = limits

    shown_mean = np.clip(mean, low, high)

    r_mean = to_radius(shown_mean, limits)
    r_zero = to_radius(
        np.clip(0.0, low, high), limits
    )

    bottom = min(r_mean, r_zero)
    height = abs(r_mean - r_zero)

    if height > 1e-10:
        ax.bar(
            theta,
            height,
            width=width,
            bottom=bottom,
            facecolor=color,
            edgecolor="#465056",
            linewidth=0.55,
            zorder=5,
        )

    draw_error_bar(
        ax, theta, mean, std, limits, width
    )

    # Mean outside display limits: visible clipping marker
    if mean > high:
        ax.scatter(
            [theta], [R_OUT + 0.024],
            s=17, marker="^",
            color="#20282C",
            zorder=12,
        )

    elif mean < low:
        ax.scatter(
            [theta], [R_IN - 0.024],
            s=17, marker="v",
            color="#20282C",
            zorder=12,
        )

    if show_value:
        angle = np.degrees(theta)
        rotation = -angle

        if rotation < -90:
            rotation += 180

        label_r = (
            min(R_OUT + 0.025, r_mean + 0.021)
            if mean >= 0
            else max(R_IN - 0.018, r_mean - 0.021)
        )

        ax.text(
            theta,
            label_r,
            fmt_number(mean),
            fontsize=5.2,
            color="#394348",
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
            zorder=12,
        )


# ============================================================
# DATASET CIRCLE
# ============================================================

def draw_dataset(ax, dataset, tables, scales, show_values):

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.14)
    ax.set_axis_off()
    ax.set_facecolor("white")

    for sector_idx, (name, mean_col, std_col) in enumerate(METRICS):

        limits = scales[name]

        start, end = draw_sector(
            ax, sector_idx, name, limits
        )

        # Reserve a small part of the sector for radial ticks
        bars_start = start + np.deg2rad(15)
        bars_end = end - np.deg2rad(2)

        step = (bars_end - bars_start) / 8
        bar_width = step * 0.76

        for method_idx, method in enumerate(METHODS):
            for version_idx, version in enumerate(("v1", "v2")):

                slot = method_idx * 2 + version_idx

                theta = (
                    bars_start
                    + (slot + 0.5) * step
                )

                mean = get_value(
                    tables, dataset, method,
                    version, mean_col
                )

                std = get_value(
                    tables, dataset, method,
                    version, std_col
                )

                base = COLORS[method]

                color = (
                    pastel(base)
                    if version == "v1"
                    else base
                )

                draw_one_bar(
                    ax,
                    theta,
                    bar_width,
                    mean,
                    std,
                    limits,
                    color,
                    show_values,
                )

    # Inner circle outline
    theta_circle = np.linspace(0, 2 * np.pi, 360)

    ax.plot(
        theta_circle,
        np.full_like(theta_circle, R_IN),
        color="#526066",
        linewidth=1.05,
        zorder=15,
    )

    # Inner double ring, as in the template
    ax.plot(
        theta_circle,
        np.full_like(theta_circle, R_IN - 0.023),
        color="#AAB3B6",
        linewidth=0.5,
        zorder=15,
    )

    label = textwrap.wrap(
        str(dataset),
        width=13,
        break_long_words=True,
        break_on_hyphens=True,
    )

    if len(label) > 3:
        label = label[:3]
        label[-1] += "…"

    ax.text(
        0.5,
        0.5,
        "\n".join(label),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="semibold",
        color="#273136",
        linespacing=1.1,
        zorder=20,
    )


# ============================================================
# FIGURE ASSEMBLY
# ============================================================

def create_figure(
    datasets, tables, scales,
    output_name, ncols, cell_size, show_values
):

    nrows = ceil(len(datasets) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            ncols * cell_size,
            nrows * cell_size + 1.0,
        ),
        subplot_kw={"projection": "polar"},
        squeeze=False,
    )

    axes = axes.ravel()

    for i, dataset in enumerate(datasets):
        draw_dataset(
            axes[i], dataset, tables,
            scales, show_values
        )

    for i in range(len(datasets), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "SCM Classification Benchmark",
        fontsize=18,
        fontweight="semibold",
        color="#263238",
        y=0.991,
    )

    legend = [
        Patch(
            facecolor=COLORS[m],
            edgecolor="#465056",
            linewidth=0.5,
            label=m,
        )
        for m in METHODS
    ]

    legend += [
        Patch(
            facecolor="#D4DFDF",
            edgecolor="#465056",
            linewidth=0.5,
            label="V1 · light",
        ),
        Patch(
            facecolor="#607982",
            edgecolor="#465056",
            linewidth=0.5,
            label="V2 · dark",
        ),
    ]

    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=6,
        frameon=False,
        fontsize=9,
        handlelength=1.5,
        columnspacing=1.8,
    )

    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.932,
        bottom=0.045,
        wspace=0.04,
        hspace=0.10,
    )

    fig.text(
        0.5,
        0.027,
        "Start = 0.3  |  Each metric has its own scale, "
        "shared across datasets  |  Triangles mark clipped values / intervals",
        ha="center",
        fontsize=8.5,
        color="#657177",
    )

    fig.savefig(
        OUT / f"{output_name}.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig.savefig(
        OUT / f"{output_name}.png",
        dpi=210,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    tables = load_tables()


    check_rows = []

    for (method, version), df in tables.items():
        for dataset, row in df.iterrows():

            check_rows.append({
                "dataset": dataset,
                "method": method,
                "version": version,
                "keep_advantage_mean": row["keep_advantage_mean"],
                "keep_advantage_std": row["keep_advantage_std"],
                "n_pairs": row["keep_n_pairs"],
            })

    pd.DataFrame(check_rows).to_csv(
        OUT / "keep_advantage_check.csv",
        index=False,
        float_format="%.6f",
    )


    # Dataset union: keep V2-only datasets
    datasets = sorted(
        set().union(*(set(df.index) for df in tables.values())),
        key=str.lower,
    )

    scales = make_scales(tables)

    print(f"\nTotal datasets: {len(datasets)}")
    print("\nMetric-specific scales:")

    for name, bounds in scales.items():
        print(f"  {name:12s}: {bounds}")

    # 1. One figure containing all datasets
    create_figure(
        datasets=datasets,
        tables=tables,
        scales=scales,
        output_name="scm_cls_all_datasets",
        ncols=NCOLS,
        cell_size=CELL_SIZE,
        show_values=SHOW_VALUES_IN_OVERVIEW,
    )

    # 2. Enlarged example to inspect fine details
    create_figure(
        datasets=[datasets[0]],
        tables=tables,
        scales=scales,
        output_name="scm_cls_detail_example",
        ncols=1,
        cell_size=6.5,
        show_values=SHOW_VALUES_IN_DETAIL,
    )

    print(f"\nSaved to: {OUT.resolve()}")


if __name__ == "__main__":
    main()

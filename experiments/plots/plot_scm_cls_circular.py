from pathlib import Path
from math import ceil, floor, log10
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb


ROOT = Path("results/training")
OUT = Path("results/figures")
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

METRICS = [
    ("Prediction", "prediction_mean", "prediction_std"),
    ("Agreement", "best_reference_agreement_mean", "best_reference_agreement_std"),
    ("Stability", "importance_ranking_stability", None),
]

TASKS = {"cls": "classification", "reg": "regression"}
NCOLS = {"cls": 5, "reg": 3}
CELL_SIZE = 4.1
SHOW_VALUES_IN_SINGLE = True
SHOW_VALUES_IN_OVERVIEW = False
R_IN = 0.30
R_OUT = 0.87
SECTOR_GAP = np.deg2rad(4)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})


def load_tables(task):
    tables = {}
    for method, slug in METHODS.items():
        for version in ("v1", "v2"):
            folder = f"scm_{task}_{slug}_w_imp_start{START}_epoch{EPOCH}_{version}"
            path = ROOT / folder / "openml_eval" / "real_eval_metrics_summary.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing CSV: {path}")
            df = pd.read_csv(path)
            df["dataset"] = df["dataset"].astype(str)
            if df["dataset"].duplicated().any():
                raise ValueError(f"Duplicate datasets in {path}")
            required = {"dataset"} | {c for _, mean, std in METRICS for c in (mean, std) if c}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns {missing} in {path}")
            tables[(method, version)] = df.set_index("dataset")
            print(f"[OK] {task} | {method:12s} | {version} | {len(df)} datasets")

    return tables


def get_value(tables, dataset, method, version, column):
    if column is None:
        return np.nan
    df = tables[(method, version)]
    if dataset not in df.index or column not in df.columns:
        return np.nan
    value = pd.to_numeric(df.at[dataset, column], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def all_values(groups, column):
    arrays = []
    for tables in groups:
        for df in tables.values():
            values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values):
                arrays.append(values)
    return np.concatenate(arrays) if arrays else np.array([])


def nice_limits(values, correlation=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No valid values for scale calculation")

    lo = min(0.0, float(values.min()))
    hi = max(0.0, float(values.max()))
    span = hi - lo
    if span < 1e-10:
        span = 0.1
        hi = lo + span

    lo = lo - 0.03 * span if lo < 0 else 0.0
    hi = hi + 0.03 * span if hi > 0 else 0.0
    raw_step = (hi - lo) / 4
    power = 10 ** floor(log10(raw_step))
    step = next(s * power for s in (1, 2, 2.5, 5, 10) if s * power >= raw_step - 1e-12)
    lower = floor(lo / step) * step
    upper = ceil(hi / step) * step
    if correlation:
        lower = max(lower, -1.0)
        upper = min(upper, 1.0)
    ticks = np.linspace(lower, upper, 5)
    return (float(lower), float(upper)), ticks


def make_scales(all_tables):
    scales = {}
    for task_key, tables in all_tables.items():
        scales[task_key] = {}
        for name, mean_col, _ in METRICS:
            if name == "Prediction" and task_key == "cls":
                bounds, ticks = (0.0, 1.0), np.linspace(0, 1, 5)
            else:
                sources = list(all_tables.values()) if name != "Prediction" else [tables]
                bounds, ticks = nice_limits(all_values(sources, mean_col), correlation=name != "Prediction")
            scales[task_key][name] = (bounds, ticks)

    return scales


def to_radius(value, limits):
    low, high = limits
    return R_IN + (value - low) / (high - low) * (R_OUT - R_IN)


def fmt(value):
    return "0" if abs(value) < 1e-10 else f"{value:.3g}".replace("-", "−")


def pastel(color, factor=0.58):
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1 - rgb) * factor)


def arc(ax, start, end, radius, **kwargs):
    theta = np.linspace(start, end, 100)
    ax.plot(theta, np.full_like(theta, radius), **kwargs)


def upright_rotation(theta):
    return ((-np.degrees(theta) + 90) % 180) - 90




def draw_curved_title(ax, mid, text, radius=1.005, fontsize=8.0):
    """Draw readable curved titles inside the outer ring."""
    angle_deg = np.degrees(mid) % 360
    direction = -1 if 90 < angle_deg < 270 else 1

    spacing = np.deg2rad(3.1)
    offsets = (np.arange(len(text)) - (len(text) - 1) / 2) * spacing

    for char, offset in zip(text, offsets):
        theta = mid + direction * offset
        ax.text(theta, radius, char, ha="center", va="center",
                fontsize=fontsize, fontweight="medium",
                rotation=upright_rotation(theta), rotation_mode="anchor",
                color="#263238", zorder=12)



def draw_sector(ax, index, name, limits, ticks):
    sector_size = 2 * np.pi / 3
    start = index * sector_size + SECTOR_GAP
    end = (index + 1) * sector_size - SECTOR_GAP
    mid = (start + end) / 2
    ax.bar(mid, R_OUT - R_IN, width=end - start, bottom=R_IN,
           color="#FAFBFA", edgecolor="#D2D7D7", linewidth=0.5, zorder=0)

    for tick in ticks:
        r = to_radius(tick, limits)
        arc(ax, start, end, r, color="#B9C1C3", linewidth=0.5, linestyle=(0, (2, 2)), zorder=1)
        ax.text(start + np.deg2rad(9), r, fmt(tick), fontsize=6.5,
                ha="center", va="center", color="#647078", zorder=9,
                bbox=dict(facecolor="#FAFBFA", edgecolor="none", pad=0.35))

    if limits[0] < 0 < limits[1]:
        zero_r = to_radius(0, limits)
        arc(ax, start, end, zero_r, color="#4E5B60", linewidth=1.0,
            linestyle=(0, (3, 2)), zorder=2)

    for angle in (start, end):
        ax.plot([angle, angle], [R_IN, R_OUT + 0.035],
                color="#6C777B", linewidth=0.7, zorder=3)

    arc(ax, start, end, R_OUT + 0.035, color="#59656B", linewidth=0.9, zorder=3)

    # Outer curved title band
    ax.bar(mid, 0.105, width=end - start, bottom=0.950,
        color="#FAFBFB", edgecolor="#59656B", linewidth=0.85, zorder=3)

    # Metric name follows the outer arc
    draw_curved_title(ax, mid, name, radius=1.005, fontsize=8.5)


    return start, end


def draw_errorbar(ax, theta, mean, std, limits, width):
    if not np.isfinite(std) or std < 0:
        return
    low, high = limits
    a, b = mean - std, mean + std
    r1 = to_radius(np.clip(a, low, high), limits)
    r2 = to_radius(np.clip(b, low, high), limits)
    ax.plot([theta, theta], [r1, r2], color="#283338", linewidth=0.7, zorder=8)
    cap = width * 0.25
    for r in (r1, r2):
        ax.plot([theta - cap, theta + cap], [r, r], color="#283338", linewidth=0.7, zorder=8)
    if b > high:
        ax.scatter([theta], [R_OUT + 0.014], marker="^", s=10, color="#283338", zorder=11)
    if a < low:
        ax.scatter([theta], [R_IN - 0.014], marker="v", s=10, color="#283338", zorder=11)


def draw_bar(ax, theta, width, mean, std, limits, color, show_value):
    if not np.isfinite(mean):
        return
    low, high = limits
    r_mean = to_radius(np.clip(mean, low, high), limits)
    r_zero = to_radius(np.clip(0, low, high), limits)
    bottom = min(r_mean, r_zero)
    height = abs(r_mean - r_zero)
    if height > 1e-10:
        ax.bar(theta, height, width=width, bottom=bottom,
               color=color, edgecolor="#465056", linewidth=0.5, zorder=5)
    else:
        ax.scatter([theta], [r_zero], s=5, color=color, zorder=6)
    draw_errorbar(ax, theta, mean, std, limits, width)
    if show_value:
        label_r = min(R_OUT + 0.04, r_mean + 0.027) if mean >= 0 else max(R_IN - 0.025, r_mean - 0.027)
        ax.text(theta, label_r, fmt(mean), fontsize=5.6, ha="center", va="center",
                rotation=upright_rotation(theta), rotation_mode="anchor",
                color="#394348", zorder=12)


def draw_dataset(ax, dataset, tables, scales, show_values=False):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.15)
    ax.set_axis_off()
    ax.set_facecolor("white")

    for sector_idx, (name, mean_col, std_col) in enumerate(METRICS):
        limits, ticks = scales[name]
        start, end = draw_sector(ax, sector_idx, name, limits, ticks)
        bars_start = start + np.deg2rad(18)
        bars_end = end - np.deg2rad(4)
        step = (bars_end - bars_start) / 8
        bar_width = step * 0.76
        for method_idx, method in enumerate(METHODS):
            for version_idx, version in enumerate(("v1", "v2")):
                theta = bars_start + (method_idx * 2 + version_idx + 0.5) * step
                mean = get_value(tables, dataset, method, version, mean_col)
                std = get_value(tables, dataset, method, version, std_col)
                color = pastel(COLORS[method]) if version == "v1" else COLORS[method]
                draw_bar(ax, theta, bar_width, mean, std, limits, color, show_values)
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(theta, np.full_like(theta, R_IN), color="#526066", linewidth=1.0, zorder=15)
    ax.plot(theta, np.full_like(theta, R_IN - 0.022), color="#AAB3B6", linewidth=0.5, zorder=15)
    label = textwrap.wrap(str(dataset), width=12, break_long_words=True)
    if len(label) > 3:
        label = label[:3]
        label[-1] += "…"
    ax.text(0.5, 0.5, "\n".join(label), transform=ax.transAxes, ha="center", va="center",
            fontsize=8.2, fontweight="semibold", color="#283439", linespacing=1.12, zorder=20)


def legend_handles():
    handles = [Patch(facecolor=color, edgecolor="#465056", linewidth=0.5, label=method)
               for method, color in COLORS.items()]
    handles += [
        Patch(facecolor="#CFDDDF", edgecolor="#465056", linewidth=0.5, label="V1 · light"),
        Patch(facecolor="#597988", edgecolor="#465056", linewidth=0.5, label="V2 · dark"),
    ]
    return handles


def safe_filename(name):
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("._")
    return value[:85] or "dataset"


def save_individual(task_key, datasets, tables, scales):
    output = OUT / task_key / "individual"
    output.mkdir(parents=True, exist_ok=True)
    for i, dataset in enumerate(datasets, 1):
        fig = plt.figure(figsize=(7, 7.3))
        ax = fig.add_axes([0.06, 0.14, 0.88, 0.76], projection="polar")
        draw_dataset(ax, dataset, tables, scales, show_values=SHOW_VALUES_IN_SINGLE)
        fig.text(0.5, 0.965, dataset, ha="center", va="center", fontsize=13, fontweight="semibold")
        fig.text(0.5, 0.932, f"SCM {task_key.upper()} | Start = {START}", ha="center",
                 fontsize=9, color="#647078")
        fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.055),
                   ncol=6, frameon=False, fontsize=8.5)
        filename = f"{i:02d}_{safe_filename(dataset)}.pdf"
        fig.savefig(output / filename, bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)
    print(f"[SAVED] {len(datasets)} individual PDFs -> {output}")


def save_overview(task_key, datasets, tables, scales):
    ncols = NCOLS[task_key]
    nrows = ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * CELL_SIZE, nrows * CELL_SIZE + 1.2),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    for ax, dataset in zip(axes.flat, datasets):
        draw_dataset(ax, dataset, tables, scales, show_values=SHOW_VALUES_IN_OVERVIEW)
    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)

    title = "Classification" if task_key == "cls" else "Regression"
    fig.suptitle(f"SCM {title} — OpenML Benchmark", fontsize=19, fontweight="semibold", y=0.994)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=6, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.94, bottom=0.045, wspace=0.05, hspace=0.10)
    fig.text(0.5, 0.025, f"Start = {START} | Mean ± Std | Metric scales shared across datasets",
             ha="center", fontsize=9, color="#647078")
    path = OUT / task_key / "all_datasets.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"[SAVED] Overview -> {path}")


def main():
    all_tables = {task_key: load_tables(task) for task_key, task in TASKS.items()}
    all_scales = make_scales(all_tables)
    for task_key, tables in all_tables.items():
        datasets = sorted(set().union(*(set(df.index) for df in tables.values())), key=str.lower)
        scales = all_scales[task_key]
        print(f"\n[{task_key.upper()}] Total datasets: {len(datasets)}")
        for name, (limits, _) in scales.items():
            print(f"  {name:12s}: {limits}")
        save_individual(task_key, datasets, tables, scales)
        save_overview(task_key, datasets, tables, scales)

    print(f"\nCompleted. All PDFs saved under: {OUT.resolve()}")


if __name__ == "__main__":
    main()

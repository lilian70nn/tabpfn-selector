from pathlib import Path
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def parse_trace(path):
    gt = {}
    scores = {}

    with open(path, "r") as f:
        for line in f:
            if line.startswith("[gt_imp]"):
                m = re.search(r"table=(\d+)\s+d=(\d+)\s+values=(.*)", line.strip())
                if m:
                    table = int(m.group(1))
                    gt[table] = np.array([float(x) for x in m.group(3).split(",")])

            elif line.startswith("[imp_score]"):
                m = re.search(r"step=(\d+)\s+table=(\d+)\s+layer=(\d+)\s+d=(\d+)\s+values=(.*)", line.strip())
                if m:
                    step = int(m.group(1))
                    table = int(m.group(2))
                    layer = int(m.group(3))
                    scores[(step, table, layer)] = np.array([float(x) for x in m.group(5).split(",")])

    return gt, scores


def safe_spearman(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b).statistic


def build_heatmap(gt, scores):
    steps = sorted({step for step, _, _ in scores})
    layers = sorted({layer for _, _, layer in scores})

    heatmap = np.full((len(layers), len(steps)), np.nan)

    for i, layer in enumerate(layers):
        for j, step in enumerate(steps):
            correlations = []

            for table, gt_imp in gt.items():
                key = (step, table, layer)
                if key not in scores:
                    continue

                rho = safe_spearman(gt_imp, scores[key])
                if np.isfinite(rho):
                    correlations.append(rho)

            if correlations:
                heatmap[i, j] = np.mean(correlations)

    return steps, layers, heatmap


def plot_heatmap(steps, layers, heatmap, save_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Selector Representation Depth")

    tick_idx = np.linspace(0, len(steps) - 1, min(10, len(steps)), dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([steps[i] for i in tick_idx])

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Spearman Correlation with GT Importance")

    ax.set_title("Feature Importance Alignment Across Training and Model Depth")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(trace_path, save_path=None):
    gt, scores = parse_trace(trace_path)
    steps, layers, heatmap = build_heatmap(gt, scores)

    if save_path is None:
        save_path = trace_path.parent / "importance_spearman_heatmap.png"
    plot_heatmap(steps, layers, heatmap, save_path)

    print(f"tables: {len(gt)}")
    print(f"steps: {len(steps)}")
    print(f"layers: {len(layers)}")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("trace_path", type=Path)
    # args = parser.parse_args()

    trace_path = Path("results/training/scm_cls_w_imp_training_results/importance_trace.txt")
    save_path = Path("results/figures/importance_spearman_heatmap.png")

    main(trace_path, save_path)
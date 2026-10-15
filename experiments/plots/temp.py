from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


GRADIENT_TRACE = "results/training/scm_classification_gradient_w_imp_start0/importance_trace.txt"
MI_TRACE = "results/training/scm_classification_mi_w_imp_start0/importance_trace.txt"
OUT_PATH = "results/figures/importance_learning_examples_table_4.png"

TABLE = 4
NUM_STEPS = 8


def parse_trace(path):
    gt, scores = {}, {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("[gt_imp]"):
                m = re.search(r"table=(\d+)\s+d=(\d+)\s+values=(.*)", line)
                if m:
                    table = int(m.group(1))
                    d = int(m.group(2))
                    gt[table] = np.array([float(x) for x in m.group(3).split(",")][:d])

            elif line.startswith("[imp_score]"):
                m = re.search(r"step=(\d+)\s+table=(\d+)\s+layer=(\d+)\s+d=(\d+)\s+values=(.*)", line)
                if m:
                    step = int(m.group(1))
                    table = int(m.group(2))
                    layer = int(m.group(3))
                    d = int(m.group(4))
                    scores[(step, table, layer)] = np.array([float(x) for x in m.group(5).split(",")][:d])

    return gt, scores


def plot_trajectory(ax, gt, scores, table, title):
    final_layer = max(layer for _, t, layer in scores if t == table)
    data = sorted((step, values) for (step, t, layer), values in scores.items() if t == table and layer == final_layer)

    steps = [step for step, _ in data]
    values = [score for _, score in data]
    feature_order = sorted(range(len(gt[table])), key=lambda j: gt[table][j], reverse=True)

    for j in feature_order:
        ax.plot(steps, [score[j] for score in values], marker="o", markersize=2.5, linewidth=1.2, label=f"F{j+1} (GT={gt[table][j]:.3f})")

    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Predicted Importance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)


def plot_profile(ax, gt, scores, table, num_steps, title):
    available_steps = sorted({step for step, t, _ in scores if t == table})
    final_layer = max(layer for _, t, layer in scores if t == table)

    idx = np.linspace(0, len(available_steps) - 1, min(num_steps, len(available_steps)), dtype=int)
    selected_steps = [available_steps[i] for i in np.unique(idx)]

    gt_imp = gt[table].copy()
    gt_imp = gt_imp / gt_imp.sum()
    features = np.arange(1, len(gt_imp) + 1)
    cmap = plt.get_cmap("viridis")

    for i, step in enumerate(selected_steps):
        pred = scores[(step, table, final_layer)].copy()
        pred = pred / pred.sum() if pred.sum() > 0 else pred
        color = cmap(i / max(len(selected_steps) - 1, 1))
        ax.plot(features, pred, marker="o", markersize=3, linewidth=1.2, color=color, alpha=0.85, label=f"Step {step}")

    ax.plot(features, gt_imp, marker="o", markersize=5, linewidth=2.2, color="black", linestyle="--", label="Ground Truth")

    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalized Feature Importance")
    ax.set_xticks(features)
    ax.set_xticklabels([f"F{i}" for i in features])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)


def main():
    gradient_gt, gradient_scores = parse_trace(GRADIENT_TRACE)
    mi_gt, mi_scores = parse_trace(MI_TRACE)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    plot_trajectory(axes[0, 0], gradient_gt, gradient_scores, TABLE, "(a) Gradient — Importance over Training")
    plot_trajectory(axes[0, 1], mi_gt, mi_scores, TABLE, "(b) Mutual Information — Importance over Training")

    plot_profile(axes[1, 0], gradient_gt, gradient_scores, TABLE, NUM_STEPS, "(c) Gradient — Importance Profile Evolution")
    plot_profile(axes[1, 1], mi_gt, mi_scores, TABLE, NUM_STEPS, "(d) Mutual Information — Importance Profile Evolution")

    fig.suptitle(f"Examples of Feature-Importance Learning Dynamics — Table {TABLE}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_trace(path):
    gt, scores = {}, {}

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


def plot_table(gt, scores, table, num_steps, save_path):
    available_steps = sorted({step for step, t, _ in scores if t == table})
    final_layer = max(layer for _, t, layer in scores if t == table)

    idx = np.linspace(0, len(available_steps) - 1, min(num_steps, len(available_steps)), dtype=int)
    selected_steps = [available_steps[i] for i in np.unique(idx)]

    gt_imp = gt[table].copy()
    gt_imp = gt_imp / gt_imp.sum()

    features = np.arange(1, len(gt_imp) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")

    for i, step in enumerate(selected_steps):
        pred = scores[(step, table, final_layer)].copy()
        pred = pred / pred.sum() if pred.sum() > 0 else pred

        color = cmap(i / max(len(selected_steps) - 1, 1))
        ax.plot(features, pred, marker="o", markersize=4, linewidth=1.5, color=color, alpha=0.85, label=f"Step {step}")

    ax.plot(features, gt_imp, marker="o", markersize=6, linewidth=2.5, color="black", linestyle="--", label="Ground Truth")

    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalized Feature Importance")
    ax.set_title(f"Feature Importance Evolution — Table {table}")
    ax.set_xticks(features)
    ax.set_xticklabels([f"F{i}" for i in features])
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(trace_path, save_path, num_tables=2, num_steps=6, seed=0):
    trace_path = Path(trace_path)
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    gt, scores = parse_trace(trace_path)
    available_tables = sorted(gt.keys())

    if len(available_tables) == 0:
        raise ValueError("No tables found in importance trace.")

    # rng = np.random.default_rng(seed)
    # selected_tables = rng.choice(available_tables, size=min(num_tables, len(available_tables)), replace=False)

    # print(f"selected tables: {selected_tables.tolist()}")

    selected_tables = [4, 5]
    for table in selected_tables:
        table = int(table)
        output_path = save_dir / f"importance_profile_table_{table}.png"
        plot_table(gt, scores, table, num_steps, output_path)
        print(f"saved: {output_path}")


if __name__ == "__main__":

    main(
        trace_path="results/training/scm_cls_w_imp_training_results/importance_trace.txt",
        save_path="results/figures",
        num_tables=2,
        num_steps=8,
        seed=0,
    )
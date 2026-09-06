from pathlib import Path
import re
import random
import matplotlib.pyplot as plt


def plot_importance_trajectory(input_path, save_path):
    input_path, save_path = Path(input_path), Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    gt = {}
    records = {}

    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[gt_imp]"):
                m = re.match(r"\[gt_imp\] table=(\d+) d=(\d+) values=(.*)", line)
                if m:
                    table, d = int(m.group(1)), int(m.group(2))
                    gt[table] = [float(x) for x in m.group(3).split(",")][:d]
            elif line.startswith("[imp_score]"):
                m = re.match(r"\[imp_score\] step=(\d+) table=(\d+) layer=(\d+) d=(\d+) values=(.*)", line)
                if m:
                    step, table, layer, d = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                    values = [float(x) for x in m.group(5).split(",")][:d]
                    records.setdefault(table, []).append((step, layer, values))

    available_tables = sorted(set(gt) & set(records))
    if len(available_tables) < 2:
        raise ValueError("Need at least two tables in importance trace.")

    rng = random.Random(0)
    selected_tables = rng.sample(available_tables, 2)

    for table in selected_tables:
        final_layer = max(layer for _, layer, _ in records[table])
        data = sorted((step, values) for step, layer, values in records[table] if layer == final_layer)
        steps = [step for step, _ in data]
        scores = [values for _, values in data]

        plt.figure(figsize=(9, 5))
        feature_order = sorted(range(len(gt[table])), key=lambda j: gt[table][j], reverse=True)

        for j in feature_order:
            plt.plot(steps, [score[j] for score in scores], marker="o", markersize=3, linewidth=1.5, label=f"F{j+1} (GT={gt[table][j]:.3f})")

        plt.xlabel("Training Step")
        plt.ylabel("Predicted Importance")
        plt.title(f"Feature Importance over Training — Table {table}")
        plt.grid(alpha=0.25)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path / f"importance_trajectory_table_{table}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Selected tables: {selected_tables}")
    print(f"Saved figures to: {save_path}")


if __name__ == "__main__":

    input_path = "results/training/scm_cls_w_imp_training_results/importance_trace.txt"
    save_path = "results/figures"

    plot_importance_trajectory(input_path, save_path)
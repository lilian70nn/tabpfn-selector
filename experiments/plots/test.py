import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

TRACE_PATH = Path("results/training/scm_cls_w_imp_training_results/importance_trace.txt")
TOP_K = 3

gt = {}
scores = {}

with open(TRACE_PATH, "r") as f:
    for line in f:
        if line.startswith("[gt_imp]"):
            m = re.search(r"table=(\d+)\s+d=(\d+)\s+values=(.*)", line)
            if m:
                table, d = int(m.group(1)), int(m.group(2))
                values = np.array([float(x) for x in m.group(3).strip().split(",")])
                gt[table] = values[:d]

        elif line.startswith("[imp_score]"):
            m = re.search(r"step=(\d+)\s+table=(\d+)\s+layer=(\d+)\s+d=(\d+)\s+values=(.*)", line)
            if m:
                step, table, layer, d = map(int, m.group(1, 2, 3, 4))
                values = np.array([float(x) for x in m.group(5).strip().split(",")])[:d]
                scores.setdefault((step, table), {})[layer] = values

steps = sorted(set(step for step, _ in scores))
tables = sorted(gt)

mean_spearman, mean_topk, mean_mse = [], [], []

print(f"{'Step':>8} {'Spearman':>12} {'Top-3':>12} {'MSE':>12}")

for step in steps:
    spearmans, topks, mses = [], [], []

    for table in tables:
        if (step, table) not in scores:
            continue

        # Use final selector layer
        layers = scores[(step, table)]
        pred = layers[max(layers)]
        target = gt[table]

        d = min(len(pred), len(target))
        pred, target = pred[:d], target[:d]

        rho = spearmanr(target, pred).statistic
        if not np.isnan(rho):
            spearmans.append(rho)

        k = min(TOP_K, d)
        gt_top = set(np.argsort(target)[-k:])
        pred_top = set(np.argsort(pred)[-k:])
        topks.append(len(gt_top & pred_top) / k)

        mses.append(np.mean((pred - target) ** 2))

    mean_spearman.append(np.mean(spearmans))
    mean_topk.append(np.mean(topks))
    mean_mse.append(np.mean(mses))

    print(f"{step:8d} {mean_spearman[-1]:12.4f} {mean_topk[-1]:12.4f} {mean_mse[-1]:12.6f}")

plt.figure(figsize=(8, 5))
plt.plot(steps, mean_spearman, marker="o")
plt.xlabel("Training Step")
plt.ylabel("Mean Spearman Correlation")
plt.title("GT vs Predicted Importance — Spearman")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(steps, mean_topk, marker="o")
plt.xlabel("Training Step")
plt.ylabel(f"Mean Top-{TOP_K} Overlap")
plt.title(f"GT vs Predicted Importance — Top-{TOP_K} Overlap")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(steps, mean_mse, marker="o")
plt.xlabel("Training Step")
plt.ylabel("Mean MSE")
plt.title("GT vs Predicted Importance — MSE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
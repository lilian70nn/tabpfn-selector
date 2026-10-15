import os
import re
import matplotlib.pyplot as plt

RESULTS_DIR = "results/training"
OUT_DIR = "results/figures"

METHODS = {
    "gradient": "Gradient",
    "eigen_top1": "Eigen Top-1",
    "eigen_90": "Eigen-90",
    "mi": "MI",
    "marginal": "Marginal",
    "permutation": "Permutation",
}

def read_pred_loss(path):
    steps, losses = [], []
    pattern = re.compile(r"\[val\]\s+step\s+(\d+).*?\|\s+pred_loss\s+([\d.eE+-]+)")
    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses

def read_importance_start(path):
    pattern = re.compile(r"\[importance supervision ON\]\s+step=(\d+)")
    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return int(m.group(1))
    return None

def plot_task(task):
    baseline_path = os.path.join(RESULTS_DIR, f"scm_{task}_imp_weight0", "train_log.txt")
    base_steps, base_losses = read_pred_loss(baseline_path)

    data = {}
    pred_all = list(base_losses)
    xmax = max(base_steps)

    for start in ["start0", "start0.3"]:
        data[start] = {}
        for method, label in METHODS.items():
            path = os.path.join(RESULTS_DIR, f"scm_{task}_{method}_w_imp_{start}", "train_log.txt")
            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue
            steps, losses = read_pred_loss(path)
            data[start][method] = (steps, losses)
            pred_all.extend(losses)
            xmax = max(xmax, max(steps))

    pred_pad = 0.03 * (max(pred_all) - min(pred_all))
    pred_ylim = (min(pred_all) - pred_pad, max(pred_all) + pred_pad)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, start, panel_title in [
        (axes[0], "start0", "Start = 0"),
        (axes[1], "start0.3", "Start = 0.3"),
    ]:
        for method, label in METHODS.items():
            if method not in data[start]:
                continue
            steps, losses = data[start][method]
            ax.plot(steps, losses, marker="o", markersize=3.5, linewidth=1.8, label=label)

        ax.plot(base_steps, base_losses, marker="o", markersize=3.5, linewidth=2.0, linestyle="--", color="black", label="Importance Weight = 0")

        ax.set_title(panel_title)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Validation Prediction Loss")
        ax.set_xlim(0, xmax)
        ax.set_ylim(*pred_ylim)
        ax.grid(alpha=0.3)

        if start == "start0.3":
            path = os.path.join(RESULTS_DIR, f"scm_{task}_gradient_w_imp_{start}", "train_log.txt")
            start_step = read_importance_start(path)
            if start_step is not None:
                ax.axvline(start_step, color="gray", linestyle=":", linewidth=1.5)

    axes[0].legend(loc="upper right", fontsize=8)

    title = "Classification" if task == "classification" else "Regression"
    fig.suptitle(f"{title} — Prediction Loss", fontsize=14)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{task}_prediction_comparison.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

def read_best_metrics(path, task):
    best_step = None
    best_pattern = re.compile(r"\[best\]\s+step\s+(\d+)")
    val_pattern = re.compile(r"\[val\]\s+step\s+(\d+).*?\|\s+pred_loss\s+([\d.eE+-]+)")
    metric_name = "balanced_accuracy" if task == "classification" else "r2"
    metric_pattern = re.compile(rf"\|\s+{metric_name}\s+([\d.eE+-]+)")

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        m = best_pattern.search(line)
        if m:
            best_step = int(m.group(1))

    if best_step is None:
        raise ValueError(f"No [best] entry found: {path}")

    for line in lines:
        m = val_pattern.search(line)
        if m and int(m.group(1)) == best_step:
            metric_match = metric_pattern.search(line)
            if metric_match is None:
                raise ValueError(f"{metric_name} not found at best step {best_step}: {path}")
            return best_step, float(m.group(2)), float(metric_match.group(1))

    raise ValueError(f"No [val] entry found for best step {best_step}: {path}")


def collect_best_prediction_metrics(task):
    baseline_path = os.path.join(RESULTS_DIR, f"scm_{task}_imp_weight0", "train_log.txt")
    base_step, base_loss, base_metric = read_best_metrics(baseline_path, task)

    rows = [{
        "method": "Importance Weight = 0",
        "start": "-",
        "best_step": base_step,
        "pred_loss": base_loss,
        "metric": base_metric,
        "delta_pred_loss": 0.0,
        "delta_metric": 0.0,
    }]

    for method, label in METHODS.items():
        for start, start_label in [("start0", "0"), ("start0.3", "0.3")]:
            path = os.path.join(RESULTS_DIR, f"scm_{task}_{method}_w_imp_{start}", "train_log.txt")
            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue

            step, pred_loss, metric = read_best_metrics(path, task)
            rows.append({
                "method": label,
                "start": start_label,
                "best_step": step,
                "pred_loss": pred_loss,
                "metric": metric,
                "delta_pred_loss": pred_loss - base_loss,
                "delta_metric": metric - base_metric,
            })

    return rows


def print_best_prediction_metrics(task):
    rows = collect_best_prediction_metrics(task)
    metric_name = "BAcc" if task == "classification" else "R2"

    print(f"\n{task.upper()}")
    print(f"{'Method':<22} {'Start':>6} {'Step':>8} {'Pred Loss':>12} {metric_name:>10} {'Δ Loss':>10} {('Δ ' + metric_name):>10}")
    print("-" * 84)

    for r in rows:
        print(f"{r['method']:<22} {r['start']:>6} {r['best_step']:>8} {r['pred_loss']:>12.4f} {r['metric']:>10.4f} {r['delta_pred_loss']:>10.4f} {r['delta_metric']:>10.4f}")



if __name__ == "__main__":

    print_best_prediction_metrics("classification")
    print_best_prediction_metrics("regression")


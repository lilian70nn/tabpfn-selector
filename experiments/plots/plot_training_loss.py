import os
import re
import argparse
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

STARTS = [
    ("start0", "Start = 0"),
    ("start0.15", "Start = 0.15"),
    ("start0.3", "Start = 0.3"),
]


def read_loss(path, loss_type):

    steps, losses = [], []
    loss_name = "pred_loss" if loss_type == "prediction" else "importance_loss"
    pattern = re.compile(rf"\[val\]\s+step\s+(\d+).*?\|\s+{loss_name}\s+([\d.eE+-]+)")

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


def plot_loss(task, loss_type, backbone):
    data = {}
    all_steps = []
    all_losses = []

    if loss_type == "prediction":
        baseline_path = os.path.join(RESULTS_DIR, f"scm_{task}_imp_weight0_epoch4_{backbone}", "train_log.txt",)
        base_steps, base_losses = read_loss(baseline_path, "prediction")
        if not base_steps:
            raise ValueError(f"No prediction-loss data: {baseline_path}")
        all_steps.extend(base_steps)
        all_losses.extend(base_losses)

    for start, _ in STARTS:
        data[start] = {}

        for method, label in METHODS.items():
            path = os.path.join(RESULTS_DIR, f"scm_{task}_{method}_w_imp_{start}_epoch4_{backbone}", "train_log.txt",)

            if not os.path.exists(path):
                print(f"Missing: {path}")
                continue

            steps, losses = read_loss(path, loss_type)

            if not steps:
                print(f"No {loss_type}-loss data: {path}")
                continue

            data[start][method] = (steps, losses)
            all_steps.extend(steps)
            all_losses.extend(losses)

    if not all_steps:
        raise ValueError(f"No {loss_type}-loss data found for {task}, {backbone}")

    loss_min = min(all_losses)
    loss_max = max(all_losses)

    loss_pad = 0.03 * (loss_max - loss_min)

    if loss_pad == 0:
        loss_pad = max(0.03 * abs(loss_max), 1e-4)

    if loss_type == "importance":
        loss_ylim = (max(0, loss_min - loss_pad), loss_max + loss_pad,)
    else:
        loss_ylim = (loss_min - loss_pad, loss_max + loss_pad,)

    xmax = max(all_steps)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    for ax, (start, panel_title) in zip(axes, STARTS):

        for method, label in METHODS.items():
            if method not in data[start]:
                continue
            steps, losses = data[start][method]
            ax.plot(steps, losses, marker="o", markersize=3.5, linewidth=1.8, label=label)

        if loss_type == "prediction":
            ax.plot(base_steps, base_losses, marker="o", markersize=3.5, linewidth=2.0, linestyle="--", color="black", label="Importance Weight = 0",)


        if start in ("start0.15", "start0.3"):
            reference_path = os.path.join(RESULTS_DIR, f"scm_{task}_gradient_w_imp_{start}_epoch4_{backbone}", "train_log.txt",)
            if os.path.exists(reference_path):
                start_step = read_importance_start(reference_path)
                if start_step is not None:
                    ax.axvline(start_step, color="gray", linestyle=":", linewidth=1.5,)

        ax.set_title(panel_title)
        ax.set_xlabel("Training Step")

        if loss_type == "prediction":
            ax.set_ylabel("Validation Prediction Loss")
        else:
            ax.set_ylabel("Validation Importance Loss")

        ax.set_xlim(0, xmax)
        ax.set_ylim(*loss_ylim)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right", fontsize=8)

    title = "Classification" if task == "classification" else "Regression"

    if loss_type == "prediction":
        fig.suptitle(f"{title} — Prediction Loss ({backbone.upper()})", fontsize=14,)
        filename = (f"{task}_prediction_comparison_epoch4_{backbone}.png")
    else:
        fig.suptitle(f"{title} — Importance-Loss Dynamics ({backbone.upper()})", fontsize=14,)
        filename = (f"{task}_importance_loss_comparison_epoch4_{backbone}.png")

    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-kind", choices=["classification", "regression"], required=True,)
    parser.add_argument("--loss", choices=["prediction", "importance"], required=True,)
    parser.add_argument("--backbone", choices=["v1", "v2"], required=True,)
    args = parser.parse_args()

    plot_loss(
        task=args.task_kind,
        loss_type=args.loss,
        backbone=args.backbone,
    )
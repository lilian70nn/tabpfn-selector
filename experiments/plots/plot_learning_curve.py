from pathlib import Path
import argparse
import re
import matplotlib.pyplot as plt


def read_val_pred_loss(log_path):
    steps, losses = [], []
    with open(log_path, "r") as f:
        for line in f:
            if not line.startswith("[val]"):
                continue
            step_match = re.search(r"step\s+(\d+)", line)
            loss_match = re.search(r"pred_loss\s+([0-9.eE+-]+)", line)
            if step_match and loss_match:
                steps.append(int(step_match.group(1)))
                losses.append(float(loss_match.group(1)))
    return steps, losses


def main(w_imp_path, wo_imp_path, save_path=None):
    steps_w, losses_w = read_val_pred_loss(w_imp_path)
    steps_wo, losses_wo = read_val_pred_loss(wo_imp_path)

    plt.figure(figsize=(7, 5))
    plt.plot(steps_w, losses_w, marker="o", markersize=3, label="w/ importance")
    plt.plot(steps_wo, losses_wo, marker="o", markersize=3, label="w/o importance")
    plt.xlabel("Training Step")
    plt.ylabel("Validation Prediction Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        save_path = Path("reg_val_pred_loss_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("w_imp", type=Path)
    # parser.add_argument("wo_imp", type=Path)
    # args = parser.parse_args()

    path1 = "results/training/scm_reg_w_imp_training_results/train_log.txt"
    path2 = "results/training/scm_reg_wo_imp_training_results/train_log.txt"
    save_path = "results/figures/reg_val_pred_loss_comparison.png"
    main(path1, path2, save_path)
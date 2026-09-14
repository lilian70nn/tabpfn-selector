import re
import matplotlib.pyplot as plt

def plot_joint_objectives(input_path, output_path):
    steps, pred_losses, imp_losses = [], [], []

    pattern = re.compile(
        r"\[val\]\s+step\s+(\d+).*?"
        r"pred_loss\s+([0-9.eE+-]+).*?"
        r"importance_loss\s+([0-9.eE+-]+)"
    )

    with open(input_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                pred_losses.append(float(match.group(2)))
                imp_losses.append(float(match.group(3)))

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        steps, pred_losses,
        marker="o", markersize=4, linewidth=2,
        label="Prediction Loss"
    )

    l2 = ax2.plot(
        steps, imp_losses,
        marker="s", markersize=4, linewidth=2,
        linestyle="--",
        label="Importance Loss"
    )

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Prediction Loss")
    ax2.set_ylabel("Validation Importance Loss")

    ax1.grid(True, alpha=0.25)

    lines = l1 + l2
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":

    path1 = "results/training/scm_cls_w_imp_training_results/train_log.txt"
    path2 = "results/training/scm_reg_w_imp_training_results/train_log.txt"

    out_path1 = "results/figures/cls_joint_objectives.png"
    out_path2 = "results/figures/reg_joint_objectives.png"

    plot_joint_objectives(path1, out_path1)
    plot_joint_objectives(path2, out_path2)


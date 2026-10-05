import re
import matplotlib.pyplot as plt

def read_val_log(path):
    steps, pred_losses, imp_losses = [], [], []
    pattern = re.compile(
        r"\[val\]\s+step\s+(\d+).*?\|\s+pred_loss\s+([\d.eE+-]+)"
        r"(?:\s+\|\s+importance_loss\s+([\d.eE+-]+))?"
    )

    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                pred_losses.append(float(m.group(2)))
                imp_losses.append(float(m.group(3)) if m.group(3) is not None else None)

    return steps, pred_losses, imp_losses

def read_importance_start(path):
    pattern = re.compile(r"\[importance supervision ON\]\s+step=(\d+)")
    with open(path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return int(m.group(1))
    return 0


def plot_start_comparison(start0_log, start03_log, wo_imp_log, out_path, title=None):

    start0_step = read_importance_start(start0_log)
    start03_step = read_importance_start(start03_log)

    s0, p0, i0 = read_val_log(start0_log)
    s03, p03, i03 = read_val_log(start03_log)
    sw, pw, _ = read_val_log(wo_imp_log)

    i0_x = [s for s, v in zip(s0, i0) if v is not None]
    i0_y = [v for v in i0 if v is not None]
    i03_x = [s for s, v in zip(s03, i03) if v is not None]
    i03_y = [v for v in i03 if v is not None]

    # Shared scales across the two panels
    pred_all = p0 + p03 + pw
    imp_all = i0_y + i03_y
    pred_pad = 0.03 * (max(pred_all) - min(pred_all))
    imp_pad = 0.05 * (max(imp_all) - min(imp_all))
    pred_ylim = (min(pred_all) - pred_pad, max(pred_all) + pred_pad)
    imp_ylim = (min(imp_all) - imp_pad, max(imp_all) + imp_pad)
    xmax = max(max(s0), max(s03), max(sw))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, steps, pred, imp_x, imp_y, panel_title, start_step in [
        (axes[0], s0, p0, i0_x, i0_y, "Start = 0", start0_step),
        (axes[1], s03, p03, i03_x, i03_y, "Start = 0.3", start03_step),
    ]:
        # Prediction loss
        l1, = ax.plot(steps, pred, marker="o", markersize=4, label="Prediction Loss")
        l2, = ax.plot(sw, pw, linestyle="--", linewidth=2, label="w/o Importance")

        ax.set_title(panel_title)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Validation Prediction Loss")
        ax.set_xlim(0, xmax)
        ax.set_ylim(*pred_ylim)
        ax.grid(alpha=0.3)

        if start_step > 0:
            ax.axvline(start_step, color="gray", linestyle=":", linewidth=1.5)

        # Importance loss
        ax2 = ax.twinx()
        l3, = ax2.plot(imp_x, imp_y, marker="s", markersize=4,
                       linestyle="--", label="Importance Loss")
        ax2.set_ylabel("Validation Importance Loss")
        ax2.set_ylim(*imp_ylim)

        ax.legend([l1, l2, l3],
                  ["Prediction Loss", "w/o Importance", "Importance Loss"],
                  loc="upper right")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    
    # Classification
    wo_cls = "results/training/scm_classification_wo_imp/train_log.txt"

    plot_start_comparison(
        "results/training/scm_classification_gradient_w_imp_start0/train_log.txt",
        "results/training/scm_classification_gradient_w_imp_start0.3/train_log.txt",
        wo_cls, "results/figures/classification_gradient_delay_comparison.png",
        title="Classification — Gradient"
    )

    plot_start_comparison(
        "results/training/scm_classification_eigen_top1_w_imp_start0/train_log.txt",
        "results/training/scm_classification_eigen_top1_w_imp_start0.3/train_log.txt",
        wo_cls, "results/figures/classification_eigen_top1_delay_comparison.png",
        title="Classification — Eigen Top-1"
    )

    plot_start_comparison(
        "results/training/scm_classification_eigen_90_w_imp_start0/train_log.txt",
        "results/training/scm_classification_eigen_90_w_imp_start0.3/train_log.txt",
        wo_cls, "results/figures/classification_eigen90_delay_comparison.png",
        title="Classification — Eigen 90"
    )

    plot_start_comparison(
        "results/training/scm_classification_mi_w_imp_start0/train_log.txt",
        "results/training/scm_classification_mi_w_imp_start0.3/train_log.txt",
        wo_cls, "results/figures/classification_mi_delay_comparison.png",
        title="Classification — MI"
    )

    plot_start_comparison(
        "results/training/scm_classification_marginal_w_imp_start0/train_log.txt",
        "results/training/scm_classification_marginal_w_imp_start0.3/train_log.txt",
        wo_cls, "results/figures/classification_marginal_delay_comparison.png",
        title="Classification — Marginal"
    )

    plot_start_comparison(
        "results/training/scm_classification_permutation_w_imp_start0/train_log.txt",
        "results/training/scm_classification_permutation_w_imp_start0.3/train_log.txt",
        wo_cls, "results/figures/classification_permutation_delay_comparison.png",
        title="Classification — Permutation"
    )


    # Regression
    wo_reg = "results/training/scm_regression_wo_imp/train_log.txt"

    plot_start_comparison(
        "results/training/scm_regression_gradient_w_imp_start0/train_log.txt",
        "results/training/scm_regression_gradient_w_imp_start0.3/train_log.txt",
        wo_reg, "results/figures/regression_gradient_delay_comparison.png",
        title="Regression — Gradient"
    )

    plot_start_comparison(
        "results/training/scm_regression_eigen_top1_w_imp_start0/train_log.txt",
        "results/training/scm_regression_eigen_top1_w_imp_start0.3/train_log.txt",
        wo_reg, "results/figures/regression_eigen_top1_delay_comparison.png",
        title="Regression — Eigen Top-1"
    )

    plot_start_comparison(
        "results/training/scm_regression_eigen_90_w_imp_start0/train_log.txt",
        "results/training/scm_regression_eigen_90_w_imp_start0.3/train_log.txt",
        wo_reg, "results/figures/regression_eigen90_delay_comparison.png",
        title="Regression — Eigen 90"
    )

    plot_start_comparison(
        "results/training/scm_regression_mi_w_imp_start0/train_log.txt",
        "results/training/scm_regression_mi_w_imp_start0.3/train_log.txt",
        wo_reg, "results/figures/regression_mi_delay_comparison.png",
        title="Regression — MI"
    )

    plot_start_comparison(
        "results/training/scm_regression_marginal_w_imp_start0/train_log.txt",
        "results/training/scm_regression_marginal_w_imp_start0.3/train_log.txt",
        wo_reg, "results/figures/regression_marginal_delay_comparison.png",
        title="Regression — Marginal"
    )

    plot_start_comparison(
        "results/training/scm_regression_permutation_w_imp_start0/train_log.txt",
        "results/training/scm_regression_permutation_w_imp_start0.3/train_log.txt",
        wo_reg, "results/figures/regression_permutation_delay_comparison.png",
        title="Regression — Permutation"
    )
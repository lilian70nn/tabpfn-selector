from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_real_data_comparison(summary_wo, summary_w, summary_w_long, metric, output_path):
    df_wo = pd.read_csv(summary_wo)
    df_w = pd.read_csv(summary_w)
    df_long = pd.read_csv(summary_w_long)

    datasets = df_wo["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.25

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.1), 5))

    ax.bar(x - width, df_wo[mean_col], width, yerr=df_wo[std_col], capsize=3, label="w/o importance (1×)")
    ax.bar(x, df_w[mean_col], width, yerr=df_w[std_col], capsize=3, label="w/ importance (1×)")
    ax.bar(x + width, df_long[mean_col], width, yerr=df_long[std_col], capsize=3, label="w/ importance (1.5×)")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha="right")
    ax.set_ylabel("Accuracy" if metric == "accuracy" else r"$R^2$")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output_path}")



if __name__ == "__main__":
    plot_real_data_comparison(
        "results/evaluation/scm_cls_wo_imp_training_results/real_eval_30repeats_summary.csv",
        "results/evaluation/scm_cls_w_imp_training_results/real_eval_30repeats_summary.csv",
        "results/evaluation/scm_cls_w_imp_training_long_results/real_eval_30repeats_summary.csv",
        metric="accuracy",
        output_path="results/figures/cls_real_data_comparison.png",
    )


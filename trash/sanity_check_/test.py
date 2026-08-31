import numpy as np
import matplotlib.pyplot as plt
import torch

from Trash.scm_task_v9 import WeightedMixedScalarSCMTask


# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cpu")

TASK_KWARGS = dict(
    n_min=400,
    n_max=512,
    d_min=8,
    d_max=16,
    test_frac=0.15,
    p_missing=0.05,
    num_roots=4,
    num_layers=5,
    hidden_width_min=8,
    hidden_width_max=12,
    final_width=1,
    connection_probs=(0.30, 0.30, 0.45, 0.85),
    # min_parents_per_node=2,
    edge_weight_concentration=0.15,
    latent_noise_scale=0.0,
    observation_noise_scale=0.03,
    dominant_mass_threshold=0.70,
    dominant_feature_fraction=0.70,
    observation_type_probs=(0.70, 0.15, 0.15),
    categorical_cardinalities=(2, 3, 4, 5, 6),
    categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    min_samples_per_category=8,
    min_component_weight=0.05,
    prototype_max_attempts=8,
    prototype_min_separation=1.0,
    binning_jitter=0.20,
    source_prior_probs=(0.45, 0.20, 0.15, 0.05),
    # root_mixture_component_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    # root_mixture_separation_min=1.5,
    # root_mixture_separation_max=3.0,
    # root_mixture_scale_min=0.40,
    # root_mixture_scale_max=0.90,
    linear_activation_prob=0.60,
    small_mlp_prob=0.25,
    soft_tree_prob=0.15,
    small_mlp_hidden_dim=None,
    soft_tree_depth=2,
    soft_tree_temperature=0.5,
    device=DEVICE,
)


# =============================================================================
# Main
# =============================================================================

def main():
    task = WeightedMixedScalarSCMTask(
        num_classes=3,
        dag_seed=0,
        aleatoric_seed=100_000,
        x_seed=200_000,
        **TASK_KWARGS,
    )

    scm = task.scm
    influence = scm.compute_sampling_influence(target_node_idx=0)

    print("=" * 80)
    print("Layer influence statistics")
    print("=" * 80)

    means = []
    medians = []
    maxs = []
    sums = []

    for layer, values in enumerate(influence):
        values = values.detach().cpu().numpy()

        mean = float(values.mean())
        median = float(np.median(values))
        maximum = float(values.max())
        total = float(values.sum())

        means.append(mean)
        medians.append(median)
        maxs.append(maximum)
        sums.append(total)

        print(
            f"Layer {layer}: "
            f"nodes={len(values)}, "
            f"mean={mean:.6f}, "
            f"median={median:.6f}, "
            f"min={values.min():.6f}, "
            f"max={maximum:.6f}, "
            f"sum={total:.6f}"
        )

    layers = np.arange(len(influence))

    plt.figure(figsize=(7, 5))
    plt.plot(layers, means, marker="o", label="mean")
    plt.plot(layers, medians, marker="o", label="median")
    plt.plot(layers, maxs, marker="o", label="max")
    plt.plot(layers, sums, marker="o", label="sum")
    plt.xlabel("Layer")
    plt.ylabel("Sampling influence")
    plt.xticks(layers)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))

    for layer, values in enumerate(influence):
        values = values.detach().cpu().numpy()
        plt.scatter(np.full(len(values), layer), values, alpha=0.7)

    plt.xlabel("Layer")
    plt.ylabel("Node sampling influence")
    plt.xticks(layers)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
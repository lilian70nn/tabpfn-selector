import torch
import numpy as np

from Trash.scm_task_v11 import WeightedMixedScalarSCMTask


TASK_KWARGS = dict(
    num_classes=3,
    n_min=400,
    n_max=512,
    d_min=8,
    d_max=16,
    test_frac=0.15,
    p_missing=0.05,
    num_roots=8,
    num_layers=5,
    hidden_width_min=6,
    hidden_width_max=10,
    final_width=1,
    connection_probs=(0.20, 0.20, 0.30, 0.85),
    edge_weight_concentration=0.30,
    latent_noise_scale=0.0,
    sampling_penalty=0.25,
    observation_noise_scale=0.03,
    observation_type_probs=(0.70, 0.15, 0.15),
    categorical_cardinalities=(2, 3, 4, 5, 6),
    categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    min_samples_per_category=8,
    min_component_weight=0.05,
    prototype_max_attempts=8,
    prototype_min_separation=1.0,
    binning_jitter=0.20,
    source_prior_probs=(0.45, 0.20, 0.15, 0.05),
    linear_activation_prob=0.60,
    small_mlp_prob=0.25,
    soft_tree_prob=0.15,
    small_mlp_hidden_dim=None,
    soft_tree_depth=2,
    soft_tree_temperature=0.5,
    device=torch.device("cpu"),
    dag_seed=4,
    x_seed=5,
    aleatoric_seed=6,
)


def summarize_latents(all_latents):
    print()
    print("=" * 100)
    print("LATENT SCALE BY LAYER")
    print("=" * 100)

    for layer_idx, layer in enumerate(all_latents):
        values = torch.cat([x.detach().float() for x in layer], dim=1)

        node_mean = values.mean(dim=0)
        node_std = values.std(dim=0, unbiased=False)
        node_absmax = values.abs().max(dim=0).values

        finite_ratio = torch.isfinite(values).float().mean().item()

        print(
            f"layer {layer_idx:02d} | "
            f"width={values.shape[1]:2d} | "
            f"mean_abs_median={node_mean.abs().median().item():10.4f} | "
            f"std_median={node_std.median().item():10.4f} | "
            f"std_mean={node_std.mean().item():10.4f} | "
            f"std_max={node_std.max().item():10.4f} | "
            f"node_absmax_median={node_absmax.median().item():10.4f} | "
            f"global_absmax={values.abs().max().item():12.4f} | "
            f"finite={finite_ratio:.6f}"
        )


def summarize_selected_features(task):
    info = task.info

    feature_ids = info["feature_ids"].detach().cpu().numpy()
    feature_importance = info["feature_importance"].detach().cpu().numpy()

    print()
    print("=" * 100)
    print("SELECTED FEATURE IMPORTANCE")
    print("=" * 100)

    print("feature ids:")
    print(feature_ids)

    print()
    print("feature importance:")
    print(np.round(feature_importance, 6))

    print()
    print(
        "importance stats | "
        f"min={feature_importance.min():.6f} | "
        f"median={np.median(feature_importance):.6f} | "
        f"mean={feature_importance.mean():.6f} | "
        f"max={feature_importance.max():.6f}"
    )


def main():
    task = WeightedMixedScalarSCMTask(**TASK_KWARGS)

    with torch.enable_grad():
        all_latents = task.scm.forward(
            task.n,
            latent_noise_scale=task.latent_noise_scale,
        )

    summarize_latents(all_latents)
    summarize_selected_features(task)

    print()
    print("=" * 100)
    print("OBSERVED DATA")
    print("=" * 100)

    X_train = task.X_train.detach().float()
    X_test = task.X_test.detach().float()

    print("X_train finite ratio:", torch.isfinite(X_train).float().mean().item())
    print("X_test finite ratio :", torch.isfinite(X_test).float().mean().item())
    print("X_train NaN ratio   :", torch.isnan(X_train).float().mean().item())
    print("X_test NaN ratio    :", torch.isnan(X_test).float().mean().item())


if __name__ == "__main__":
    main()
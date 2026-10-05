import numpy as np
import torch

from src.data.scm_task_v10 import WeightedMixedScalarSCMTask


DEVICE = torch.device("cpu")

TASK_KWARGS = dict(
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
    device=DEVICE,
)


NUM_TABLES = 100
BASE_SEED = 0

all_strengths = []


for table_id in range(NUM_TABLES):

    num_classes = 2 + (table_id % 3)

    task = WeightedMixedScalarSCMTask(
        num_classes=num_classes,
        dag_seed=BASE_SEED + table_id,
        aleatoric_seed=100_000 + BASE_SEED + table_id,
        x_seed=200_000 + BASE_SEED + table_id,
        **TASK_KWARGS,
    )

    strength = (
        task.info["feature_strength"]
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    all_strengths.append(strength)

    print(
        f"table={table_id:03d} "
        f"classes={num_classes} "
        f"d={len(strength):2d} | "
        f"min={strength.min():.6f} "
        f"median={np.median(strength):.6f} "
        f"mean={strength.mean():.6f} "
        f"max={strength.max():.6f}"
    )


all_strengths = np.concatenate(all_strengths)

positive = all_strengths[all_strengths > 1e-12]


print()
print("=" * 80)
print("ALL FEATURE STRENGTHS")
print("=" * 80)

for q in [
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
]:
    print(
        f"p{int(q * 100):02d} = "
        f"{np.quantile(all_strengths, q):.8f}"
    )

print()
print("mean =", all_strengths.mean())
print("std  =", all_strengths.std())
print("zero fraction =", np.mean(all_strengths <= 1e-12))


print()
print("=" * 80)
print("POSITIVE FEATURE STRENGTHS")
print("=" * 80)

for q in [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]:
    print(
        f"p{int(q * 100):02d} = "
        f"{np.quantile(positive, q):.8f}"
    )

print()
print("positive mean =", positive.mean())
print("positive std  =", positive.std())
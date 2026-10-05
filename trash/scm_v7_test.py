import time
from collections import Counter, defaultdict

import torch

from Trash.scm_task_v7 import AdaptiveObservationHead, MixedLatentSCMTask


def build_task(device: torch.device, dag_seed: int, aleatoric_seed: int, x_seed: int):
    return MixedLatentSCMTask(
        num_classes=3,
        n_min=500,
        n_max=500,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_missing=0.05,
        num_roots=3,
        num_layers=5,
        max_nodes_per_layer=8,
        latent_dim=6,
        latent_noise_scale=0.05,
        observation_noise_scale=0.05,
        root_prior_probs=(0.45, 0.20, 0.15, 0.05, 0.15),
        root_mixture_component_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        root_mixture_separation_min=1.5,
        root_mixture_separation_max=3.0,
        root_mixture_scale_min=0.40,
        root_mixture_scale_max=0.90,
        dominant_parent_prob=0.40,
        dominant_parent_weight=0.75,
        # continuous / prototype / threshold-binning
        observation_type_probs=(0.65, 0.175, 0.175),
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        min_samples_per_category=8,
        min_component_weight=0.05,
        prototype_max_attempts=8,
        prototype_min_separation=1.0,
        binning_jitter=0.20,
        dag_seed=dag_seed,
        aleatoric_seed=aleatoric_seed,
        x_seed=x_seed,
        device=device,
    )


def get_outputs(task):
    return task.X_train, task.y_train, task.X_test, task.y_test, task.info


def print_single_task(device: torch.device):
    print("========== Generate one task ==========")
    start = time.perf_counter()
    task = build_task(device, 1, 2, 3)
    elapsed = time.perf_counter() - start
    X_train, y_train, X_test, y_test, info = get_outputs(task)
    X_full = torch.cat([X_train, X_test], dim=0)
    y_full = torch.cat([y_train, y_test], dim=0)

    print(f"Device: {device}")
    print(f"Generation time: {elapsed:.4f}s")
    print(f"X_train={tuple(X_train.shape)}, X_test={tuple(X_test.shape)}")
    print(f"y_train={tuple(y_train.shape)}, y_test={tuple(y_test.shape)}")
    print(f"Missing rate={torch.isnan(X_full).float().mean().item():.4f}")
    print(f"Target counts={torch.bincount(y_full.long()).tolist()}")
    print(f"Roots={info['root_prior_types']}")
    print(f"Root mixture M={info['root_mixture_components'].cpu().tolist()}")

    feature_type = info["feature_type"].cpu()
    cardinality = info["cardinality"].cpu()
    type_names = info["feature_observation_type_names"]
    quality = info["feature_observation_quality"].cpu()

    print("\n========== Feature summary ==========")
    for col, name in enumerate(type_names):
        column = X_full[:, col].cpu()
        valid = column[~torch.isnan(column)]
        is_cat = int(feature_type[col].item()) == MixedLatentSCMTask.CATEGORICAL
        if is_cat:
            k = int(cardinality[col].item())
            counts = torch.bincount(valid.long(), minlength=k).tolist()
            extra = ""
            if name == "prototype_discretization":
                extra = f", prototypes_shape={tuple(info['feature_prototypes'][col].shape)}"
            elif name == "threshold_binning":
                extra = f", thresholds={info['feature_thresholds'][col].cpu().tolist()}"
            print(
                f"x_{col+1}: {name}, K={k}, counts={counts}, "
                f"quality={quality[col].item():.3f}{extra}"
            )
        else:
            print(
                f"x_{col+1}: {name}, continuous, mean={valid.mean().item():.3f}, "
                f"std={valid.std(unbiased=False).item():.3f}"
            )


def benchmark(
    device: torch.device,
    num_tasks: int = 1000,
) -> None:
    final_mechanism_counts = Counter()
    sampled_mechanism_counts = Counter()
    actual_type_counts = Counter()

    cardinality_by_mechanism = defaultdict(Counter)

    categorical_features = 0
    total_features = 0

    prototype_selected = 0
    prototype_success = 0
    prototype_fallback = 0

    binning_selected = 0
    continuous_selected = 0

    times = []

    categorical_counts_per_task = []

    for seed in range(num_tasks):
        start = time.perf_counter()

        task = build_task(
            device,
            seed,
            10_000 + seed,
            20_000 + seed,
        )

        times.append(
            time.perf_counter() - start
        )

        _, _, _, _, info = get_outputs(task)

        names = info[
            "feature_observation_type_names"
        ]

        feature_type = (
            info["feature_type"]
            .detach()
            .cpu()
        )

        cardinality = (
            info["cardinality"]
            .detach()
            .cpu()
        )

        heads = task.feature_observation_heads

        task_categorical = 0
        total_features += len(names)

        for i, (
            name,
            head,
        ) in enumerate(
            zip(names, heads)
        ):
            final_mechanism_counts[name] += 1

            sampled_id = (
                head.sampled_observation_type_id
            )

            sampled_name = (
                AdaptiveObservationHead
                .OBSERVATION_TYPE_NAMES[
                    sampled_id
                ]
            )

            sampled_mechanism_counts[
                sampled_name
            ] += 1

            if sampled_id == AdaptiveObservationHead.CONTINUOUS:
                continuous_selected += 1

            elif sampled_id == AdaptiveObservationHead.PROTOTYPE:
                prototype_selected += 1

            elif sampled_id == AdaptiveObservationHead.BINNING:
                binning_selected += 1

            is_categorical = (
                int(feature_type[i].item())
                == MixedLatentSCMTask.CATEGORICAL
            )

            actual_type_counts[
                "categorical"
                if is_categorical
                else "continuous"
            ] += 1

            if is_categorical:
                categorical_features += 1
                task_categorical += 1

                k = int(
                    cardinality[i].item()
                )

                cardinality_by_mechanism[
                    name
                ][
                    k
                ] += 1

            if (
                sampled_id
                == AdaptiveObservationHead.PROTOTYPE
            ):
                if (
                    name
                    == "prototype_discretization"
                ):
                    prototype_success += 1
                else:
                    prototype_fallback += 1

        categorical_counts_per_task.append(
            task_categorical
        )

    times_t = torch.tensor(
        times,
        dtype=torch.float64,
    )

    observed_sampled_probs = {
        name: count / total_features
        for name, count
        in sampled_mechanism_counts.items()
    }

    print(
        "\n========== Benchmark result =========="
    )

    print(f"Tasks: {num_tasks}")
    print(f"Total features: {total_features}")

    print(
        f"Mean time: "
        f"{times_t.mean().item():.4f}s"
    )

    print(
        f"Median time: "
        f"{times_t.median().item():.4f}s"
    )

    print(
        f"Min time: "
        f"{times_t.min().item():.4f}s"
    )

    print(
        f"Max time: "
        f"{times_t.max().item():.4f}s"
    )

    print(
        "Overall categorical ratio: "
        f"{categorical_features / total_features:.2%}"
    )

    print(
        "Mean categorical features per task: "
        f"{sum(categorical_counts_per_task) / num_tasks:.2f}"
    )

    print(
        "Sampled mechanism counts:",
        dict(sampled_mechanism_counts),
    )

    print(
        "Observed sampled mechanism probabilities:",
        {
            name: round(prob, 4)
            for name, prob
            in observed_sampled_probs.items()
        },
    )

    print(
        "Final mechanism counts:",
        dict(final_mechanism_counts),
    )

    print(
        "Actual feature-type counts:",
        dict(actual_type_counts),
    )

    print(
        "\nPrototype statistics:"
    )

    print(
        f"  selected: {prototype_selected}"
    )

    print(
        f"  successful: {prototype_success}"
    )

    print(
        f"  fallback: {prototype_fallback}"
    )

    if prototype_selected > 0:
        print(
            "  success rate: "
            f"{prototype_success / prototype_selected:.2%}"
        )

        print(
            "  fallback rate: "
            f"{prototype_fallback / prototype_selected:.2%}"
        )

    print(
        "\nSelected mechanism totals:"
    )

    print(
        f"  continuous: {continuous_selected}"
    )

    print(
        f"  prototype: {prototype_selected}"
    )

    print(
        f"  binning: {binning_selected}"
    )

    print(
        "\nCardinality by final mechanism:"
    )

    for name, counter in (
        cardinality_by_mechanism.items()
    ):
        print(
            f"  {name}: "
            f"{dict(sorted(counter.items()))}"
        )

    print(
        "\nCategorical features per task:"
    )

    categorical_counts_tensor = torch.tensor(
        categorical_counts_per_task,
        dtype=torch.float64,
    )

    print(
        f"  mean="
        f"{categorical_counts_tensor.mean().item():.3f}"
    )

    print(
        f"  std="
        f"{categorical_counts_tensor.std(unbiased=False).item():.3f}"
    )

    print(
        f"  min="
        f"{int(categorical_counts_tensor.min().item())}"
    )

    print(
        f"  max="
        f"{int(categorical_counts_tensor.max().item())}"
    )


if __name__ == "__main__":
    device = torch.device("cpu")
    print_single_task(device)
    benchmark(device, num_tasks=1000)

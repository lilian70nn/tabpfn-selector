# scm_v6_test.py

import time
from collections import Counter

import torch

from Trash.scm_task_v6 import MixedLatentSCMTask


def get_task_outputs(task):
    required = [
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "info",
    ]

    missing = [
        name
        for name in required
        if not hasattr(task, name)
    ]

    if missing:
        raise AttributeError(
            "GenerateTask 没有保存以下属性："
            f"{missing}。"
        )

    return (
        task.X_train,
        task.y_train,
        task.X_test,
        task.y_test,
        task.info,
    )


def build_task(
    device: torch.device,
    dag_seed: int,
    aleatoric_seed: int,
    x_seed: int,
) -> MixedLatentSCMTask:
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
        latent_dim=8,

        latent_noise_scale=0.05,
        observation_noise_scale=0.05,

        root_prior_probs=(
            0.40,
            0.20,
            0.15,
            0.05,
            0.20,
        ),

        root_mixture_component_probs=(
            0.35,
            0.30,
            0.20,
            0.10,
            0.05,
        ),

        root_mixture_separation_min=1.0,
        root_mixture_separation_max=3.0,
        root_mixture_scale_min=0.35,
        root_mixture_scale_max=0.90,

        dominant_parent_prob=0.40,
        dominant_parent_weight=0.80,

        max_cardinality=6,
        min_samples_per_category=8,
        min_component_weight=0.05,
        min_cluster_separation=1.5,
        min_bic_improvement=10.0,

        num_em_restarts=3,
        max_em_iterations=75,
        em_tolerance=1e-4,
        variance_floor=1e-3,

        dag_seed=dag_seed,
        aleatoric_seed=aleatoric_seed,
        x_seed=x_seed,

        device=device,
    )


def print_one_task(
    device: torch.device,
) -> None:
    print(
        "========== Generate one task =========="
    )

    start = time.perf_counter()

    task = build_task(
        device=device,
        dag_seed=1,
        aleatoric_seed=2,
        x_seed=3,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        info,
    ) = get_task_outputs(
        task
    )

    X_full = torch.cat(
        [
            X_train,
            X_test,
        ],
        dim=0,
    )

    y_full = torch.cat(
        [
            y_train,
            y_test,
        ],
        dim=0,
    )

    print(
        f"Generation time: "
        f"{elapsed:.4f}s"
    )

    print(
        f"X_train: "
        f"{tuple(X_train.shape)}"
    )

    print(
        f"X_test:  "
        f"{tuple(X_test.shape)}"
    )

    print(
        "Target counts:",
        torch.bincount(
            y_full.long().cpu(),
            minlength=3,
        ).tolist(),
    )

    print(
        "Missing rate:",
        round(
            float(
                torch.isnan(
                    X_full
                ).float().mean().item()
            ),
            4,
        ),
    )

    print(
        "Root priors:",
        info[
            "root_prior_types"
        ],
    )

    print(
        "Root mixture M:",
        info[
            "root_mixture_components"
        ].detach().cpu().tolist(),
    )

    feature_type = (
        info[
            "feature_type"
        ].detach().cpu()
    )

    cardinality = (
        info[
            "cardinality"
        ].detach().cpu()
    )

    bic_improvement = (
        info[
            "categorical_cluster_score"
        ].detach().cpu()
    )

    categorical_mask = (
        feature_type
        == MixedLatentSCMTask.CATEGORICAL
    )

    print(
        "Categorical features:",
        int(
            categorical_mask.sum().item()
        ),
        "/",
        int(
            feature_type.numel()
        ),
    )

    for col in range(
        int(
            feature_type.numel()
        )
    ):
        if bool(
            categorical_mask[
                col
            ].item()
        ):
            K = int(
                cardinality[
                    col
                ].item()
            )

            values = X_full[
                :,
                col,
            ]

            valid = values[
                ~torch.isnan(values)
            ].long()

            counts = torch.bincount(
                valid,
                minlength=K,
            )

            print(
                f"x_{col + 1}: "
                f"categorical, "
                f"K={K}, "
                f"counts={counts.tolist()}, "
                f"BIC improvement="
                f"{float(bic_improvement[col].item()):.3f}"
            )


def benchmark_multiple_tasks(
    device: torch.device,
    num_tasks: int = 20,
) -> None:
    print(
        "\n========== Benchmark =========="
    )

    elapsed_times: list[
        float
    ] = []

    total_features = 0
    total_categorical = 0

    cardinalities: list[
        int
    ] = []

    bic_improvements: list[
        float
    ] = []

    root_prior_counter: Counter[
        str
    ] = Counter()

    mixture_component_counter: Counter[
        int
    ] = Counter()

    for seed in range(
        num_tasks
    ):
        start = time.perf_counter()

        task = build_task(
            device=device,
            dag_seed=seed,
            aleatoric_seed=(
                10_000 + seed
            ),
            x_seed=(
                20_000 + seed
            ),
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        elapsed_times.append(
            elapsed
        )

        _, _, _, _, info = (
            get_task_outputs(
                task
            )
        )

        feature_type = info[
            "feature_type"
        ]

        cardinality = info[
            "cardinality"
        ]

        bic_improvement = info[
            "categorical_cluster_score"
        ]

        categorical_mask = (
            feature_type
            == MixedLatentSCMTask.CATEGORICAL
        )

        n_features = int(
            feature_type.numel()
        )

        n_categorical = int(
            categorical_mask.sum().item()
        )

        total_features += (
            n_features
        )

        total_categorical += (
            n_categorical
        )

        selected_k = (
            cardinality[
                categorical_mask
            ]
            .detach()
            .cpu()
            .tolist()
        )

        selected_bic = (
            bic_improvement[
                categorical_mask
            ]
            .detach()
            .cpu()
            .tolist()
        )

        cardinalities.extend(
            int(value)
            for value in selected_k
        )

        bic_improvements.extend(
            float(value)
            for value in selected_bic
        )

        root_types = info[
            "root_prior_types"
        ]

        root_prior_counter.update(
            root_types
        )

        mixture_m = (
            info[
                "root_mixture_components"
            ]
            .detach()
            .cpu()
            .tolist()
        )

        mixture_component_counter.update(
            value
            for value in mixture_m
            if value > 0
        )

        print(
            f"Task {seed:03d}: "
            f"time={elapsed:.3f}s, "
            f"categorical="
            f"{n_categorical}/"
            f"{n_features}, "
            f"K={selected_k}"
        )

    times = torch.tensor(
        elapsed_times,
        dtype=torch.float64,
    )

    print(
        "\n========== Result =========="
    )

    print(
        f"Mean time: "
        f"{times.mean().item():.4f}s"
    )

    print(
        "Categorical ratio:",
        f"{total_categorical / total_features:.2%}",
    )

    print(
        "Root prior counts:",
        dict(
            root_prior_counter
        ),
    )

    print(
        "Mixture component counts:",
        dict(
            sorted(
                mixture_component_counter.items()
            )
        ),
    )

    if cardinalities:
        k_tensor = torch.tensor(
            cardinalities,
            dtype=torch.long,
        )

        unique_k, counts_k = (
            torch.unique(
                k_tensor,
                return_counts=True,
            )
        )

        distribution = {
            int(k.item()): int(
                count.item()
            )
            for k, count in zip(
                unique_k,
                counts_k,
            )
        }

        print(
            "Observed K distribution:",
            distribution,
        )

    if bic_improvements:
        values = torch.tensor(
            bic_improvements,
            dtype=torch.float64,
        )

        print(
            "BIC improvement: "
            f"mean="
            f"{values.mean().item():.3f}, "
            f"min="
            f"{values.min().item():.3f}, "
            f"max="
            f"{values.max().item():.3f}"
        )


if __name__ == "__main__":
    device = torch.device(
        "cpu"
    )

    print_one_task(
        device=device
    )

    # Start with 20 because GMM+EM is slower than K-means.
    benchmark_multiple_tasks(
        device=device,
        num_tasks=20,
    )

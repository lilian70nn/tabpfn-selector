# test_scm_generator_v5.py

# add doninant parent prob and weight to the task generation

import time
from collections import Counter

import torch

from Trash.scm_task_v5 import MixedLatentSCMTask


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
            f"{missing}。\n"
            "请检查 GenerateTask.__init__() 是否把 _generate() 的返回值保存为：\n"
            "self.X_train, self.y_train, self.X_test, "
            "self.y_test, self.info"
        )

    return (
        task.X_train,
        task.y_train,
        task.X_test,
        task.y_test,
        task.info,
    )


def print_root_prior_summary(info: dict) -> None:
    root_prior_types = info["root_prior_types"]

    root_prior_type_ids = (
        info["root_prior_type_ids"]
        .detach()
        .cpu()
        .tolist()
    )

    root_mixture_components = (
        info["root_mixture_components"]
        .detach()
        .cpu()
        .tolist()
    )

    print("\n========== Root prior summary ==========")

    for root_idx, (
        prior_name,
        prior_id,
        mixture_components,
    ) in enumerate(
        zip(
            root_prior_types,
            root_prior_type_ids,
            root_mixture_components,
        )
    ):
        if prior_name == "mixture":
            print(
                f"root_{root_idx}: "
                f"type={prior_name}, "
                f"type_id={prior_id}, "
                f"latent_mixture_components={mixture_components}"
            )
        else:
            print(
                f"root_{root_idx}: "
                f"type={prior_name}, "
                f"type_id={prior_id}"
            )


def print_feature_summary(
    X_full: torch.Tensor,
    info: dict,
) -> None:
    feature_type = info["feature_type"].detach().cpu()
    cardinality = info["cardinality"].detach().cpu()

    cluster_score = (
        info["categorical_cluster_score"]
        .detach()
        .cpu()
    )

    feature_ids = (
        info["feature_ids"]
        .detach()
        .cpu()
    )

    n_features = int(feature_type.numel())

    categorical_mask = (
        feature_type
        == MixedLatentSCMTask.CATEGORICAL
    )

    n_categorical = int(
        categorical_mask.sum().item()
    )

    n_continuous = (
        n_features
        - n_categorical
    )

    print("\n========== Feature summary ==========")
    projection_dim = (
        info["categorical_projection_dim"]
        .detach()
        .cpu()
    )
    print(f"Number of features:    {n_features}")
    print(f"Continuous features:   {n_continuous}")
    print(f"Categorical features:  {n_categorical}")

    for col in range(n_features):
        projection_d = int(
            projection_dim[col].item()
        )
        is_categorical = bool(
            categorical_mask[col].item()
        )

        feature_name = f"x_{col + 1}"
        source_node_id = int(feature_ids[col].item())
        score = float(cluster_score[col].item())

        column = X_full[:, col]
        valid = column[~torch.isnan(column)]

        if not is_categorical:
            if valid.numel() == 0:
                print(
                    f"{feature_name}: continuous, "
                    f"source_node={source_node_id}, "
                    "all values are missing"
                )
                continue

            print(
                f"{feature_name}: continuous, "
                f"source_node={source_node_id}, "
                f"mean={valid.mean().item():.3f}, "
                f"std={valid.std(unbiased=False).item():.3f}, "
                f"min={valid.min().item():.3f}, "
                f"max={valid.max().item():.3f}, "
                f"cluster_score={score:.3f}"
            )

        else:
            K = int(cardinality[col].item())
            valid_labels = valid.long()

            counts = torch.bincount(
                valid_labels,
                minlength=K,
            )

            centers = (
                info["feature_cluster_centers"][col]
                .detach()
                .cpu()
            )

            center_norms = torch.linalg.vector_norm(
                centers,
                dim=1,
            )

            print(
                f"{feature_name}: categorical, "
                f"source_node={source_node_id}, "
                f"projection_dim={projection_d}, "
                f"K={K}, "
                f"counts={counts.tolist()}, "
                f"centers_shape={tuple(centers.shape)}, "
                f"center_norms="
                f"{[round(x, 3) for x in center_norms.tolist()]}, "
                f"cluster_score={score:.3f}"
            )


def build_task(
    device: torch.device,
    dag_seed: int,
    aleatoric_seed: int,
    x_seed: int,
) -> MixedLatentSCMTask:
    """
    One shared configuration for both the single-task inspection and benchmark.
    """
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

        # Root prior probabilities:
        # Gaussian, Uniform, heavy-tailed, skewed, mixture.
        # root_prior_probs=(
        #     0.50,
        #     0.20,
        #     0.15,
        #     0.05,
        #     0.10,
        # ),
        root_prior_probs=(
            0.45,  # Gaussian
            0.20,  # Uniform
            0.15,  # heavy-tailed
            0.05,  # skewed
            0.15,  # mixture：原来是 0.10
        ),

        # Internal mixture-root component probabilities for M=2,...,6.
        # M is not the final observed categorical cardinality K.
        root_mixture_component_probs=(
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),

        root_mixture_separation_min=1.5,
        root_mixture_separation_max=3.0,
        root_mixture_scale_min=0.40,
        root_mixture_scale_max=0.90,

        # Latent-space categorical detection.
        max_cardinality=8,
        min_samples_per_category=8,
        min_component_weight=0.05,
        min_cluster_separation=1.5,
        min_cluster_score=0.40,

        num_kmeans_restarts=3,
        max_kmeans_iterations=50,
        kmeans_tolerance=1e-4,

        # Feature-specific subspace used only for categorical detection.
        # Each feature samples exactly one projection dimension.
        cluster_projection_dims=(
            2,
            3,
            4,
        ),

        cluster_projection_probs=(
            0.25,
            0.50,
            0.25,
        ),

        dag_seed=dag_seed,
        aleatoric_seed=aleatoric_seed,
        x_seed=x_seed,

        device=device,
        dominant_parent_prob=0.40,
        dominant_parent_weight=0.75,
    )


def generate_one_task(
    device: torch.device,
) -> None:
    print("========== Generate one task ==========")
    print(f"Device: {device}")

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
    ) = get_task_outputs(task)

    X_full = torch.cat(
        [X_train, X_test],
        dim=0,
    )

    y_full = torch.cat(
        [y_train, y_test],
        dim=0,
    )

    print(f"\nGeneration time: {elapsed:.4f} seconds")
    print(f"X_train shape: {tuple(X_train.shape)}")
    print(f"y_train shape: {tuple(y_train.shape)}")
    print(f"X_test shape:  {tuple(X_test.shape)}")
    print(f"y_test shape:  {tuple(y_test.shape)}")

    print("\nFirst 10 rows of X_train:")
    print(X_train[:10].detach().cpu())

    print("\nFirst 20 y_train values:")
    print(y_train[:20].detach().cpu())

    if task.num_classes is not None:
        class_counts = torch.bincount(
            y_full.long().detach().cpu(),
            minlength=int(task.num_classes),
        )

        print("\nTarget class counts:")
        print(class_counts)

    missing_rate = float(
        torch.isnan(X_full)
        .float()
        .mean()
        .item()
    )

    print(
        f"\nObserved missing rate: "
        f"{missing_rate:.4f}"
    )

    print("\nTask edge probability:")
    print(
        float(
            info["task_edge_prob"].item()
        )
    )

    print_root_prior_summary(
        info
    )

    print_feature_summary(
        X_full=X_full.detach().cpu(),
        info=info,
    )


def benchmark_multiple_tasks(
    device: torch.device,
    num_tasks: int = 100,
) -> None:
    print(
        "\n\n========== Benchmark multiple tasks =========="
    )

    elapsed_times: list[float] = []

    categorical_counts: list[int] = []
    total_feature_counts: list[int] = []
    cardinalities: list[int] = []
    categorical_scores: list[float] = []

    root_prior_counter: Counter[str] = Counter()
    mixture_component_counter: Counter[int] = Counter()

    tasks_with_mixture_root = 0
    categorical_with_mixture_root = 0
    total_features_with_mixture_root = 0

    tasks_without_mixture_root = 0
    categorical_without_mixture_root = 0
    total_features_without_mixture_root = 0

    projection_feature_counter: Counter[int] = Counter()
    projection_categorical_counter: Counter[int] = Counter()
    projection_cardinality_counter: dict[int, Counter[int]] = {}

    for seed in range(num_tasks):
        start = time.perf_counter()

        task = build_task(
            device=device,
            dag_seed=seed,
            aleatoric_seed=10_000 + seed,
            x_seed=20_000 + seed,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        elapsed_times.append(
            elapsed
        )

        _, _, _, _, info = get_task_outputs(
            task
        )

        feature_type = info["feature_type"]
        cardinality = info["cardinality"]

        cluster_score = info[
            "categorical_cluster_score"
        ]

        projection_dim = info[
            "categorical_projection_dim"
        ]

        root_prior_types = info[
            "root_prior_types"
        ]

        root_mixture_components = (
            info["root_mixture_components"]
            .detach()
            .cpu()
            .tolist()
        )

        root_prior_counter.update(
            root_prior_types
        )

        mixture_component_counter.update(
            value
            for value in root_mixture_components
            if value > 0
        )

        has_mixture_root = (
            "mixture"
            in root_prior_types
        )

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

        total_feature_counts.append(
            n_features
        )

        categorical_counts.append(
            n_categorical
        )

        if has_mixture_root:
            tasks_with_mixture_root += 1
            categorical_with_mixture_root += n_categorical
            total_features_with_mixture_root += n_features
        else:
            tasks_without_mixture_root += 1
            categorical_without_mixture_root += n_categorical
            total_features_without_mixture_root += n_features

        selected_cardinalities = (
            cardinality[categorical_mask]
            .detach()
            .cpu()
            .tolist()
        )

        selected_scores = (
            cluster_score[categorical_mask]
            .detach()
            .cpu()
            .tolist()
        )

        cardinalities.extend(
            int(value)
            for value in selected_cardinalities
        )

        categorical_scores.extend(
            float(value)
            for value in selected_scores
        )

        projection_dims = (
            projection_dim
            .detach()
            .cpu()
            .tolist()
        )

        feature_types = (
            feature_type
            .detach()
            .cpu()
            .tolist()
        )

        feature_cardinalities = (
            cardinality
            .detach()
            .cpu()
            .tolist()
        )

        for projection_d, feature_t, feature_k in zip(
            projection_dims,
            feature_types,
            feature_cardinalities,
        ):
            projection_d = int(projection_d)
            feature_t = int(feature_t)
            feature_k = int(feature_k)

            # 该投影维度一共生成了多少 feature
            projection_feature_counter[
                projection_d
            ] += 1

            if (
                feature_t
                == MixedLatentSCMTask.CATEGORICAL
            ):
                # 该投影维度中有多少 feature 被判为 categorical
                projection_categorical_counter[
                    projection_d
                ] += 1

                # 该投影维度下最终 K 的分布
                if (
                    projection_d
                    not in projection_cardinality_counter
                ):
                    projection_cardinality_counter[
                        projection_d
                    ] = Counter()

                projection_cardinality_counter[
                    projection_d
                ][
                    feature_k
                ] += 1

        print(
            f"Task {seed:03d}: "
            f"time={elapsed:.4f}s, "
            f"roots={root_prior_types}, "
            f"M={root_mixture_components}, "
            f"categorical={n_categorical}/{n_features}, "
            f"K={selected_cardinalities}"
        )

    times = torch.tensor(
        elapsed_times,
        dtype=torch.float64,
    )

    total_categorical = sum(
        categorical_counts
    )

    total_features = sum(
        total_feature_counts
    )

    categorical_ratio = (
        total_categorical
        / total_features
        if total_features > 0
        else 0.0
    )

    tasks_with_categorical = sum(
        count > 0
        for count in categorical_counts
    )

    print(
        "\n========== Benchmark result =========="
    )

    print(f"Tasks:       {num_tasks}")
    print(f"Mean time:   {times.mean().item():.4f} s")
    print(f"Median time: {times.median().item():.4f} s")
    print(f"Min time:    {times.min().item():.4f} s")
    print(f"Max time:    {times.max().item():.4f} s")

    print(
        "Mean categorical features per task: "
        f"{total_categorical / num_tasks:.2f}"
    )

    print(
        "Overall categorical feature ratio: "
        f"{categorical_ratio:.2%}"
    )

    print(
        "Tasks with at least one categorical feature: "
        f"{tasks_with_categorical}/{num_tasks} "
        f"({tasks_with_categorical / num_tasks:.2%})"
    )

    print(
        "Categorical counts per task:",
        categorical_counts,
    )

    print(
        "\nRoot prior counts:",
        dict(root_prior_counter),
    )

    print(
        "Mixture-root latent component counts:",
        dict(sorted(mixture_component_counter.items())),
    )

    if total_features_with_mixture_root > 0:
        ratio_with_mixture = (
            categorical_with_mixture_root
            / total_features_with_mixture_root
        )

        print(
            "Categorical ratio in tasks with mixture roots: "
            f"{ratio_with_mixture:.2%} "
            f"({tasks_with_mixture_root} tasks)"
        )

    if total_features_without_mixture_root > 0:
        ratio_without_mixture = (
            categorical_without_mixture_root
            / total_features_without_mixture_root
        )

        print(
            "Categorical ratio in tasks without mixture roots: "
            f"{ratio_without_mixture:.2%} "
            f"({tasks_without_mixture_root} tasks)"
        )

    if cardinalities:
        cardinality_tensor = torch.tensor(
            cardinalities,
            dtype=torch.long,
        )

        unique_k, counts_k = torch.unique(
            cardinality_tensor,
            return_counts=True,
        )

        cardinality_distribution = {
            int(k.item()): int(count.item())
            for k, count in zip(
                unique_k,
                counts_k,
            )
        }

        print(
            "Observed categorical cardinality distribution:",
            cardinality_distribution,
        )

    else:
        print(
            "No categorical features were generated."
        )

    if categorical_scores:
        scores = torch.tensor(
            categorical_scores,
            dtype=torch.float64,
        )

        print(
            "Categorical cluster scores: "
            f"mean={scores.mean().item():.3f}, "
            f"min={scores.min().item():.3f}, "
            f"max={scores.max().item():.3f}"
        )

        print(
            "\nProjection-dimension statistics:"
        )

        for projection_d in sorted(
            projection_feature_counter
        ):
            n_projection_features = (
                projection_feature_counter[
                    projection_d
                ]
            )

            n_projection_categorical = (
                projection_categorical_counter[
                    projection_d
                ]
            )

            categorical_rate = (
                n_projection_categorical
                / n_projection_features
                if n_projection_features > 0
                else 0.0
            )

            k_distribution = dict(
                sorted(
                    projection_cardinality_counter
                    .get(
                        projection_d,
                        Counter(),
                    )
                    .items()
                )
            )

            print(
                f"projection_dim={projection_d}: "
                f"features={n_projection_features}, "
                f"categorical={n_projection_categorical}, "
                f"categorical_rate={categorical_rate:.2%}, "
                f"K_distribution={k_distribution}"
            )


if __name__ == "__main__":
    device = torch.device("cpu")

    generate_one_task(
        device=device,
    )

    # 10 tasks is only a smoke test. Use at least 100 for prior evaluation.
    benchmark_multiple_tasks(
        device=device,
        num_tasks=100,
    )

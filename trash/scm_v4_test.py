# test_scm_generator.py

import time

import torch

from Trash.scm_task_v4 import MixedLatentSCMTask


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

    n_features = int(feature_type.numel())

    categorical_mask = (
        feature_type
        == MixedLatentSCMTask.CATEGORICAL
    )

    n_categorical = int(
        categorical_mask.sum().item()
    )

    n_continuous = n_features - n_categorical

    print("\n========== Feature summary ==========")
    print(f"Number of features:    {n_features}")
    print(f"Continuous features:   {n_continuous}")
    print(f"Categorical features:  {n_categorical}")

    for col in range(n_features):
        is_categorical = bool(
            categorical_mask[col].item()
        )

        feature_name = f"x_{col + 1}"
        score = float(cluster_score[col].item())

        column = X_full[:, col]
        valid = column[~torch.isnan(column)]

        if not is_categorical:
            if valid.numel() == 0:
                print(
                    f"{feature_name}: continuous, "
                    "all values are missing"
                )
                continue

            print(
                f"{feature_name}: continuous, "
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

            # centers shape is now [K, latent_dim].
            center_norms = torch.linalg.vector_norm(
                centers,
                dim=1,
            )

            print(
                f"{feature_name}: categorical, "
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
    Keep generation parameters in one place so the one-task test and benchmark
    use exactly the same configuration.
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
        latent_dim=8,

        latent_noise_scale=0.05,
        observation_noise_scale=0.05,

        # Latent-space K-means categorical detection.
        max_cardinality=6,
        min_samples_per_category=8,
        min_component_weight=0.05,

        # Start conservatively. Lower these if categorical features are too rare.
        min_cluster_separation=1.5,
        min_cluster_score=0.40,

        num_kmeans_restarts=3,
        max_kmeans_iterations=50,
        kmeans_tolerance=1e-4,

        dag_seed=dag_seed,
        aleatoric_seed=aleatoric_seed,
        x_seed=x_seed,

        device=device,
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

    elapsed = time.perf_counter() - start

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
        torch.isnan(X_full).float().mean().item()
    )

    print(f"\nObserved missing rate: {missing_rate:.4f}")

    print("\nTask edge probability:")
    print(float(info["task_edge_prob"].item()))

    print_feature_summary(
        X_full=X_full.detach().cpu(),
        info=info,
    )


def benchmark_multiple_tasks(
    device: torch.device,
    num_tasks: int = 10,
) -> None:
    print("\n\n========== Benchmark multiple tasks ==========")

    elapsed_times: list[float] = []
    categorical_counts: list[int] = []
    total_feature_counts: list[int] = []
    cardinalities: list[int] = []
    categorical_scores: list[float] = []

    for seed in range(num_tasks):
        start = time.perf_counter()

        task = build_task(
            device=device,
            dag_seed=seed,
            aleatoric_seed=10_000 + seed,
            x_seed=20_000 + seed,
        )

        elapsed = time.perf_counter() - start
        elapsed_times.append(elapsed)

        _, _, _, _, info = get_task_outputs(task)

        feature_type = info["feature_type"]
        cardinality = info["cardinality"]
        cluster_score = info[
            "categorical_cluster_score"
        ]

        categorical_mask = (
            feature_type
            == MixedLatentSCMTask.CATEGORICAL
        )

        n_features = int(feature_type.numel())
        n_categorical = int(
            categorical_mask.sum().item()
        )

        total_feature_counts.append(n_features)
        categorical_counts.append(n_categorical)

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

        print(
            f"Task {seed:02d}: "
            f"time={elapsed:.4f}s, "
            f"categorical={n_categorical}/{n_features}, "
            f"K={selected_cardinalities}, "
            f"scores="
            f"{[round(value, 3) for value in selected_scores]}"
        )

    times = torch.tensor(
        elapsed_times,
        dtype=torch.float64,
    )

    total_categorical = sum(categorical_counts)
    total_features = sum(total_feature_counts)

    categorical_ratio = (
        total_categorical / total_features
        if total_features > 0
        else 0.0
    )

    tasks_with_categorical = sum(
        count > 0
        for count in categorical_counts
    )

    print("\n========== Benchmark result ==========")
    print(f"Tasks:       {num_tasks}")
    print(f"Mean time:   {times.mean().item():.4f} s")
    print(f"Median time: {times.median().item():.4f} s")
    print(f"Min time:    {times.min().item():.4f} s")
    print(f"Max time:    {times.max().item():.4f} s")

    print(
        "Mean categorical features per task: "
        f"{sum(categorical_counts) / num_tasks:.2f}"
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
            for k, count in zip(unique_k, counts_k)
        }

        print(
            "Categorical cardinality distribution:",
            cardinality_distribution,
        )
    else:
        print("No categorical features were generated.")

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


if __name__ == "__main__":
    device = torch.device("cpu")

    generate_one_task(
        device=device,
    )

    benchmark_multiple_tasks(
        device=device,
        num_tasks=10,
    )
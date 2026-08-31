# test_scm_generator.py

import time

import torch

from Trash.scm_task_v3 import MixedLatentSCMTask


def get_task_outputs(task):
    """
    兼容 GenerateTask 可能使用的两种存储方式。

    常见情况：
        task.X_train
        task.y_train
        task.X_test
        task.y_test
        task.info
    """
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
    cluster_score = info["categorical_cluster_score"].detach().cpu()

    n_features = int(feature_type.numel())
    n_categorical = int(
        (feature_type == MixedLatentSCMTask.CATEGORICAL).sum().item()
    )
    n_continuous = n_features - n_categorical

    print("\n========== Feature summary ==========")
    print(f"Number of features:    {n_features}")
    print(f"Continuous features:   {n_continuous}")
    print(f"Categorical features:  {n_categorical}")

    for col in range(n_features):
        is_cat = (
            int(feature_type[col].item())
            == MixedLatentSCMTask.CATEGORICAL
        )

        feature_name = f"x_{col + 1}"
        score = float(cluster_score[col].item())

        if not is_cat:
            column = X_full[:, col]
            valid = column[~torch.isnan(column)]

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
            column = X_full[:, col]
            valid = column[~torch.isnan(column)].long()

            counts = torch.bincount(
                valid,
                minlength=K,
            )

            centers = info["feature_cluster_centers"][col]
            centers = centers.detach().cpu()

            print(
                f"{feature_name}: categorical, "
                f"K={K}, "
                f"counts={counts.tolist()}, "
                f"centers={centers.tolist()}, "
                f"cluster_score={score:.3f}"
            )


def generate_one_task(
    device: torch.device,
) -> None:
    print("========== Generate one task ==========")
    print(f"Device: {device}")

    start = time.perf_counter()

    task = MixedLatentSCMTask(
        num_classes=3,

        # 固定 n 和 d，方便测速
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

        # 先使用相对轻量的 GMM 配置
        max_cardinality=6,
        min_samples_per_category=8,
        min_bic_improvement=8.0,
        min_cluster_separation=1,
        min_component_weight=0.03,
        num_em_restarts=3,
        max_em_iterations=50,
        em_tolerance=1e-4,
        variance_floor=1e-3,

        dag_seed=1,
        aleatoric_seed=2,
        x_seed=3,

        device=device,
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

    elapsed_times = []
    categorical_counts = []
    cardinalities = []

    for seed in range(num_tasks):
        start = time.perf_counter()

        task = MixedLatentSCMTask(
            num_classes=3,

            n_min=500,
            n_max=500,
            d_min=8,
            d_max=16,

            num_roots=3,
            num_layers=5,
            max_nodes_per_layer=8,
            latent_dim=8,

            max_cardinality=6,
            min_samples_per_category=8,
            min_bic_improvement=8.0,
            min_cluster_separation=1,
            min_component_weight=0.03,
            num_em_restarts=3,
            max_em_iterations=50,
            em_tolerance=1e-4,
            variance_floor=1e-3,

            dag_seed=seed,
            aleatoric_seed=10_000 + seed,
            x_seed=20_000 + seed,

            device=device,
        )

        elapsed = time.perf_counter() - start
        elapsed_times.append(elapsed)

        _, _, _, _, info = get_task_outputs(task)

        feature_type = info["feature_type"]
        cardinality = info["cardinality"]

        categorical_mask = (
            feature_type
            == MixedLatentSCMTask.CATEGORICAL
        )

        n_categorical = int(
            categorical_mask.sum().item()
        )
        categorical_counts.append(n_categorical)

        selected_cardinalities = cardinality[
            categorical_mask
        ].detach().cpu().tolist()

        cardinalities.extend(selected_cardinalities)

        print(
            f"Task {seed:02d}: "
            f"time={elapsed:.4f}s, "
            f"categorical={n_categorical}/{task.n_features}, "
            f"K={selected_cardinalities}"
        )

    times = torch.tensor(
        elapsed_times,
        dtype=torch.float64,
    )

    print("\n========== Benchmark result ==========")
    print(f"Tasks:       {num_tasks}")
    print(f"Mean time:   {times.mean().item():.4f} s")
    print(f"Median time: {times.median().item():.4f} s")
    print(f"Min time:    {times.min().item():.4f} s")
    print(f"Max time:    {times.max().item():.4f} s")

    mean_categorical = (
        sum(categorical_counts)
        / len(categorical_counts)
    )

    print(
        "Mean categorical features per task: "
        f"{mean_categorical:.2f}"
    )
    print(
        "Categorical counts per task:",
        categorical_counts,
    )

    if cardinalities:
        print(
            "Observed categorical cardinalities:",
            cardinalities,
        )
    else:
        print(
            "No categorical features were generated."
        )


if __name__ == "__main__":
    # 这种大量小规模 EM 计算，建议先用 CPU 测试。
    device = torch.device("cpu")

    generate_one_task(device=device)

    benchmark_multiple_tasks(
        device=device,
        num_tasks=10,
    )
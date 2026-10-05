import torch

from .scm_task_v2.calibration.difficulty import compute_difficulty_metrics


def make_linear_data(n=512, d=10, noise_std=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)

    X = torch.randn(n, d, generator=g)

    noise = noise_std * torch.randn(n, generator=g)

    y = (
        2.0 * X[:, 0]
        - 3.0 * X[:, 1]
        + 1.5 * X[:, 2]
        + noise
    )

    return X, y


def make_interaction_data(n=512, d=10, noise_std=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)

    X = torch.randn(n, d, generator=g)

    noise = noise_std * torch.randn(n, generator=g)

    y = (
        X[:, 0] * X[:, 1]
        + X[:, 2] * X[:, 3]
        + noise
    )

    return X, y


def make_nonlinear_data(n=512, d=10, noise_std=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)

    X = torch.randn(n, d, generator=g)

    noise = noise_std * torch.randn(n, generator=g)

    y = (
        X[:, 0] * X[:, 1]
        + torch.sin(2.0 * X[:, 2])
        + 0.5 * X[:, 3].square()
        + noise
    )

    return X, y


def split_data(X, y, test_frac=0.15):
    n = X.shape[0]
    n_test = max(1, int(n * test_frac))

    X_train = X[:-n_test]
    y_train = y[:-n_test]

    X_test = X[-n_test:]
    y_test = y[-n_test:]

    return X_train, y_train, X_test, y_test


def evaluate_dataset(
    generator_fn,
    num_tasks=100,
    n=512,
    d=10,
    noise_std=0.1,
    epochs=100,
    mlp_hidden_dim=64,
):
    records = []

    for seed in range(num_tasks):
        X, y = generator_fn(
            n=n,
            d=d,
            noise_std=noise_std,
            seed=seed,
        )

        X_train, y_train, X_test, y_test = split_data(X, y)

        metrics = compute_difficulty_metrics(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            num_classes=None,
            epochs=epochs,
            mlp_hidden_dim=mlp_hidden_dim,
            probe_seed=100000 + seed,
            feature_type=None,
            cardinality=None,
        )

        records.append(metrics)

    keys = records[0].keys()

    summary = {
        key: sum(record[key] for record in records) / len(records)
        for key in keys
    }

    return summary


def print_summary(name, summary):
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    for key, value in summary.items():
        print(f"{key:20s}: {value:.6f}")


def main():
    num_tasks = 100
    n = 512
    d = 10
    noise_std = 0.1

    # Keep these identical to the current calibration probe.
    epochs = 100
    mlp_hidden_dim = 64

    linear_summary = evaluate_dataset(
        make_linear_data,
        num_tasks=num_tasks,
        n=n,
        d=d,
        noise_std=noise_std,
        epochs=epochs,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    interaction_summary = evaluate_dataset(
        make_interaction_data,
        num_tasks=num_tasks,
        n=n,
        d=d,
        noise_std=noise_std,
        epochs=epochs,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    nonlinear_summary = evaluate_dataset(
        make_nonlinear_data,
        num_tasks=num_tasks,
        n=n,
        d=d,
        noise_std=noise_std,
        epochs=epochs,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    print_summary("LINEAR DATA", linear_summary)
    print_summary("INTERACTION DATA", interaction_summary)
    print_summary("NONLINEAR DATA", nonlinear_summary)


if __name__ == "__main__":
    main()
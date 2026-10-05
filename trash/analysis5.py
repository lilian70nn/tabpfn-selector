import random
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, r2_score

from ..src.data.scm_task_v2.task import SCMTask


PRIOR = {
    "n_min": 400,
    "n_max": 512,
    "d_min": 8,
    "d_max": 16,
    "test_frac": 0.15,
    "p_missing": 0.05,
    "num_roots": 5,
    "num_layers": 3,
    "final_width": 1,
    "latent_noise_scale": (0.0, 0.03),
    "scale_min": 0.25,
    "scale_max": 4.0,
    "categorical_cardinalities": (2, 3, 4, 5, 6),
    "categorical_cardinality_probs": (0.40, 0.30, 0.18, 0.08, 0.04),
    "min_samples_per_category": 8,
    "min_component_weight": 0.05,
    "observation_noise_scale": 0.03,
    "connection_probs": ((0.20, 0.35), (0.45, 0.65)),
    "source_prior_probs": (0.55, 0.20, 0.15, 0.10),
    "arity_probs": (2.25, 3.0, 3.5),
    "unary_op_probs": (0.75, 1.0, 2.25, 2.25, 1.25, 1.0, 1.5),
    "binary_op_probs": (2.5, 2.0, 3.0, 1.5),
    "ternary_op_probs": (3.0, 1.0, 1.0, 3.0),
    "observation_type_probs": (7.0, 1.5, 1.5),
}


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _local_label_consistency(X, y, k=10):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    X_scaled = (X - mean) / std

    n = X_scaled.shape[0]
    if n <= 1:
        return {
            "raw_local_agreement": np.nan,
            "random_local_agreement": np.nan,
            "normalized_local_consistency": np.nan,
        }

    k_actual = min(k, n - 1)
    diff = X_scaled[:, None, :] - X_scaled[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    np.fill_diagonal(distances, np.inf)

    neighbor_indices = np.argpartition(distances, kth=k_actual - 1, axis=1)[:, :k_actual]
    neighbor_labels = y[neighbor_indices]
    center_labels = y[:, None]

    same_label = neighbor_labels == center_labels
    per_sample_agreement = same_label.mean(axis=1)
    raw_agreement = float(per_sample_agreement.mean())

    counts = np.bincount(y)
    proportions = counts / counts.sum()
    random_agreement = float(np.sum(proportions ** 2))

    denominator = 1.0 - random_agreement
    if denominator <= 1e-12:
        normalized_consistency = np.nan
    else:
        normalized_consistency = (raw_agreement - random_agreement) / denominator

    return {
        "raw_local_agreement": raw_agreement,
        "random_local_agreement": random_agreement,
        "normalized_local_consistency": float(normalized_consistency),
    }


class _MLP(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, d_out),
        )

    def forward(self, x):
        return self.net(x)


def _latent_mlp_score(X_train, X_test, y_train, y_test, num_classes, epochs=500, seed=0):
    torch.manual_seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.int64).reshape(-1)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_train, dtype=torch.long, device=device)

    model = _MLP(Xtr.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(Xtr_t)
        loss = torch.nn.functional.cross_entropy(logits, ytr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(dim=1).cpu().numpy()

    score = float(balanced_accuracy_score(y_test, pred))

    del model, optimizer, Xtr_t, Xte_t, ytr_t

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return score


def _target_latent_r2(X_train, X_test, z_train, z_test, epochs=500, seed=0):
    torch.manual_seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    z_train = np.asarray(z_train, dtype=np.float32).reshape(-1)
    z_test = np.asarray(z_test, dtype=np.float32).reshape(-1)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)

    z_mean = float(z_train.mean())
    z_std = float(z_train.std())
    if z_std < 1e-12:
        z_std = 1.0

    ztr_scaled = ((z_train - z_mean) / z_std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ztr_t = torch.tensor(ztr_scaled, dtype=torch.float32, device=device)

    model = _MLP(Xtr.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(Xtr_t).squeeze(1)
        loss = torch.nn.functional.mse_loss(pred, ztr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_scaled = model(Xte_t).squeeze(1).cpu().numpy()

    pred = pred_scaled * z_std + z_mean
    score = float(r2_score(z_test, pred))

    del model, optimizer, Xtr_t, Xte_t, ztr_t

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return score


def _evaluate_task(task, k=10, seed=0):
    info = task.info

    latent_X_train = _to_numpy(info["selected_latent_X_train"]).astype(np.float32)
    latent_X_test = _to_numpy(info["selected_latent_X_test"]).astype(np.float32)

    target_latent_train = _to_numpy(info["target_latent_train"]).reshape(-1).astype(np.float32)
    target_latent_test = _to_numpy(info["target_latent_test"]).reshape(-1).astype(np.float32)

    y_train_np = _to_numpy(task.y_train).reshape(-1).astype(np.int64)
    y_test_np = _to_numpy(task.y_test).reshape(-1).astype(np.int64)

    latent_X_all = np.concatenate([latent_X_train, latent_X_test], axis=0)
    y_all = np.concatenate([y_train_np, y_test_np], axis=0)

    consistency = _local_label_consistency(latent_X_all, y_all, k=k)

    latent_mlp = _latent_mlp_score(
        latent_X_train,
        latent_X_test,
        y_train_np,
        y_test_np,
        num_classes=task.num_classes,
        epochs=500,
        seed=seed,
    )

    target_latent_r2 = _target_latent_r2(
        latent_X_train,
        latent_X_test,
        target_latent_train,
        target_latent_test,
        epochs=500,
        seed=seed,
    )

    counts = np.bincount(y_all, minlength=task.num_classes)
    proportions = counts / counts.sum()

    result = {
        "num_classes": task.num_classes,
        "latent_mlp": latent_mlp,
        "target_latent_r2": target_latent_r2,
        "class_min_fraction": float(proportions.min()),
        "class_max_fraction": float(proportions.max()),
        "class_min_max_ratio": float(counts.min() / max(counts.max(), 1)),
    }

    result.update(consistency)
    return result


def run_analysis(prior, class_counts=(2, 3, 4), n_tasks=100, k=10, base_seed=0, output_csv="local_consistency.csv"):
    rows = []

    for num_classes in class_counts:
        print()
        print("=" * 60)
        print(f"{num_classes}-CLASS")
        print("=" * 60)

        valid = 0

        for idx in range(n_tasks):
            rng = random.Random(base_seed + idx)

            dag_seed = rng.randrange(2**31)
            x_seed = rng.randrange(2**31)
            aleatoric_seed = rng.randrange(2**31)

            try:
                task = SCMTask(
                    **prior,
                    num_classes=num_classes,
                    dag_seed=dag_seed,
                    x_seed=x_seed,
                    aleatoric_seed=aleatoric_seed,
                )

                info = task.info
                if not isinstance(info, dict):
                    print(f"\r{idx + 1}/{n_tasks}, valid={valid}", end="", flush=True)
                    continue

                if "is_valid" in info:
                    value = info["is_valid"].item() if hasattr(info["is_valid"], "item") else info["is_valid"]
                    if not bool(value):
                        print(f"\r{idx + 1}/{n_tasks}, valid={valid}", end="", flush=True)
                        continue

                y_train = _to_numpy(task.y_train).reshape(-1).astype(np.int64)
                y_test = _to_numpy(task.y_test).reshape(-1).astype(np.int64)
                expected = np.arange(num_classes)

                if not np.array_equal(np.unique(y_train), expected):
                    print(f"\r{idx + 1}/{n_tasks}, valid={valid}", end="", flush=True)
                    continue

                if not np.array_equal(np.unique(y_test), expected):
                    print(f"\r{idx + 1}/{n_tasks}, valid={valid}", end="", flush=True)
                    continue

                result = _evaluate_task(task, k=k, seed=base_seed + idx)
                rows.append(result)
                valid += 1

            except Exception as exc:
                print()
                print(f"task {idx} failed: {type(exc).__name__}: {exc}")

            if (idx + 1) % 10 == 0:
                print(f"\r{idx + 1}/{n_tasks}, valid={valid}", flush=True)

        print()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    if len(df) == 0:
        print("No valid tasks.")
        return df

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary_columns = [
        "latent_mlp",
        "target_latent_r2",
        "raw_local_agreement",
        "random_local_agreement",
        "normalized_local_consistency",
        "class_min_fraction",
        "class_max_fraction",
        "class_min_max_ratio",
    ]

    summary = df.groupby("num_classes")[summary_columns].mean()

    print()
    print("Mean diagnostics:")
    print(summary.to_string())

    print()
    print("=" * 60)
    print("MEDIAN DIAGNOSTICS")
    print("=" * 60)

    median_summary = df.groupby("num_classes")[summary_columns].median()
    print(median_summary.to_string())

    print()
    print("=" * 60)
    print("CORRELATION WITH LATENT MLP")
    print("=" * 60)

    for num_classes in sorted(df["num_classes"].unique()):
        subset = df[df["num_classes"] == num_classes]

        if len(subset) < 2:
            print(f"{num_classes}-class: not enough tasks")
            continue

        consistency_corr = subset[["latent_mlp", "normalized_local_consistency"]].corr().iloc[0, 1]
        r2_corr = subset[["latent_mlp", "target_latent_r2"]].corr().iloc[0, 1]

        print(
            f"{num_classes}-class | "
            f"consistency vs MLP={consistency_corr:.4f} | "
            f"target latent R2 vs MLP={r2_corr:.4f}"
        )

    print()
    print("=" * 60)
    print("KEY RESULT")
    print("=" * 60)

    for num_classes in sorted(df["num_classes"].unique()):
        subset = df[df["num_classes"] == num_classes]

        mean_consistency = subset["normalized_local_consistency"].mean()
        median_consistency = subset["normalized_local_consistency"].median()
        raw_agreement = subset["raw_local_agreement"].mean()
        random_agreement = subset["random_local_agreement"].mean()
        mean_mlp = subset["latent_mlp"].mean()
        mean_r2 = subset["target_latent_r2"].mean()
        median_r2 = subset["target_latent_r2"].median()

        print(
            f"{num_classes}-class | "
            f"consistency mean={mean_consistency:.4f}, median={median_consistency:.4f} | "
            f"raw={raw_agreement:.4f}, random={random_agreement:.4f} | "
            f"target latent R2 mean={mean_r2:.4f}, median={median_r2:.4f} | "
            f"latent MLP={mean_mlp:.4f}"
        )

    print()
    print(f"Saved results to: {output_csv}")

    return df


if __name__ == "__main__":
    run_analysis(
        prior=PRIOR,
        class_counts=(2, 3, 4),
        n_tasks=300,
        k=10,
        base_seed=0,
        output_csv="local_consistency.csv",
    )
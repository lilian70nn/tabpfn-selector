import random
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from src.data.scm_task_v2.task import SCMTask


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_data(task):
    X_train = _to_numpy(task.X_train)
    X_test = _to_numpy(task.X_test)
    y_train = _to_numpy(task.y_train).reshape(-1).astype(np.int64)
    y_test = _to_numpy(task.y_test).reshape(-1).astype(np.int64)
    return X_train, X_test, y_train, y_test


def _get_feature_type(task):
    return _to_numpy(task.info["feature_type"]).astype(np.int64)


def _is_table_valid(task, num_classes=4):
    info = task.info
    if not isinstance(info, dict):
        return False
    if "is_valid" in info:
        value = info["is_valid"].item() if hasattr(info["is_valid"], "item") else info["is_valid"]
        if not bool(value):
            return False
    X_train, X_test, y_train, y_test = _get_data(task)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] == 0:
        return False
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        return False
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        return False
    expected = np.arange(num_classes)
    if not np.array_equal(np.unique(y_train), expected):
        return False
    if not np.array_equal(np.unique(y_test), expected):
        return False
    return True


def _build_preprocessor(feature_type):
    continuous = np.where(feature_type == 0)[0]
    categorical = np.where(feature_type != 0)[0]
    transformers = []
    if len(continuous) > 0:
        cont = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        transformers.append(("continuous", cont, continuous))
    if len(categorical) > 0:
        cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
        transformers.append(("categorical", cat, categorical))
    return ColumnTransformer(transformers, remainder="drop")


def _fit_logistic_score(X_train, X_test, y_train, y_test, feature_type, seed):
    preprocessor = _build_preprocessor(feature_type)
    model = Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000, random_state=seed))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(balanced_accuracy_score(y_test, pred))


class _MLP(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, d_out))

    def forward(self, x):
        return self.net(x)


def _fit_mlp_score(X_train, X_test, y_train, y_test, feature_type, num_classes=4, epochs=500, seed=0):
    torch.manual_seed(seed)
    preprocessor = _build_preprocessor(feature_type)
    Xtr = preprocessor.fit_transform(X_train).astype(np.float32)
    Xte = preprocessor.transform(X_test).astype(np.float32)
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
    return float(balanced_accuracy_score(y_test, pred))


def _fit_rf_score(X_train, X_test, y_train, y_test, feature_type, seed):
    preprocessor = _build_preprocessor(feature_type)
    estimator = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=1)
    model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(balanced_accuracy_score(y_test, pred))


def _target_latent_diagnostics(target_latent, y, num_classes=4):
    rows = {}
    class_means = []
    class_mins = []
    class_maxs = []

    for c in range(num_classes):
        values = target_latent[y == c]
        if len(values) == 0:
            rows[f"target_latent_class_{c}_mean"] = np.nan
            rows[f"target_latent_class_{c}_std"] = np.nan
            rows[f"target_latent_class_{c}_min"] = np.nan
            rows[f"target_latent_class_{c}_max"] = np.nan
            class_means.append(np.nan)
            class_mins.append(np.nan)
            class_maxs.append(np.nan)
            continue

        mean = float(values.mean())
        std = float(values.std())
        minimum = float(values.min())
        maximum = float(values.max())

        rows[f"target_latent_class_{c}_mean"] = mean
        rows[f"target_latent_class_{c}_std"] = std
        rows[f"target_latent_class_{c}_min"] = minimum
        rows[f"target_latent_class_{c}_max"] = maximum

        class_means.append(mean)
        class_mins.append(minimum)
        class_maxs.append(maximum)

    ordered = bool(np.all(np.diff(class_means) > 0) or np.all(np.diff(class_means) < 0))
    rows["target_class_means_ordered"] = float(ordered)

    sorted_classes = np.argsort(class_means)
    overlaps = []

    for i in range(num_classes - 1):
        left = sorted_classes[i]
        right = sorted_classes[i + 1]
        left_min = class_mins[left]
        left_max = class_maxs[left]
        right_min = class_mins[right]
        right_max = class_maxs[right]

        overlap = max(0.0, min(left_max, right_max) - max(left_min, right_min))
        union = max(left_max, right_max) - min(left_min, right_min)

        if union <= 1e-12:
            overlap_ratio = 0.0
        else:
            overlap_ratio = overlap / union

        rows[f"target_adjacent_overlap_{i}"] = float(overlap_ratio)
        overlaps.append(overlap_ratio)

    rows["target_mean_adjacent_overlap"] = float(np.mean(overlaps))
    rows["target_max_adjacent_overlap"] = float(np.max(overlaps))

    overall_std = float(np.std(target_latent))
    if overall_std <= 1e-12:
        separation = 0.0
    else:
        separation = float(np.std(class_means) / overall_std)

    rows["target_class_mean_separation"] = separation
    return rows


def _evaluate_task(task, seed=0, mlp_epochs=500):
    X_train, X_test, y_train, y_test = _get_data(task)
    info = task.info
    feature_type = _get_feature_type(task)

    latent_X_train = _to_numpy(info["selected_latent_X_train"])
    latent_X_test = _to_numpy(info["selected_latent_X_test"])
    latent_feature_type = np.zeros(latent_X_train.shape[1], dtype=np.int64)

    target_latent_train = _to_numpy(info["target_latent_train"]).reshape(-1)
    target_latent_test = _to_numpy(info["target_latent_test"]).reshape(-1)
    target_latent = np.concatenate([target_latent_train, target_latent_test])
    y_all = np.concatenate([y_train, y_test])

    counts = np.bincount(y_all, minlength=4)
    proportions = counts / counts.sum()

    categorical_mask = feature_type != 0
    quality = _to_numpy(info["feature_observation_quality"]).astype(np.float64)
    retention = _to_numpy(info["feature_retention"]).astype(np.float64)
    feature_importance = _to_numpy(info["feature_importance"]).astype(np.float64)

    cat_ratio = float(categorical_mask.mean())
    categorical_quality = float(quality[categorical_mask].mean()) if categorical_mask.any() else np.nan
    categorical_retention = float(retention[categorical_mask].mean()) if categorical_mask.any() else np.nan

    observed_logistic = _fit_logistic_score(X_train, X_test, y_train, y_test, feature_type, seed)
    observed_mlp = _fit_mlp_score(X_train, X_test, y_train, y_test, feature_type, num_classes=4, epochs=mlp_epochs, seed=seed)
    observed_rf = _fit_rf_score(X_train, X_test, y_train, y_test, feature_type, seed)

    latent_logistic = _fit_logistic_score(latent_X_train, latent_X_test, y_train, y_test, latent_feature_type, seed)
    latent_mlp = _fit_mlp_score(latent_X_train, latent_X_test, y_train, y_test, latent_feature_type, num_classes=4, epochs=mlp_epochs, seed=seed)
    latent_rf = _fit_rf_score(latent_X_train, latent_X_test, y_train, y_test, latent_feature_type, seed)

    result = {
        "class_0_count": int(counts[0]),
        "class_1_count": int(counts[1]),
        "class_2_count": int(counts[2]),
        "class_3_count": int(counts[3]),
        "class_min_fraction": float(proportions.min()),
        "class_max_fraction": float(proportions.max()),
        "class_min_max_ratio": float(counts.min() / max(counts.max(), 1)),
        "cat_ratio": cat_ratio,
        "categorical_quality": categorical_quality,
        "categorical_retention": categorical_retention,
        "mean_retention": float(retention.mean()),
        "min_retention": float(retention.min()),
        "importance_weighted_retention": float(np.sum(feature_importance * retention)),
        "max_importance": float(feature_importance.max()),
        "top3_importance": float(np.sort(feature_importance)[-min(3, len(feature_importance)):].sum()),
        "observed_logistic": observed_logistic,
        "observed_mlp": observed_mlp,
        "observed_rf": observed_rf,
        "latent_logistic": latent_logistic,
        "latent_mlp": latent_mlp,
        "latent_rf": latent_rf,
        "logistic_observation_loss": latent_logistic - observed_logistic,
        "mlp_observation_loss": latent_mlp - observed_mlp,
        "rf_observation_loss": latent_rf - observed_rf,
    }

    result.update(_target_latent_diagnostics(target_latent, y_all, num_classes=4))
    return result


def diagnose_4class(prior, n_tasks=100, mlp_epochs=500, base_seed=0, output_csv="four_class_diagnostics.csv"):
    rows = []
    n_valid = 0

    for idx in range(int(n_tasks)):
        rng = random.Random(int(base_seed) + idx)
        dag_seed = rng.randrange(2**31)
        x_seed = rng.randrange(2**31)
        aleatoric_seed = rng.randrange(2**31)

        task = SCMTask(**prior, num_classes=4, dag_seed=dag_seed, x_seed=x_seed, aleatoric_seed=aleatoric_seed)

        if not _is_table_valid(task, num_classes=4):
            print(f"\r{idx + 1}/{n_tasks}, valid={n_valid}", end="", flush=True)
            continue

        n_valid += 1
        row = _evaluate_task(task, seed=base_seed + idx, mlp_epochs=mlp_epochs)
        row["task_id"] = idx
        rows.append(row)

        print(f"\r{idx + 1}/{n_tasks}, valid={n_valid}", end="", flush=True)

    print()
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"\nvalid_rate: {n_valid / max(int(n_tasks), 1):.4f}")

    if len(df) == 0:
        print("No valid tasks.")
        return df

    summary_columns = [
        "class_min_fraction",
        "class_max_fraction",
        "class_min_max_ratio",
        "cat_ratio",
        "categorical_quality",
        "categorical_retention",
        "mean_retention",
        "min_retention",
        "importance_weighted_retention",
        "max_importance",
        "top3_importance",
        "observed_logistic",
        "observed_mlp",
        "observed_rf",
        "latent_logistic",
        "latent_mlp",
        "latent_rf",
        "logistic_observation_loss",
        "mlp_observation_loss",
        "rf_observation_loss",
        "target_class_means_ordered",
        "target_mean_adjacent_overlap",
        "target_max_adjacent_overlap",
        "target_class_mean_separation",
    ]

    print("\nMean diagnostics:")
    print(df[summary_columns].mean().to_string())

    print("\nMedian diagnostics:")
    print(df[summary_columns].median().to_string())

    print("\nCorrelations with MLP observation loss:")
    corr_columns = [
        "cat_ratio",
        "categorical_quality",
        "categorical_retention",
        "mean_retention",
        "min_retention",
        "importance_weighted_retention",
        "max_importance",
        "top3_importance",
        "class_min_fraction",
        "class_min_max_ratio",
        "target_mean_adjacent_overlap",
        "target_max_adjacent_overlap",
        "target_class_mean_separation",
    ]
    correlations = df[corr_columns].corrwith(df["mlp_observation_loss"]).sort_values()
    print(correlations.to_string())

    return df


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

    "observation_type_probs": (6.0, 2.0, 2.0),
}


if __name__ == "__main__":
    result = diagnose_4class(prior=PRIOR, n_tasks=100, mlp_epochs=500, base_seed=0, output_csv="four_class_diagnostics.csv")





import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, r2_score
from Trash.scm_task_v1.task import SCMTask


def _to_numpy(x):
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_data(task):
    return _to_numpy(task.X_train), _to_numpy(task.X_test), _to_numpy(task.y_train).reshape(-1), _to_numpy(task.y_test).reshape(-1)


def _get_feature_type(task):
    return _to_numpy(task.info["feature_type"]).astype(np.int64)


def _is_table_valid(task, num_classes):
    info = task.info
    if not isinstance(info, dict): return False
    if "is_valid" in info:
        v = info["is_valid"].item() if hasattr(info["is_valid"], "item") else info["is_valid"]
        if not bool(v): return False

    X_train, X_test, y_train, y_test = _get_data(task)
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] == 0: return False
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all(): return False
    if np.isinf(X_train).any() or np.isinf(X_test).any(): return False
    if np.any(np.all(np.isnan(X_train), axis=0)) or np.any(np.all(np.isnan(X_test), axis=0)): return False

    if num_classes is None:
        if len(y_train) < 2 or len(y_test) < 2: return False
        if np.var(y_train) < 1e-8 or np.var(y_test) < 1e-8: return False
    else:
        train_classes = np.unique(y_train.astype(np.int64))
        test_classes = np.unique(y_test.astype(np.int64))
        expected = np.arange(num_classes)
        if not np.array_equal(train_classes, expected) or not np.array_equal(test_classes, expected): return False

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


def _fit_linear_score(X_train, X_test, y_train, y_test, feature_type, num_classes, seed):
    preprocessor = _build_preprocessor(feature_type)

    if num_classes is None:
        model = Pipeline([("preprocess", preprocessor), ("model", LinearRegression())])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return float(r2_score(y_test, pred))

    model = Pipeline([("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000, random_state=seed))])
    model.fit(X_train, y_train.astype(np.int64))
    pred = model.predict(X_test)
    return float(balanced_accuracy_score(y_test.astype(np.int64), pred))


class _MLP(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, d_out))

    def forward(self, x):
        return self.net(x)


def _fit_mlp_score(X_train, X_test, y_train, y_test, feature_type, num_classes, epochs, seed):
    torch.manual_seed(seed)
    preprocessor = _build_preprocessor(feature_type)
    Xtr = preprocessor.fit_transform(X_train).astype(np.float32)
    Xte = preprocessor.transform(X_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    if num_classes is None:
        ytr_t = torch.tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)
        model = _MLP(Xtr.shape[1], 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(Xtr_t)
            loss = torch.nn.functional.mse_loss(pred, ytr_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad(): pred = model(Xte_t).squeeze(1).cpu().numpy()
        return float(r2_score(y_test, pred))

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
    with torch.no_grad(): pred = model(Xte_t).argmax(dim=1).cpu().numpy()
    return float(balanced_accuracy_score(y_test.astype(np.int64), pred))


def _estimate_importance(X_train, X_test, y_train, y_test, feature_type, num_classes, seed):
    preprocessor = _build_preprocessor(feature_type)

    if num_classes is None:
        estimator = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
        scoring = "r2"
    else:
        estimator = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced", n_jobs=1)
        scoring = "balanced_accuracy"

    model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    model.fit(X_train, y_train.astype(np.int64) if num_classes is not None else y_train)
    result = permutation_importance(model, X_test, y_test.astype(np.int64) if num_classes is not None else y_test, scoring=scoring, n_repeats=5, random_state=seed, n_jobs=1)
    importance = np.maximum(result.importances_mean.astype(np.float64), 0.0)

    s = importance.sum()
    if s > 0: importance /= s
    return importance


def _compare_importance(gt, estimated, topk=3):
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    estimated = np.asarray(estimated, dtype=np.float64).reshape(-1)

    if len(gt) != len(estimated) or len(gt) == 0: return np.nan, np.nan

    if np.std(gt) < 1e-12 or np.std(estimated) < 1e-12:
        rho = 0.0
    else:
        rho = spearmanr(gt, estimated).statistic
        if not np.isfinite(rho): rho = 0.0

    k = min(int(topk), len(gt))
    gt_top = set(np.argsort(gt)[-k:])
    est_top = set(np.argsort(estimated)[-k:])
    overlap = len(gt_top & est_top) / k
    return float(rho), float(overlap)


def _evaluate_task(task, num_classes, mlp_epochs, seed, topk):
    X_train, X_test, y_train, y_test = _get_data(task)
    feature_type = _get_feature_type(task)

    cat_ratio = float(np.mean(feature_type != 0))
    linear_score = _fit_linear_score(X_train, X_test, y_train, y_test, feature_type, num_classes, seed)
    mlp_score = _fit_mlp_score(X_train, X_test, y_train, y_test, feature_type, num_classes, mlp_epochs, seed)

    if abs(mlp_score) < 1e-8:
        nonlinearity_ratio = np.nan
    else:
        nonlinearity_ratio = (mlp_score - linear_score) / abs(mlp_score)

    estimated_importance = _estimate_importance(X_train, X_test, y_train, y_test, feature_type, num_classes, seed)
    gt_importance = _to_numpy(task.info["feature_importance"])
    importance_spearman, importance_topk = _compare_importance(gt_importance, estimated_importance, topk)

    return {
        "cat_ratio": cat_ratio,
        "mlp_score": mlp_score,
        "nonlinearity_ratio": float(nonlinearity_ratio),
        "importance_spearman": importance_spearman,
        "importance_topk": importance_topk,
    }


def _mean(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def evaluate_prior(prior, n_tasks=300, task_kind="classification", min_classes=2, max_classes=4, mlp_epochs=500, topk=3, base_seed=0, prior_name="prior"):
    if task_kind not in ("classification", "regression"): raise ValueError("task_kind must be 'classification' or 'regression'.")

    valid_rows = []
    n_valid = 0

    for idx in range(int(n_tasks)):
        rng = random.Random(int(base_seed) + idx)

        if task_kind == "classification":
            num_classes = rng.randint(int(min_classes), int(max_classes))
        else:
            num_classes = None

        dag_seed = rng.randrange(2**31)
        x_seed = rng.randrange(2**31)
        aleatoric_seed = rng.randrange(2**31)

        task = SCMTask(**prior, num_classes=num_classes, dag_seed=dag_seed, x_seed=x_seed, aleatoric_seed=aleatoric_seed)

        if not _is_table_valid(task, num_classes): continue

        n_valid += 1
        metrics = _evaluate_task(task, num_classes, mlp_epochs, seed=base_seed + idx, topk=topk)
        metrics["num_classes"] = num_classes
        valid_rows.append(metrics)

        if (idx + 1) % 10 == 0:
            print(f"\r{prior_name}: {idx + 1}/{n_tasks}, valid={n_valid}", end="", flush=True)

    print()

    valid_rate = n_valid / max(int(n_tasks), 1)

    if task_kind == "regression":
        if not valid_rows:
            return pd.DataFrame([{"priors": prior_name, "valid_rate": valid_rate, "num_classes": np.nan, "task_ratio": 1.0, "cat_ratio": np.nan, "mlp_score": np.nan, "nonlinearity_ratio": np.nan, "importance_spearman": np.nan, "importance_topk": np.nan}])

        return pd.DataFrame([{
            "priors": prior_name,
            "valid_rate": valid_rate,
            "num_classes": np.nan,
            "task_ratio": 1.0,
            "cat_ratio": _mean(valid_rows, "cat_ratio"),
            "mlp_score": _mean(valid_rows, "mlp_score"),
            "nonlinearity_ratio": _mean(valid_rows, "nonlinearity_ratio"),
            "importance_spearman": _mean(valid_rows, "importance_spearman"),
            "importance_topk": _mean(valid_rows, "importance_topk"),
        }])

    output = []

    for num_classes in range(int(min_classes), int(max_classes) + 1):
        group = [row for row in valid_rows if row["num_classes"] == num_classes]
        output.append({
            "priors": prior_name,
            "valid_rate": valid_rate,
            "num_classes": num_classes,
            "task_ratio": len(group) / max(n_valid, 1),
            "cat_ratio": _mean(group, "cat_ratio") if group else np.nan,
            "mlp_score": _mean(group, "mlp_score") if group else np.nan,
            "nonlinearity_ratio": _mean(group, "nonlinearity_ratio") if group else np.nan,
            "importance_spearman": _mean(group, "importance_spearman") if group else np.nan,
            "importance_topk": _mean(group, "importance_topk") if group else np.nan,
        })

    return pd.DataFrame(output)



TASK_KWARGS = dict(
    n_min=400,
    n_max=512,
    d_min=8,
    d_max=16,
    test_frac=0.15,
    p_missing=0.05,
    sampling_penalty=0.25,
    device=None,
    # dag_seed=None,
    # aleatoric_seed=None,
    # x_seed=None,
    

    num_roots=4,
    num_layers=5,
    hidden_width_min=8,
    hidden_width_max=12,
    final_width=1,
    connection_probs=((0.10, 0.30), (0.10, 0.35), (0.20, 0.50), (0.40, 0.80)),
    edge_weight_concentration=(0.30, 0.80),
    latent_noise_scale=(0.0, 0.0),
    child_method_probs=(3, 2, 3, 2),
    source_prior_probs=(0.45, 0.20, 0.15, 0.05),
    joint_mlp_hidden_dim=8,
    edge_family_probs=(4.0, 3.0, 2.0),
    small_mlp_hidden_dim=8,
    soft_tree_depth=2,
    soft_tree_temperature=0.5,




    observation_type_probs=(7.0, 1.5, 1.5),
    categorical_cardinalities=(2, 3, 4, 5, 6),
    categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    min_samples_per_category=8,
    min_component_weight=0.05,
    observation_noise_scale=0.03,
    #device=torch.device("cpu"),
)


if __name__ == "__main__":
    result = evaluate_prior(
        prior=TASK_KWARGS,
        n_tasks=300,
        task_kind="regression",
        # min_classes=2,
        # max_classes=4,
        mlp_epochs=500,
        topk=3,
        base_seed=0,
        prior_name="prior",
    )

    result.to_csv("analysis.csv", index=False)
    print(result)
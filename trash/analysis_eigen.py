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
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ..src.data.scm_task_v2.task import SCMTask
from ..src.data.config import SCM_PRIOR


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
    if "is_valid" in info and not info["is_valid"]:
        return False

    X_train, X_test, y_train, y_test = _get_data(task)
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        return False
    if np.isinf(X_train).any() or np.isinf(X_test).any():
        return False
    if np.any(np.all(np.isnan(X_train), axis=0)) or np.any(np.all(np.isnan(X_test), axis=0)):
        return False

    if num_classes is None:
        if np.var(y_train) < 1e-8 or np.var(y_test) < 1e-8:
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
        with torch.no_grad():
            pred = model(Xte_t).squeeze(1).cpu().numpy()
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
    with torch.no_grad(): 
        pred = model(Xte_t).argmax(dim=1).cpu().numpy()
    return float(balanced_accuracy_score(y_test.astype(np.int64), pred))


def _estimate_rf_importance(X_train, X_test, y_train, y_test, feature_type, num_classes, seed):
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
    if s > 0: 
        importance /= s
    return importance

def _estimate_mlp_importance(X_train, X_test, y_train, y_test, feature_type, num_classes, epochs, seed, n_repeats=5):
    torch.manual_seed(seed)
    preprocessor = _build_preprocessor(feature_type)
    Xtr = preprocessor.fit_transform(X_train).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)

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

        def score(X):
            Xt = torch.tensor(preprocessor.transform(X).astype(np.float32), dtype=torch.float32, device=device)
            model.eval()
            with torch.no_grad():
                pred = model(Xt).squeeze(1).cpu().numpy()
            return r2_score(y_test, pred)

    else:
        ytr_t = torch.tensor(y_train, dtype=torch.long, device=device)
        model = _MLP(Xtr.shape[1], num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(Xtr_t)
            loss = torch.nn.functional.cross_entropy(logits, ytr_t)
            loss.backward()
            optimizer.step()

        def score(X):
            Xt = torch.tensor(preprocessor.transform(X).astype(np.float32), dtype=torch.float32, device=device)
            model.eval()
            with torch.no_grad():
                pred = model(Xt).argmax(dim=1).cpu().numpy()
            return balanced_accuracy_score(y_test.astype(np.int64), pred)

    baseline = score(X_test)
    rng = np.random.default_rng(seed)
    importance = np.zeros(X_test.shape[1], dtype=np.float64)

    for j in range(X_test.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, j] = X_perm[rng.permutation(len(X_perm)), j]
            drops.append(baseline - score(X_perm))
        importance[j] = np.mean(drops)

    importance = np.maximum(importance, 0.0)
    if importance.sum() > 0:
        importance /= importance.sum()
    return importance


def _estimate_mi_importance(X_train, y_train, feature_type, num_classes, seed):
    X = np.asarray(X_train, dtype=np.float64).copy()
    discrete = feature_type != 0

    for j in range(X.shape[1]):
        missing = np.isnan(X[:, j])
        if not missing.any():
            continue

        observed = X[~missing, j]
        if len(observed) == 0:
            X[missing, j] = 0.0
        elif discrete[j]:
            values, counts = np.unique(observed, return_counts=True)
            X[missing, j] = values[np.argmax(counts)]
        else:
            X[missing, j] = np.median(observed)

    if num_classes is None:
        importance = mutual_info_regression(X, y_train, discrete_features=discrete, random_state=seed)
    else:
        importance = mutual_info_classif(X, y_train.astype(np.int64), discrete_features=discrete, random_state=seed)

    importance = np.maximum(np.asarray(importance, dtype=np.float64), 0.0)
    if importance.sum() > 0:
        importance /= importance.sum()
    return importance


def _compare_importance(gt, estimated, topk=3):
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    estimated = np.asarray(estimated, dtype=np.float64).reshape(-1)

    if len(gt) != len(estimated) or len(gt) == 0:
        return np.nan, np.nan

    if np.std(gt) < 1e-12 or np.std(estimated) < 1e-12:
        rho = 0.0
    else:
        rho = spearmanr(gt, estimated).statistic
        if not np.isfinite(rho): 
            rho = 0.0

    k = min(int(topk), len(gt))
    gt_top = set(np.argsort(gt)[-k:])
    est_top = set(np.argsort(estimated)[-k:])
    overlap = len(gt_top & est_top) / k
    return float(rho), float(overlap)




def _mean(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


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

    gt_importance = _to_numpy(task.info["feature_importance"]).astype(np.float64).reshape(-1)

    rf_importance = _estimate_rf_importance(X_train, X_test, y_train, y_test, feature_type, num_classes, seed)
    rf_spearman, rf_topk = _compare_importance(gt_importance, rf_importance, topk)

    mlp_importance = _estimate_mlp_importance(X_train, X_test, y_train, y_test, feature_type, num_classes, mlp_epochs, seed)
    mlp_spearman, mlp_topk = _compare_importance(gt_importance, mlp_importance, topk)

    mi_importance = _estimate_mi_importance(X_train, y_train, feature_type, num_classes, seed)
    mi_spearman, mi_topk = _compare_importance(gt_importance, mi_importance, topk)

    eigenvalues = task.info["importance_eigenvalues"]

    if eigenvalues is None:
        top_eigenvalue_ratio = np.nan
    else:
        eigenvalues = _to_numpy(eigenvalues).astype(np.float64).reshape(-1)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        top_eigenvalue_ratio = (
            float(eigenvalues.max() / eigenvalues.sum())
            if eigenvalues.sum() > 1e-12
            else np.nan
        )

    gt_sorted = np.sort(gt_importance)[::-1]
    gt_top1 = float(gt_sorted[0]) if len(gt_sorted) else np.nan
    gt_top3 = float(gt_sorted[:min(3, len(gt_sorted))].sum()) if len(gt_sorted) else np.nan
    gt_nonzero = int(np.sum(gt_importance > 1e-6))
    gt_nonzero_ratio = float(gt_nonzero / len(gt_importance)) if len(gt_importance) else np.nan

    return {
        "cat_ratio": cat_ratio,
        "mlp_score": mlp_score,
        "nonlinearity_ratio": float(nonlinearity_ratio),
        "gt_top1": gt_top1,
        "gt_top3": gt_top3,
        "gt_nonzero": gt_nonzero,
        "gt_nonzero_ratio": gt_nonzero_ratio,
        "rf_spearman": rf_spearman,
        "rf_topk": rf_topk,
        "mlp_spearman": mlp_spearman,
        "mlp_topk": mlp_topk,
        "mi_spearman": mi_spearman,
        "mi_topk": mi_topk,
        "top_eigenvalue_ratio": top_eigenvalue_ratio,
    }


def evaluate_prior(prior, n_tasks=300, task_kind="classification", min_classes=2, max_classes=4, mlp_epochs=500, topk=3, base_seed=0, prior_name="prior"):
    if task_kind not in ("classification", "regression"): 
        raise ValueError("task_kind must be 'classification' or 'regression'.")

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

        if not _is_table_valid(task, num_classes): 
            continue

        n_valid += 1
        metrics = _evaluate_task(task, num_classes, mlp_epochs, seed=base_seed + idx, topk=topk)
        metrics["num_classes"] = num_classes

        if num_classes is not None:
            _, _, y_train, y_test = _get_data(task)
            y = np.concatenate([y_train, y_test]).astype(np.int64)
            counts = np.bincount(y, minlength=num_classes)
            metrics["class_balance"] = float(counts.min() / counts.max())
        else:
            metrics["class_balance"] = np.nan

        valid_rows.append(metrics)

        if (idx + 1) % 10 == 0:
            print(f"\r{prior_name}: {idx + 1}/{n_tasks}, valid={n_valid}", end="", flush=True)

    print()

    valid_rate = n_valid / max(int(n_tasks), 1)

    if task_kind == "regression":
        if not valid_rows:
            return pd.DataFrame([{
                "priors": prior_name,
                "valid_rate": valid_rate,
                "num_classes": np.nan,
                "task_ratio": 1.0,
                "cat_ratio": np.nan,
                "class_balance": np.nan,
                "mlp_score": np.nan,
                "nonlinearity_ratio": np.nan,
                "gt_top1": np.nan,
                "gt_top3": np.nan,
                "gt_nonzero": np.nan,
                "gt_nonzero_ratio": np.nan,
                "rf_spearman": np.nan,
                "rf_topk": np.nan,
                "mlp_spearman": np.nan,
                "mlp_topk": np.nan,
                "mi_spearman": np.nan,
                "mi_topk": np.nan,
                "top_eigenvalue_ratio": np.nan,
            }])

        return pd.DataFrame([{
            "priors": prior_name,
            "valid_rate": valid_rate,
            "num_classes": np.nan,
            "task_ratio": 1.0,
            "cat_ratio": _mean(valid_rows, "cat_ratio"),
            "class_balance": np.nan,
            "mlp_score": _mean(valid_rows, "mlp_score"),
            "nonlinearity_ratio": _mean(valid_rows, "nonlinearity_ratio"),
            "gt_top1": _mean(valid_rows, "gt_top1"),
            "gt_top3": _mean(valid_rows, "gt_top3"),
            "gt_nonzero": _mean(valid_rows, "gt_nonzero"),
            "gt_nonzero_ratio": _mean(valid_rows, "gt_nonzero_ratio"),
            "rf_spearman": _mean(valid_rows, "rf_spearman"),
            "rf_topk": _mean(valid_rows, "rf_topk"),
            "mlp_spearman": _mean(valid_rows, "mlp_spearman"),
            "mlp_topk": _mean(valid_rows, "mlp_topk"),
            "mi_spearman": _mean(valid_rows, "mi_spearman"),
            "mi_topk": _mean(valid_rows, "mi_topk"),
            "top_eigenvalue_ratio": _mean(valid_rows, "top_eigenvalue_ratio"),
            
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
            "class_balance": _mean(group, "class_balance") if group else np.nan,
            "mlp_score": _mean(group, "mlp_score") if group else np.nan,
            "nonlinearity_ratio": _mean(group, "nonlinearity_ratio") if group else np.nan,
            "gt_top1": _mean(group, "gt_top1") if group else np.nan,
            "gt_top3": _mean(group, "gt_top3") if group else np.nan,
            "gt_nonzero": _mean(group, "gt_nonzero") if group else np.nan,
            "gt_nonzero_ratio": _mean(group, "gt_nonzero_ratio") if group else np.nan,
            "rf_spearman": _mean(group, "rf_spearman") if group else np.nan,
            "rf_topk": _mean(group, "rf_topk") if group else np.nan,
            "mlp_spearman": _mean(group, "mlp_spearman") if group else np.nan,
            "mlp_topk": _mean(group, "mlp_topk") if group else np.nan,
            "mi_spearman": _mean(group, "mi_spearman") if group else np.nan,
            "mi_topk": _mean(group, "mi_topk") if group else np.nan,
            "top_eigenvalue_ratio": _mean(group, "top_eigenvalue_ratio") if group else np.nan,
        })

    return pd.DataFrame(output)




if __name__ == "__main__":

    seed = 17

    config = {
        **SCM_PRIOR,
        "importance_method": "eigen_90",
    }
    result = evaluate_prior(
        prior=config,
        n_tasks=500,
        task_kind="regression",
        mlp_epochs=500,
        topk=3,
        base_seed=seed,
        prior_name="prior",
    )

    result.to_csv("analysis_eigen_90_reg.csv", index=False)

    result = evaluate_prior(
        prior=config,
        n_tasks=500,
        task_kind="classification",
        min_classes=2,
        max_classes=4,
        mlp_epochs=500,
        topk=3,
        base_seed=seed,
        prior_name="prior",
    )
    result.to_csv("analysis_eigen_90_cls.csv", index=False)



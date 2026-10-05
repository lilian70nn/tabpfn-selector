import csv
import os

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from tqdm import tqdm
from sklearn.metrics import mutual_info_score

from ..src.data.datasets import SyntheticTaskDataset
from ..src.data.scm_task_v2.task import SCMTask


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_importance(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    total = x.sum()
    if total <= 1e-12:
        return np.zeros_like(x)
    return x / total


def _impute_for_mi(X, feature_type):
    X = np.asarray(X, dtype=np.float64).copy()
    feature_type = np.asarray(feature_type, dtype=np.int64)

    for j in range(X.shape[1]):
        missing = ~np.isfinite(X[:, j])
        if not missing.any():
            continue

        observed = X[~missing, j]

        if observed.size == 0:
            fill = 0.0
        elif feature_type[j] == 0:
            fill = float(np.median(observed))
        else:
            values, counts = np.unique(observed, return_counts=True)
            fill = float(values[np.argmax(counts)])

        X[missing, j] = fill

    return X


def _build_preprocessor(feature_type):
    feature_type = np.asarray(feature_type, dtype=np.int64)
    continuous_idx = np.where(feature_type == 0)[0].tolist()
    categorical_idx = np.where(feature_type != 0)[0].tolist()
    transformers = []

    if continuous_idx:
        continuous_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        transformers.append(("continuous", continuous_pipeline, continuous_idx))

    if categorical_idx:
        categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(("categorical", categorical_pipeline, categorical_idx))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def estimate_mi_importance(X_train, y_train, feature_type, task_kind, seed=0):
    X_train = _to_numpy(X_train)
    y_train = _to_numpy(y_train).reshape(-1)
    feature_type = _to_numpy(feature_type).astype(np.int64)
    X_train = _impute_for_mi(X_train, feature_type)

    importance = np.zeros(X_train.shape[1], dtype=np.float64)

    if task_kind == "regression":
        n_bins = min(10, max(2, int(np.sqrt(len(y_train)))))
        edges = np.unique(np.quantile(y_train, np.linspace(0.0, 1.0, n_bins + 1)))
        y_discrete = np.digitize(y_train, edges[1:-1], right=False)

        for j in range(X_train.shape[1]):
            x = X_train[:, j]

            if np.unique(x).size <= 1:
                continue

            if feature_type[j] != 0:
                importance[j] = mutual_info_score(x.astype(np.int64), y_discrete)
            else:
                importance[j] = mutual_info_regression(
                    x.reshape(-1, 1),
                    y_train.astype(np.float64),
                    discrete_features=False,
                    random_state=seed + j,
                    n_jobs=1,
                )[0]

    else:
        discrete_features = feature_type != 0
        importance = mutual_info_classif(
            X_train,
            y_train.astype(np.int64),
            discrete_features=discrete_features,
            random_state=seed,
            n_jobs=1,
        )

    return _normalize_importance(importance)


def estimate_rf_importance(X_train, y_train, X_test, y_test, feature_type, task_kind, seed=0, n_estimators=300, n_repeats=10):
    X_train = _to_numpy(X_train)
    y_train = _to_numpy(y_train).reshape(-1)
    X_test = _to_numpy(X_test)
    y_test = _to_numpy(y_test).reshape(-1)
    feature_type = _to_numpy(feature_type).astype(np.int64)

    preprocessor = _build_preprocessor(feature_type)

    if task_kind == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1, class_weight="balanced")
        scoring = "accuracy"
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1)
        scoring = "r2"
        y_train = y_train.astype(np.float64)
        y_test = y_test.astype(np.float64)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    result = permutation_importance(pipeline, X_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return _normalize_importance(result.importances_mean)


def estimate_linear_perm_importance(X_train, y_train, X_test, y_test, feature_type, task_kind, seed=0, n_repeats=10):
    X_train = _to_numpy(X_train)
    y_train = _to_numpy(y_train).reshape(-1)
    X_test = _to_numpy(X_test)
    y_test = _to_numpy(y_test).reshape(-1)
    feature_type = _to_numpy(feature_type).astype(np.int64)

    preprocessor = _build_preprocessor(feature_type)

    if task_kind == "classification":
        model = LogisticRegression(max_iter=2000, random_state=seed)
        scoring = "accuracy"
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)
    else:
        model = Ridge(alpha=1.0)
        scoring = "r2"
        y_train = y_train.astype(np.float64)
        y_test = y_test.astype(np.float64)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    result = permutation_importance(pipeline, X_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return _normalize_importance(result.importances_mean)


def compare_importance(ground_truth, estimated, top_k=3):
    ground_truth = _normalize_importance(ground_truth)
    estimated = _normalize_importance(estimated)

    d = len(ground_truth)

    if d != len(estimated):
        raise ValueError(f"Importance length mismatch: ground_truth={d}, estimated={len(estimated)}.")

    if np.std(ground_truth) <= 1e-12 or np.std(estimated) <= 1e-12:
        spearman = 0.0
    else:
        spearman = spearmanr(ground_truth, estimated).statistic
        if not np.isfinite(spearman):
            spearman = 0.0

    gt_top1 = int(np.argmax(ground_truth))
    est_top1 = int(np.argmax(estimated))
    k = min(int(top_k), d)
    gt_topk = set(np.argsort(ground_truth)[-k:])
    est_topk = set(np.argsort(estimated)[-k:])

    return {
        "spearman": float(spearman),
        "top1_match": float(gt_top1 == est_top1),
        "topk_overlap": float(len(gt_topk & est_topk) / k),
        "l1_distance": float(np.abs(ground_truth - estimated).sum()),
    }


def _summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def analyze_importance(dataset, output_csv, n_tasks=None, seed_offset=0, rf_estimators=300, permutation_repeats=10, top_k=3, resume=True):
    if n_tasks is None:
        n_tasks = len(dataset)

    n_tasks = min(int(n_tasks), len(dataset))

    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")

    fieldnames = ["task_index", "task_kind", "num_classes", "d", "method", "spearman", "top1_match", "topk_overlap", "l1_distance"]

    completed_tasks = set()

    if resume and os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)

        if not existing.empty:
            counts = existing.groupby("task_index")["method"].nunique()
            completed_tasks = set(counts[counts >= 3].index.astype(int).tolist())

    if not resume and os.path.exists(output_csv):
        os.remove(output_csv)

    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0

    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
            f.flush()

        progress = tqdm(range(n_tasks), desc="Importance analysis", unit="task")

        for task_index in progress:
            if task_index in completed_tasks:
                progress.set_postfix_str("resume: skipped")
                continue

            progress.set_postfix_str(f"task={task_index}")

            task = dataset[task_index]
            seed = int(seed_offset + task_index)

            X_train = task.X_train
            y_train = task.y_train
            X_test = task.X_test
            y_test = task.y_test
            feature_type = task.info["feature_type"]
            ground_truth = _normalize_importance(_to_numpy(task.info["feature_importance"]))

            task_kind = "regression" if task.n_classes is None else "classification"
            num_classes = None if task.n_classes is None else int(task.n_classes)

            progress.set_postfix_str(f"task={task_index} MI")
            mi_importance = estimate_mi_importance(X_train=X_train, y_train=y_train, feature_type=feature_type, task_kind=task_kind, seed=seed)

            progress.set_postfix_str(f"task={task_index} RF")
            rf_importance = estimate_rf_importance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, feature_type=feature_type, task_kind=task_kind, seed=seed, n_estimators=rf_estimators, n_repeats=permutation_repeats)

            progress.set_postfix_str(f"task={task_index} Linear")
            linear_perm_importance = estimate_linear_perm_importance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, feature_type=feature_type, task_kind=task_kind, seed=seed, n_repeats=permutation_repeats)

            estimates = {"mi": mi_importance, "rf": rf_importance, "linear_perm": linear_perm_importance}

            for method_name, estimated in estimates.items():
                metrics = compare_importance(ground_truth, estimated, top_k=top_k)

                writer.writerow({
                    "task_index": task_index,
                    "task_kind": task_kind,
                    "num_classes": "" if num_classes is None else num_classes,
                    "d": len(ground_truth),
                    "method": method_name,
                    "spearman": metrics["spearman"],
                    "top1_match": metrics["top1_match"],
                    "topk_overlap": metrics["topk_overlap"],
                    "l1_distance": metrics["l1_distance"],
                })

            f.flush()

    return output_csv


def summarize_csv(csv_path):
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV is empty.")

    duplicate_mask = df.duplicated(subset=["task_index", "method"], keep="last")

    if duplicate_mask.any():
        df = df.loc[~duplicate_mask].copy()

    metrics = ["spearman", "top1_match", "topk_overlap", "l1_distance"]

    def summarize_group(group):
        result = {}

        for method_name, method_df in group.groupby("method"):
            result[method_name] = {}

            for metric in metrics:
                values = method_df[metric].to_numpy(dtype=np.float64)
                result[method_name][metric] = _summarize(values)

        return result

    task_kinds = df["task_kind"].dropna().unique()

    if len(task_kinds) != 1:
        raise ValueError("CSV contains mixed regression and classification tasks.")

    task_kind = task_kinds[0]

    if task_kind == "regression":
        return {
            "task_kind": "regression",
            "n_tasks": int(df["task_index"].nunique()),
            **summarize_group(df),
        }

    summary = {
        "task_kind": "classification",
        "n_tasks": int(df["task_index"].nunique()),
        "by_num_classes": {},
    }

    for num_classes, group in df.groupby("num_classes"):
        summary["by_num_classes"][int(num_classes)] = {
            "n_tasks": int(group["task_index"].nunique()),
            **summarize_group(group),
        }

    return summary


def _print_method_summary(method_name, method):
    print(
        f"{method_name:12s} | "
        f"Spearman mean={method['spearman']['mean']:+.4f} std={method['spearman']['std']:.4f} "
        f"p10={method['spearman']['p10']:+.4f} p25={method['spearman']['p25']:+.4f} "
        f"median={method['spearman']['median']:+.4f} "
        f"p75={method['spearman']['p75']:+.4f} p90={method['spearman']['p90']:+.4f} | "
        f"Top1 mean={method['top1_match']['mean']:.4f} std={method['top1_match']['std']:.4f} | "
        f"Top3 mean={method['topk_overlap']['mean']:.4f} std={method['topk_overlap']['std']:.4f} | "
        f"L1 mean={method['l1_distance']['mean']:.4f} std={method['l1_distance']['std']:.4f} "
        f"p10={method['l1_distance']['p10']:.4f} p25={method['l1_distance']['p25']:.4f} "
        f"median={method['l1_distance']['median']:.4f} "
        f"p75={method['l1_distance']['p75']:.4f} p90={method['l1_distance']['p90']:.4f}"
    )


def print_importance_summary(summary):
    print("\n================ IMPORTANCE ANALYSIS ================")

    if summary["task_kind"] == "regression":
        print(f"\nREGRESSION | n_tasks={summary['n_tasks']}")

        for method_name in ("mi", "rf", "linear_perm"):
            _print_method_summary(method_name, summary[method_name])

    else:
        for num_classes in sorted(summary["by_num_classes"]):
            group = summary["by_num_classes"][num_classes]
            print(f"\n{num_classes}-CLASS | n_tasks={group['n_tasks']}")

            for method_name in ("mi", "rf", "linear_perm"):
                _print_method_summary(method_name, group[method_name])

    print("=====================================================")


prior = {
    "n_min": 400,
    "n_max": 512,
    "d_min": 8,
    "d_max": 16,
    "test_frac": 0.15,
    "p_missing": 0.05,

    "num_roots": 5,
    "num_layers": 3,
    "final_width": 1,

    "connection_probs": ((0.30, 0.40), (0.55, 0.75)),
    "latent_noise_scale": (0.0, 0.03),
    "source_prior_probs": (0.45, 0.20, 0.15, 0.05),

    "arity_probs": (2.5, 3.0, 3.0),
    "unary_op_probs": (0.5, 1.5, 2.0, 2.0, 1.5, 1.0, 1.5),
    "binary_op_probs": (2.0, 2.0, 2.0, 2.0),
    "ternary_op_probs": (3.0, 1.0, 1.0, 3.0),

    "scale_min": 0.25,
    "scale_max": 4.0,

    "observation_type_probs": (6.0, 2.0, 2.0),
    "categorical_cardinalities": (2, 3, 4, 5, 6),
    "categorical_cardinality_probs": (0.40, 0.30, 0.18, 0.08, 0.04),

    "min_samples_per_category": 8,
    "min_component_weight": 0.05,
    "observation_noise_scale": 0.03,
}


if __name__ == "__main__":
    output_csv = "importance_classification.csv"

    dataset = SyntheticTaskDataset(
        num_tasks=300,
        task_factory=SCMTask,
        task_kwargs=prior,
        task_kind="classification",
        base_seed=0,
        max_attempts=10,
        min_classes=2, 
        max_classes=4
    )

    analyze_importance(
        dataset=dataset,
        output_csv=output_csv,
        n_tasks=300,
        seed_offset=500000,
        rf_estimators=300,
        permutation_repeats=10,
        top_k=3,
        resume=True,
    )

    summary = summarize_csv(output_csv)
    print_importance_summary(summary)



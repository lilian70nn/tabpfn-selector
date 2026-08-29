import numpy as np
import pandas as pd
import torch

from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2.task import SCMTask

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, log_loss, r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
from tqdm import tqdm


def _safe_spearman(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}.")
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else 0.0


def _topk_overlap(a, b, k=5):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    k = min(int(k), len(a))
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    return float(len(top_a & top_b) / k)


def _build_preprocessor(feature_type):
    feature_type = np.asarray(feature_type, dtype=np.int64)
    continuous_idx = np.where(feature_type == 0)[0].tolist()
    categorical_idx = np.where(feature_type != 0)[0].tolist()
    transformers = []

    if continuous_idx:
        transformers.append(("continuous", SimpleImputer(strategy="median"), continuous_idx))

    if categorical_idx:
        categorical_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
        transformers.append(("categorical", categorical_pipeline, categorical_idx))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_rf(is_classification, seed=0, n_estimators=200):
    if is_classification:
        return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=1)
    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=1)


def _evaluate_model(model, X_train, y_train, X_test, y_test, is_classification, n_classes):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {}

    if is_classification:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
        metrics["macro_f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        y_prob = model.predict_proba(X_test)
        metrics["log_loss"] = float(log_loss(y_test, y_prob, labels=model.classes_))
        if n_classes == 2:
            metrics["auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
        else:
            metrics["auc"] = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro", labels=model.classes_))
    else:
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))

    return metrics


def _single_feature_importance(X_train, y_train, X_test, y_test, feature_type, is_classification, seed=0):
    d = X_train.shape[1]
    scores = np.zeros(d, dtype=np.float64)

    for j in range(d):
        X_train_j = X_train[:, [j]]
        X_test_j = X_test[:, [j]]

        if feature_type[j] == 0:
            preprocessor_j = SimpleImputer(strategy="median")
        else:
            preprocessor_j = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))

        if is_classification:
            estimator = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1)
        else:
            estimator = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=seed, n_jobs=1)

        model = make_pipeline(preprocessor_j, estimator)
        model.fit(X_train_j, y_train)
        y_pred = model.predict(X_test_j)

        if is_classification:
            scores[j] = balanced_accuracy_score(y_test, y_pred)
        else:
            scores[j] = r2_score(y_test, y_pred)

    return scores


def _permutation_importance(fitted_rf, X_test, y_test, is_classification, seed=0, n_repeats=5):
    scoring = "balanced_accuracy" if is_classification else "r2"
    result = permutation_importance(fitted_rf, X_test, y_test, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return np.asarray(result.importances_mean, dtype=np.float64)


def _drop_column_importance(X_train, y_train, X_test, y_test, feature_type, is_classification, seed=0):
    d = X_train.shape[1]
    full_preprocessor = _build_preprocessor(feature_type)
    full_model = make_pipeline(full_preprocessor, _make_rf(is_classification, seed=seed, n_estimators=150))
    full_model.fit(X_train, y_train)
    full_pred = full_model.predict(X_test)

    if is_classification:
        full_score = balanced_accuracy_score(y_test, full_pred)
    else:
        full_score = r2_score(y_test, full_pred)

    importance = np.zeros(d, dtype=np.float64)

    for j in range(d):
        keep = np.arange(d) != j
        X_train_drop = X_train[:, keep]
        X_test_drop = X_test[:, keep]
        feature_type_drop = feature_type[keep]

        preprocessor = _build_preprocessor(feature_type_drop)
        model = make_pipeline(preprocessor, _make_rf(is_classification, seed=seed, n_estimators=150))
        model.fit(X_train_drop, y_train)
        pred = model.predict(X_test_drop)

        if is_classification:
            drop_score = balanced_accuracy_score(y_test, pred)
        else:
            drop_score = r2_score(y_test, pred)

        importance[j] = full_score - drop_score

    return importance


def run_scm_sanity_check(dataset, top_k=5):
    rows = []

    for i in tqdm(range(len(dataset)), desc="SCM sanity", unit="task"):
        task = dataset[i]

        X_train = task.X_train.detach().cpu().numpy()
        y_train = task.y_train.detach().cpu().numpy().reshape(-1)
        X_test = task.X_test.detach().cpu().numpy()
        y_test = task.y_test.detach().cpu().numpy().reshape(-1)

        info = task.info
        feature_type = info["feature_type"].detach().cpu().numpy().astype(np.int64)
        gt = info["feature_importance"].detach().cpu().numpy().astype(np.float64)

        is_classification = task.n_classes is not None
        task_kind = "classification" if is_classification else "regression"
        n_classes = int(task.n_classes) if is_classification else None

        if is_classification:
            y_train = y_train.astype(np.int64)
            y_test = y_test.astype(np.int64)
        else:
            y_train = y_train.astype(np.float64)
            y_test = y_test.astype(np.float64)

        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        row = {
            "dataset_id": i,
            "task_kind": task_kind,
            "n_classes": np.nan if n_classes is None else n_classes,
            "n": len(X_all),
            "d": X_train.shape[1],
            "missing_rate": float(np.isnan(X_all).mean()),
            "continuous_ratio": float((feature_type == 0).mean()),
            "categorical_ratio": float((feature_type != 0).mean()),
        }

        if is_classification:
            labels, counts = np.unique(y_all, return_counts=True)
            row["class_ratio"] = (counts / counts.sum()).tolist()
        else:
            row["y_mean"] = float(np.mean(y_all))
            row["y_std"] = float(np.std(y_all))

        if "feature_observation_type_ids" in info:
            obs_type = info["feature_observation_type_ids"].detach().cpu().numpy()
            row["obs_continuous_ratio"] = float((obs_type == 0).mean())
            row["obs_prototype_ratio"] = float((obs_type == 1).mean())
            row["obs_binning_ratio"] = float((obs_type == 2).mean())

        preprocessor = _build_preprocessor(feature_type)

        if is_classification:
            models = {
                "dummy": DummyClassifier(strategy="prior"),
                "logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
                "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=1),
                "extra_trees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=0, n_jobs=1),
            }
        else:
            models = {
                "dummy": DummyRegressor(strategy="mean"),
                "ridge": Ridge(alpha=1.0),
                "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=1),
                "extra_trees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=1),
            }

        fitted_rf = None

        for model_name, estimator in models.items():
            model = make_pipeline(preprocessor, estimator)
            metrics = _evaluate_model(model, X_train, y_train, X_test, y_test, is_classification, n_classes)

            for metric_name, value in metrics.items():
                row[f"{model_name}_{metric_name}"] = value

            if model_name == "random_forest":
                fitted_rf = model

        single_imp = _single_feature_importance(X_train, y_train, X_test, y_test, feature_type, is_classification, seed=i)
        permutation_imp = _permutation_importance(fitted_rf, X_test, y_test, is_classification, seed=i, n_repeats=5)
        drop_column_imp = _drop_column_importance(X_train, y_train, X_test, y_test, feature_type, is_classification, seed=i)

        row["spearman_gt_single"] = _safe_spearman(gt, single_imp)
        row["spearman_gt_permutation"] = _safe_spearman(gt, permutation_imp)
        row["spearman_gt_drop_column"] = _safe_spearman(gt, drop_column_imp)

        row["topk_gt_single"] = _topk_overlap(gt, single_imp, top_k)
        row["topk_gt_permutation"] = _topk_overlap(gt, permutation_imp, top_k)
        row["topk_gt_drop_column"] = _topk_overlap(gt, drop_column_imp, top_k)

        row["gt_importance"] = gt.tolist()
        row["single_feature_importance"] = single_imp.tolist()
        row["permutation_importance"] = permutation_imp.tolist()
        row["drop_column_importance"] = drop_column_imp.tolist()

        rows.append(row)

    return pd.DataFrame(rows)


def make_summary(df):
    group_cols = ["n_classes"] if df["task_kind"].iloc[0] == "classification" else ["task_kind"]
    excluded = {"dataset_id", "n", "d", "n_classes"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
    rows = []

    for group_value, group in df.groupby(group_cols[0]):
        row = {group_cols[0]: group_value, "n_tasks": len(group)}

        for col in numeric_cols:
            values = group[col].dropna().to_numpy(dtype=np.float64)
            if len(values) == 0:
                continue
            row[f"mean_{col}"] = float(np.mean(values))
            row[f"median_{col}"] = float(np.median(values))

        rows.append(row)

    return pd.DataFrame(rows)


TASK_KWARGS = {
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
    base_seeds = [0, 10_000, 20_000, 30_000, 40_000]
    num_tasks_per_seed = 100
    all_dfs = []

    for base_seed in base_seeds:
        print(f"\n===== BASE SEED {base_seed} =====")

        dataset = SyntheticTaskDataset(
            num_tasks=num_tasks_per_seed,
            task_factory=SCMTask,
            task_kind="classification",
            min_classes=2,
            max_classes=4,
            base_seed=base_seed,
            task_kwargs=TASK_KWARGS,
            max_attempts=10,
        )

        df_seed = run_scm_sanity_check(dataset, top_k=5)
        df_seed["base_seed"] = base_seed
        all_dfs.append(df_seed)

    df = pd.concat(all_dfs, ignore_index=True)
    summary = make_summary(df)

    df.to_csv("scm_v2_sanity_all_tables.csv", index=False)
    summary.to_csv("scm_v2_sanity_summary.csv", index=False)

    print("\n================ OVERALL SUMMARY ================")
    print(summary.to_string(index=False))
    print("=================================================")
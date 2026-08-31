from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task import MixedSCMTask


# ============================================================
# Configuration
# ============================================================

NUM_TABLES = 10
BASE_SEED = 0

PERMUTATION_REPEATS = 10
TOP_K_FRAC = 0.25

PREDICTION_CSV = "scm_prediction_metrics.csv"
IMPORTANCE_CSV = "scm_importance_metrics.csv"
SUBSET_CSV = "scm_feature_subset_metrics.csv"


# ============================================================
# Utilities
# ============================================================

def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def unpack_task(task):
    """
    Support two possible SyntheticTaskDataset return formats:

    1. Tuple/list:
       X_train, y_train, X_test, y_test, info

    2. Task object with attributes:
       task.X_train, task.y_train, ...
    """

    if isinstance(task, (tuple, list)):
        if len(task) != 5:
            raise ValueError(
                "Expected five elements: "
                "X_train, y_train, X_test, y_test, info. "
                f"Received {len(task)} elements."
            )

        return task

    required_attributes = [
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "info",
    ]

    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(task, attribute)
    ]

    if missing:
        raise TypeError(
            f"Cannot unpack dataset item of type {type(task)}. "
            f"Missing attributes: {missing}"
        )

    return (
        task.X_train,
        task.y_train,
        task.X_test,
        task.y_test,
        task.info,
    )


def normalize_importance(importance):
    importance = np.asarray(importance, dtype=float)
    importance = np.maximum(importance, 0.0)

    total = importance.sum()

    if total <= 1e-12:
        return np.zeros_like(importance)

    return importance / total


def safe_spearman(reference, estimated):
    reference = np.asarray(reference, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    if len(reference) != len(estimated):
        raise ValueError(
            f"Importance lengths differ: "
            f"{len(reference)} versus {len(estimated)}"
        )

    if np.std(reference) <= 1e-12:
        return np.nan

    if np.std(estimated) <= 1e-12:
        return np.nan

    result = spearmanr(reference, estimated)

    if hasattr(result, "statistic"):
        return float(result.statistic)

    return float(result.correlation)


def get_topk_indices(importance, fraction=0.25):
    importance = np.asarray(importance, dtype=float)

    num_features = len(importance)
    k = max(1, int(np.ceil(fraction * num_features)))

    return np.argsort(-importance)[:k]


def topk_overlap(reference, estimated, fraction=0.25):
    reference_top = set(
        get_topk_indices(reference, fraction).tolist()
    )

    estimated_top = set(
        get_topk_indices(estimated, fraction).tolist()
    )

    return len(reference_top & estimated_top) / len(reference_top)


# ============================================================
# Preprocessing
# ============================================================

def make_one_hot_encoder():
    """
    Compatible with both newer and older sklearn versions.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def make_preprocessor(feature_type):
    feature_type = np.asarray(feature_type, dtype=int)

    continuous_indices = np.where(feature_type == 0)[0].tolist()
    categorical_indices = np.where(feature_type == 1)[0].tolist()

    transformers = []

    if continuous_indices:
        continuous_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "continuous",
                continuous_pipeline,
                continuous_indices,
            )
        )

    if categorical_indices:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "one_hot",
                    make_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_indices,
            )
        )

    if not transformers:
        raise ValueError("No usable features were found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# Models
# ============================================================

def make_models(feature_type, seed):
    return {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(feature_type),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),

        "random_forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(feature_type),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "extra_trees": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(feature_type),
                ),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# Prediction metrics
# ============================================================

def compute_auc(model, X_test, y_test):
    """
    Automatically handles binary and multiclass tables.
    """

    try:
        probabilities = model.predict_proba(X_test)
        model_classes = np.asarray(model.classes_)
        test_classes = np.unique(y_test)

        # Binary classification
        if len(model_classes) == 2:
            if len(test_classes) < 2:
                return np.nan

            positive_class = model_classes[1]
            binary_target = (y_test == positive_class).astype(int)

            return float(
                roc_auc_score(
                    binary_target,
                    probabilities[:, 1],
                )
            )

        # Multiclass classification
        if len(test_classes) != len(model_classes):
            return np.nan

        return float(
            roc_auc_score(
                y_test,
                probabilities,
                labels=model_classes,
                multi_class="ovr",
                average="macro",
            )
        )

    except (AttributeError, ValueError):
        return np.nan


def fit_and_evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, prediction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, prediction)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "auc": compute_auc(
            model,
            X_test,
            y_test,
        ),
    }

    return metrics


# ============================================================
# Importance methods
# ============================================================

def compute_model_permutation_importance(
    fitted_model,
    X_test,
    y_test,
    seed,
):
    """
    Since fitted_model is a pipeline, sklearn permutes the
    original table columns before preprocessing.

    Therefore the result contains one importance value per
    original SCM feature.
    """

    result = permutation_importance(
        estimator=fitted_model,
        X=X_test,
        y=y_test,
        scoring="balanced_accuracy",
        n_repeats=PERMUTATION_REPEATS,
        random_state=seed,
        n_jobs=-1,
    )

    return normalize_importance(
        result.importances_mean
    )


def impute_for_mutual_information(
    X,
    feature_type,
):
    X = np.asarray(X, dtype=float).copy()
    feature_type = np.asarray(feature_type, dtype=int)

    for feature_index in range(X.shape[1]):
        column = X[:, feature_index]
        missing_mask = np.isnan(column)

        if not missing_mask.any():
            continue

        observed = column[~missing_mask]

        if len(observed) == 0:
            fill_value = 0.0

        elif feature_type[feature_index] == 0:
            fill_value = float(np.median(observed))

        else:
            values, counts = np.unique(
                observed,
                return_counts=True,
            )

            fill_value = float(
                values[np.argmax(counts)]
            )

        column[missing_mask] = fill_value
        X[:, feature_index] = column

    return X


def compute_mutual_information(
    X_train,
    y_train,
    feature_type,
    seed,
):
    X_imputed = impute_for_mutual_information(
        X_train,
        feature_type,
    )

    discrete_features = (
        np.asarray(feature_type, dtype=int) == 1
    )

    importance = mutual_info_classif(
        X=X_imputed,
        y=y_train,
        discrete_features=discrete_features,
        random_state=seed,
    )

    return normalize_importance(importance)


# ============================================================
# Feature subset checks
# ============================================================

def evaluate_feature_subset(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_type,
    selected_indices,
    seed,
):
    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    if len(selected_indices) == 0:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "auc": np.nan,
        }

    X_train_subset = X_train[:, selected_indices]
    X_test_subset = X_test[:, selected_indices]
    subset_feature_type = feature_type[selected_indices]

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(subset_feature_type),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return fit_and_evaluate(
        model=model,
        X_train=X_train_subset,
        y_train=y_train,
        X_test=X_test_subset,
        y_test=y_test,
    )


# ============================================================
# Evaluate one SCM table
# ============================================================

def evaluate_one_table(task, table_id):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        info,
    ) = unpack_task(task)

    X_train = to_numpy(X_train).astype(float)
    X_test = to_numpy(X_test).astype(float)

    y_train = to_numpy(y_train).astype(int)
    y_test = to_numpy(y_test).astype(int)

    feature_type = to_numpy(
        info["feature_type"]
    ).astype(int)

    ground_truth_importance = to_numpy(
        info["importance_ratio"]
    ).astype(float)

    is_active = to_numpy(
        info["is_active"]
    ).astype(float)

    feature_ids = to_numpy(
        info["feature_ids"]
    ).astype(int)

    target_id = int(
        to_numpy(info["target_id"]).item()
    )

    all_classes = np.unique(
        np.concatenate([y_train, y_test])
    )

    # This is different for every table.
    num_classes = len(all_classes)
    num_features = X_train.shape[1]

    active_indices = np.where(
        is_active > 0.5
    )[0]

    inactive_indices = np.where(
        is_active <= 0.5
    )[0]

    gt_top_indices = get_topk_indices(
        ground_truth_importance,
        fraction=TOP_K_FRAC,
    )

    all_feature_indices = np.arange(num_features)

    remaining_after_topk_removal = np.setdiff1d(
        all_feature_indices,
        gt_top_indices,
    )

    print()
    print("=" * 100)
    print(
        f"TABLE {table_id} | "
        f"classes={num_classes} {all_classes.tolist()} | "
        f"features={num_features} | "
        f"active={len(active_indices)}"
    )
    print("=" * 100)

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    print(
        "y_train:",
        np.unique(y_train, return_counts=True),
    )

    print(
        "y_test :",
        np.unique(y_test, return_counts=True),
    )

    print("feature_ids:", feature_ids.tolist())
    print("target_id:", target_id)

    prediction_rows = []
    importance_rows = []

    models = make_models(
        feature_type=feature_type,
        seed=BASE_SEED + table_id,
    )

    for model_name, model in models.items():
        metrics = fit_and_evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        prediction_rows.append(
            {
                "table_id": table_id,
                "num_classes": num_classes,
                "num_features": num_features,
                "num_active": len(active_indices),
                "model": model_name,
                **metrics,
            }
        )

        estimated_importance = (
            compute_model_permutation_importance(
                fitted_model=model,
                X_test=X_test,
                y_test=y_test,
                seed=BASE_SEED + table_id,
            )
        )

        importance_rows.append(
            {
                "table_id": table_id,
                "num_classes": num_classes,
                "num_features": num_features,
                "method": f"{model_name}_permutation",
                "spearman": safe_spearman(
                    ground_truth_importance,
                    estimated_importance,
                ),
                "topk_overlap": topk_overlap(
                    ground_truth_importance,
                    estimated_importance,
                    fraction=TOP_K_FRAC,
                ),
            }
        )

        print(
            f"{model_name:22s} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"f1={metrics['macro_f1']:.4f} | "
            f"auc={metrics['auc']:.4f}"
        )

    # Mutual information
    mi_importance = compute_mutual_information(
        X_train=X_train,
        y_train=y_train,
        feature_type=feature_type,
        seed=BASE_SEED + table_id,
    )

    importance_rows.append(
        {
            "table_id": table_id,
            "num_classes": num_classes,
            "num_features": num_features,
            "method": "mutual_information",
            "spearman": safe_spearman(
                ground_truth_importance,
                mi_importance,
            ),
            "topk_overlap": topk_overlap(
                ground_truth_importance,
                mi_importance,
                fraction=TOP_K_FRAC,
            ),
        }
    )

    print()
    print("Importance agreement with SCM ground truth:")

    for row in importance_rows:
        print(
            f"{row['method']:35s} | "
            f"spearman={row['spearman']:.4f} | "
            f"top-k={row['topk_overlap']:.4f}"
        )

    # Feature subset sanity checks
    subset_rows = []

    subset_definitions = {
        "all_features": all_feature_indices,
        "active_only": active_indices,
        "inactive_only": inactive_indices,
        "remove_gt_topk": remaining_after_topk_removal,
    }

    print()
    print("Random Forest feature subset tests:")

    for subset_name, selected_indices in subset_definitions.items():
        metrics = evaluate_feature_subset(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_type=feature_type,
            selected_indices=selected_indices,
            seed=BASE_SEED + table_id,
        )

        subset_rows.append(
            {
                "table_id": table_id,
                "num_classes": num_classes,
                "num_features": num_features,
                "num_active": len(active_indices),
                "subset": subset_name,
                "num_selected": len(selected_indices),
                **metrics,
            }
        )

        print(
            f"{subset_name:20s} | "
            f"features={len(selected_indices):2d} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f}"
        )

    print()
    print("Ground-truth importance:")
    print(np.round(ground_truth_importance, 4))

    print("Mutual information:")
    print(np.round(mi_importance, 4))

    return prediction_rows, importance_rows, subset_rows


# ============================================================
# Main
# ============================================================

def main():
    warnings.filterwarnings("ignore")

    dataset = SyntheticTaskDataset(
        num_tasks=NUM_TABLES,
        task_factory=MixedSCMTask,
        task_kind="classification",

        # Each table independently samples 2, 3, or 4 classes.
        min_classes=2,
        max_classes=4,

        base_seed=BASE_SEED,

        task_kwargs=dict(
            n_min=400,
            n_max=512,
            d_min=8,
            d_max=16,
            test_frac=0.15,

            p_cat=0.3,
            max_cardinality=10,

            p_missing=0.05,
            node_noise_scale=0.05,

            num_roots=5,
            num_layers=4,
            max_nodes_per_layer=12,
            edge_prob=0.3,
            min_parents_per_node=1,
            num_bins=5,

            device=torch.device("cpu"),
        ),
    )

    print("Number of generated tables:", len(dataset))

    all_prediction_rows = []
    all_importance_rows = []
    all_subset_rows = []

    for table_id in range(NUM_TABLES):
        task = dataset[table_id]

        (
            prediction_rows,
            importance_rows,
            subset_rows,
        ) = evaluate_one_table(
            task=task,
            table_id=table_id,
        )

        all_prediction_rows.extend(prediction_rows)
        all_importance_rows.extend(importance_rows)
        all_subset_rows.extend(subset_rows)

    prediction_df = pd.DataFrame(all_prediction_rows)
    importance_df = pd.DataFrame(all_importance_rows)
    subset_df = pd.DataFrame(all_subset_rows)

    prediction_df.to_csv(
        PREDICTION_CSV,
        index=False,
    )

    importance_df.to_csv(
        IMPORTANCE_CSV,
        index=False,
    )

    subset_df.to_csv(
        SUBSET_CSV,
        index=False,
    )

    print()
    print("=" * 100)
    print("PREDICTION SUMMARY")
    print("=" * 100)

    prediction_summary = (
        prediction_df
        .groupby("model")[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "auc",
            ]
        ]
        .agg(["mean", "std"])
    )

    print(prediction_summary)

    print()
    print("=" * 100)
    print("IMPORTANCE SUMMARY")
    print("=" * 100)

    importance_summary = (
        importance_df
        .groupby("method")[
            [
                "spearman",
                "topk_overlap",
            ]
        ]
        .agg(["mean", "std"])
    )

    print(importance_summary)

    print()
    print("=" * 100)
    print("FEATURE SUBSET SUMMARY")
    print("=" * 100)

    subset_summary = (
        subset_df
        .groupby("subset")[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "auc",
            ]
        ]
        .agg(["mean", "std"])
    )

    print(subset_summary)

    print()
    print("Saved:")
    print(f"  {PREDICTION_CSV}")
    print(f"  {IMPORTANCE_CSV}")
    print(f"  {SUBSET_CSV}")


if __name__ == "__main__":
    main()
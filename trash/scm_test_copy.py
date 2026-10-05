# sanity_check/mixed_linear_validation.py

from __future__ import annotations

import time
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# 修改成你实际的文件路径
# ============================================================================

from src.data.linear_task import MixedLinearTask


# ============================================================================
# Configuration
# ============================================================================

NUM_TABLES = 25
BASE_SEED = 0

MIN_CLASSES = 2
MAX_CLASSES = 4

DEVICE = torch.device("cpu")

PRINT_EVERY_TABLE = True
RUN_SUBSET_CHECKS = True

N_ESTIMATORS = 300

TABLE_CSV = "mixed_linear_table_diagnostics.csv"
MODEL_CSV = "mixed_linear_model_metrics.csv"
SUBSET_CSV = "mixed_linear_subset_metrics.csv"
FEATURE_CSV = "mixed_linear_feature_diagnostics.csv"


# ============================================================================
# Generator configuration
# ============================================================================

TASK_KWARGS = dict(
    n_min=400,
    n_max=512,
    d_min=8,
    d_max=16,
    test_frac=0.15,

    p_categorical=0.30,
    max_cardinality=10,

    p_active=0.50,
    p_missing=0.05,
    noise_level=0.10,

    device=DEVICE,
)


# ============================================================================
# Generic helpers
# ============================================================================

def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def unpack_task(task):
    """
    Support:

    1. A five-element tuple:
       X_train, y_train, X_test, y_test, info

    2. A GenerateTask instance exposing those attributes.
    """

    if isinstance(task, (tuple, list)):
        if len(task) != 5:
            raise ValueError(
                "Expected five task outputs: "
                "X_train, y_train, X_test, y_test, info."
            )

        return task

    required = (
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "info",
    )

    missing = [
        name
        for name in required
        if not hasattr(task, name)
    ]

    if missing:
        raise TypeError(
            f"Task object is missing attributes: {missing}"
        )

    return (
        task.X_train,
        task.y_train,
        task.X_test,
        task.y_test,
        task.info,
    )


def make_one_hot_encoder():
    """
    Compatible with older and newer scikit-learn versions.
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


def format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"

    return f"{value:.4f}"


# ============================================================================
# Task construction
# ============================================================================

def build_task(
    table_id: int,
    num_classes: int,
) -> MixedLinearTask:
    return MixedLinearTask(
        num_classes=num_classes,

        dag_seed=BASE_SEED + table_id,
        aleatoric_seed=100_000 + BASE_SEED + table_id,
        x_seed=200_000 + BASE_SEED + table_id,

        **TASK_KWARGS,
    )


# ============================================================================
# Data validation
# ============================================================================

def validate_table(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    info: dict,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warning_messages: list[str] = []

    if X_train.ndim != 2:
        errors.append(
            f"X_train must be 2D, got shape {X_train.shape}."
        )

    if X_test.ndim != 2:
        errors.append(
            f"X_test must be 2D, got shape {X_test.shape}."
        )

    if y_train.ndim != 1:
        errors.append(
            f"y_train must be 1D, got shape {y_train.shape}."
        )

    if y_test.ndim != 1:
        errors.append(
            f"y_test must be 1D, got shape {y_test.shape}."
        )

    if X_train.shape[0] != y_train.shape[0]:
        errors.append(
            "X_train and y_train have different numbers of rows."
        )

    if X_test.shape[0] != y_test.shape[0]:
        errors.append(
            "X_test and y_test have different numbers of rows."
        )

    if X_train.shape[1] != X_test.shape[1]:
        errors.append(
            "X_train and X_test have different feature counts."
        )

    d = X_train.shape[1]

    feature_type = to_numpy(
        info["feature_type"]
    ).astype(int)

    cardinality = to_numpy(
        info["cardinality"]
    ).astype(int)

    sampled_active = to_numpy(
        info["sampled_active"]
    ).astype(float)

    is_active = to_numpy(
        info["is_active"]
    ).astype(float)

    feature_strength = to_numpy(
        info["feature_strength"]
    ).astype(float)

    importance_ratio = to_numpy(
        info["importance_ratio"]
    ).astype(float)

    metadata_lengths = {
        "feature_type": len(feature_type),
        "cardinality": len(cardinality),
        "sampled_active": len(sampled_active),
        "is_active": len(is_active),
        "feature_strength": len(feature_strength),
        "importance_ratio": len(importance_ratio),
    }

    for name, length in metadata_lengths.items():
        if length != d:
            errors.append(
                f"{name} has length {length}; expected {d}."
            )

    X_full = np.concatenate(
        [X_train, X_test],
        axis=0,
    )

    y_full = np.concatenate(
        [y_train, y_test],
        axis=0,
    )

    if np.isinf(X_full).any():
        errors.append(
            "X contains positive or negative infinity."
        )

    if not np.isfinite(y_full).all():
        errors.append(
            "y contains non-finite values."
        )

    classes = np.unique(y_full)

    expected_classes = np.arange(
        len(classes)
    )

    if not np.array_equal(
        classes,
        expected_classes,
    ):
        warning_messages.append(
            f"Class labels are {classes.tolist()}, "
            f"not {expected_classes.tolist()}."
        )

    if not np.array_equal(
        sampled_active > 0.5,
        is_active > 0.5,
    ):
        mismatch_count = int(
            np.sum(
                (sampled_active > 0.5)
                != (is_active > 0.5)
            )
        )

        warning_messages.append(
            f"sampled_active and is_active differ "
            f"for {mismatch_count} features."
        )

    inactive_strength = feature_strength[
        sampled_active <= 0.5
    ]

    if (
        inactive_strength.size > 0
        and np.max(
            np.abs(inactive_strength)
        ) > 1e-7
    ):
        errors.append(
            "At least one sampled-inactive feature has nonzero strength."
        )

    active_strength = feature_strength[
        sampled_active > 0.5
    ]

    if (
        active_strength.size > 0
        and np.any(active_strength <= 1e-8)
    ):
        warning_messages.append(
            "At least one sampled-active feature has near-zero strength."
        )

    strength_sum = float(
        feature_strength.sum()
    )

    importance_sum = float(
        importance_ratio.sum()
    )

    if strength_sum > 1e-12:
        if not np.isclose(
            importance_sum,
            1.0,
            atol=1e-5,
        ):
            errors.append(
                f"importance_ratio sums to {importance_sum:.8f}, expected 1."
            )

        expected_ratio = (
            feature_strength
            / strength_sum
        )

        if not np.allclose(
            importance_ratio,
            expected_ratio,
            atol=1e-6,
            rtol=1e-5,
        ):
            errors.append(
                "importance_ratio does not match normalized feature_strength."
            )

    for col in range(d):
        column = X_full[:, col]
        valid = column[
            ~np.isnan(column)
        ]

        if valid.size == 0:
            errors.append(
                f"Feature {col} is entirely missing."
            )
            continue

        if feature_type[col] == MixedLinearTask.CONTINUOUS:
            if cardinality[col] != 0:
                errors.append(
                    f"Continuous feature {col} has "
                    f"cardinality={cardinality[col]}."
                )

            if float(np.std(valid)) <= 1e-8:
                warning_messages.append(
                    f"Continuous feature {col} is nearly constant."
                )

        elif feature_type[col] == MixedLinearTask.CATEGORICAL:
            k = int(
                cardinality[col]
            )

            if k < 2:
                errors.append(
                    f"Categorical feature {col} has invalid K={k}."
                )
                continue

            rounded = np.round(valid)

            if not np.allclose(
                valid,
                rounded,
                atol=1e-6,
            ):
                errors.append(
                    f"Categorical feature {col} contains "
                    "non-integer values."
                )

            integer_values = rounded.astype(int)

            if integer_values.min() < 0:
                errors.append(
                    f"Categorical feature {col} contains negative ids."
                )

            if integer_values.max() >= k:
                errors.append(
                    f"Categorical feature {col} contains id "
                    f"{integer_values.max()} but K={k}."
                )

            observed_k = len(
                np.unique(integer_values)
            )

            if observed_k < 2:
                warning_messages.append(
                    f"Categorical feature {col} has only "
                    f"{observed_k} observed category."
                )

        else:
            errors.append(
                f"Feature {col} has unknown feature_type "
                f"{feature_type[col]}."
            )

    return errors, warning_messages


# ============================================================================
# Preprocessing
# ============================================================================

def make_preprocessor(
    feature_type: np.ndarray,
) -> ColumnTransformer:
    feature_type = np.asarray(
        feature_type,
        dtype=int,
    )

    continuous_indices = np.where(
        feature_type == MixedLinearTask.CONTINUOUS
    )[0].tolist()

    categorical_indices = np.where(
        feature_type == MixedLinearTask.CATEGORICAL
    )[0].tolist()

    transformers = []

    if continuous_indices:
        continuous_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
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
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
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
        raise ValueError(
            "No usable features were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================================
# Models
# ============================================================================

def make_models(
    feature_type: np.ndarray,
    seed: int,
):
    return {
        "dummy_prior": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(feature_type),
                ),
                (
                    "classifier",
                    DummyClassifier(
                        strategy="prior",
                        random_state=seed,
                    ),
                ),
            ]
        ),

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
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=2,
                        max_features="sqrt",
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
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# ============================================================================
# Metrics
# ============================================================================

def compute_auc(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    try:
        probabilities = model.predict_proba(
            X_test
        )

        model_classes = np.asarray(
            model.classes_
        )

        if len(model_classes) == 2:
            positive_class = model_classes[1]

            binary_target = (
                y_test == positive_class
            ).astype(int)

            if np.unique(binary_target).size < 2:
                return float("nan")

            return float(
                roc_auc_score(
                    binary_target,
                    probabilities[:, 1],
                )
            )

        if np.unique(y_test).size != len(model_classes):
            return float("nan")

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
        return float("nan")


def compute_logloss(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    try:
        probabilities = model.predict_proba(
            X_test
        )

        return float(
            log_loss(
                y_test,
                probabilities,
                labels=np.asarray(model.classes_),
            )
        )

    except (AttributeError, ValueError):
        return float("nan")


def fit_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(
        X_test
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                prediction,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                prediction,
            )
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
        "logloss": compute_logloss(
            model,
            X_test,
            y_test,
        ),
    }


# ============================================================================
# Subset evaluation
# ============================================================================

def empty_metrics() -> dict[str, float]:
    return {
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "auc": float("nan"),
        "logloss": float("nan"),
    }


def evaluate_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_type: np.ndarray,
    selected_indices: np.ndarray,
    seed: int,
) -> dict[str, float]:
    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    if selected_indices.size == 0:
        return empty_metrics()

    subset_feature_type = feature_type[
        selected_indices
    ]

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    subset_feature_type
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return fit_and_evaluate(
        model=model,
        X_train=X_train[:, selected_indices],
        y_train=y_train,
        X_test=X_test[:, selected_indices],
        y_test=y_test,
    )


# ============================================================================
# Evaluate one table
# ============================================================================

def evaluate_one_table(
    task,
    table_id: int,
):
    (
        X_train_t,
        y_train_t,
        X_test_t,
        y_test_t,
        info,
    ) = unpack_task(task)

    X_train = to_numpy(
        X_train_t
    ).astype(float)

    X_test = to_numpy(
        X_test_t
    ).astype(float)

    y_train = to_numpy(
        y_train_t
    ).astype(int)

    y_test = to_numpy(
        y_test_t
    ).astype(int)

    X_full = np.concatenate(
        [X_train, X_test],
        axis=0,
    )

    y_full = np.concatenate(
        [y_train, y_test],
        axis=0,
    )

    feature_type = to_numpy(
        info["feature_type"]
    ).astype(int)

    cardinality = to_numpy(
        info["cardinality"]
    ).astype(int)

    sampled_active = (
        to_numpy(
            info["sampled_active"]
        ).astype(float)
        > 0.5
    )

    is_active = (
        to_numpy(
            info["is_active"]
        ).astype(float)
        > 0.5
    )

    feature_strength = to_numpy(
        info["feature_strength"]
    ).astype(float)

    importance_ratio = to_numpy(
        info["importance_ratio"]
    ).astype(float)

    (
        validation_errors,
        validation_warnings,
    ) = validate_table(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        info=info,
    )

    d = X_train.shape[1]

    num_classes = int(
        np.unique(y_full).size
    )

    continuous_indices = np.where(
        feature_type == MixedLinearTask.CONTINUOUS
    )[0]

    categorical_indices = np.where(
        feature_type == MixedLinearTask.CATEGORICAL
    )[0]

    active_indices = np.where(
        sampled_active
    )[0]

    inactive_indices = np.where(
        ~sampled_active
    )[0]

    strongest_order = np.argsort(
        -feature_strength
    )

    strongest_feature_index = int(
        strongest_order[0]
    )

    top_quarter_count = max(
        1,
        int(np.ceil(0.25 * d)),
    )

    top_strength_indices = strongest_order[
        :top_quarter_count
    ]

    remaining_without_strongest = np.setdiff1d(
        np.arange(d),
        np.asarray(
            [strongest_feature_index]
        ),
    )

    remaining_without_top = np.setdiff1d(
        np.arange(d),
        top_strength_indices,
    )

    missing_rate = float(
        np.isnan(X_full).mean()
    )

    continuous_count = int(
        len(continuous_indices)
    )

    categorical_count = int(
        len(categorical_indices)
    )

    active_count = int(
        len(active_indices)
    )

    inactive_count = int(
        len(inactive_indices)
    )

    class_values, class_counts = np.unique(
        y_full,
        return_counts=True,
    )

    class_count_dict = {
        int(class_id): int(count)
        for class_id, count
        in zip(class_values, class_counts)
    }

    table_row = {
        "table_id": table_id,
        "n_total": X_full.shape[0],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "num_features": d,
        "num_classes": num_classes,

        "continuous_features": continuous_count,
        "categorical_features": categorical_count,
        "categorical_ratio": categorical_count / d,

        "active_features": active_count,
        "inactive_features": inactive_count,
        "active_ratio": active_count / d,

        "missing_rate": missing_rate,

        "mean_active_strength": (
            float(
                feature_strength[
                    sampled_active
                ].mean()
            )
            if sampled_active.any()
            else float("nan")
        ),

        "max_strength": float(
            feature_strength.max()
        ),

        "min_positive_strength": (
            float(
                feature_strength[
                    feature_strength > 1e-8
                ].min()
            )
            if np.any(
                feature_strength > 1e-8
            )
            else float("nan")
        ),

        "strongest_feature_index": strongest_feature_index,
        "strongest_feature_strength": float(
            feature_strength[
                strongest_feature_index
            ]
        ),

        "importance_ratio_sum": float(
            importance_ratio.sum()
        ),

        "num_validation_errors": len(
            validation_errors
        ),

        "num_validation_warnings": len(
            validation_warnings
        ),

        "target_counts": "|".join(
            f"{key}:{value}"
            for key, value
            in class_count_dict.items()
        ),
    }

    feature_rows = []

    for col in range(d):
        column = X_full[:, col]

        valid = column[
            ~np.isnan(column)
        ]

        feature_row = {
            "table_id": table_id,
            "feature_index": col,

            "feature_type": (
                "categorical"
                if feature_type[col]
                == MixedLinearTask.CATEGORICAL
                else "continuous"
            ),

            "cardinality": int(
                cardinality[col]
            ),

            "sampled_active": bool(
                sampled_active[col]
            ),

            "is_active": bool(
                is_active[col]
            ),

            "feature_strength": float(
                feature_strength[col]
            ),

            "importance_ratio": float(
                importance_ratio[col]
            ),

            "missing_rate": float(
                np.isnan(column).mean()
            ),

            "mean": (
                float(valid.mean())
                if valid.size > 0
                else float("nan")
            ),

            "std": (
                float(valid.std())
                if valid.size > 0
                else float("nan")
            ),

            "min": (
                float(valid.min())
                if valid.size > 0
                else float("nan")
            ),

            "max": (
                float(valid.max())
                if valid.size > 0
                else float("nan")
            ),
        }

        if (
            feature_type[col]
            == MixedLinearTask.CATEGORICAL
        ):
            k = int(
                cardinality[col]
            )

            counts = np.bincount(
                valid.astype(int),
                minlength=k,
            )

            feature_row[
                "observed_category_count"
            ] = int(
                np.sum(counts > 0)
            )

            feature_row[
                "smallest_category_count"
            ] = int(
                counts.min()
            )

            feature_row[
                "smallest_category_fraction"
            ] = float(
                counts.min()
                / counts.sum()
            )

            feature_row[
                "category_counts"
            ] = "|".join(
                str(int(value))
                for value in counts
            )

        else:
            feature_row[
                "observed_category_count"
            ] = np.nan

            feature_row[
                "smallest_category_count"
            ] = np.nan

            feature_row[
                "smallest_category_fraction"
            ] = np.nan

            feature_row[
                "category_counts"
            ] = ""

        feature_rows.append(
            feature_row
        )

    if PRINT_EVERY_TABLE:
        print()
        print("=" * 118)
        print(
            f"TABLE {table_id:03d} | "
            f"classes={num_classes} | "
            f"features={d} | "
            f"continuous={continuous_count} | "
            f"categorical={categorical_count} | "
            f"active={active_count} | "
            f"missing={missing_rate:.2%}"
        )
        print("=" * 118)

        print(
            "Target class counts:",
            class_count_dict,
        )

        print(
            "Feature strengths:",
            np.round(
                feature_strength,
                4,
            ).tolist(),
        )

        print(
            "Active mask:",
            sampled_active.astype(int).tolist(),
        )

        print(
            "Importance ratio:",
            np.round(
                importance_ratio,
                4,
            ).tolist(),
        )

        if validation_errors:
            print()
            print("VALIDATION ERRORS:")

            for message in validation_errors:
                print(
                    f"  ERROR: {message}"
                )

        if validation_warnings:
            print()
            print("VALIDATION WARNINGS:")

            for message in validation_warnings:
                print(
                    f"  WARNING: {message}"
                )

    model_rows = []

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

        model_rows.append(
            {
                "table_id": table_id,
                "num_classes": num_classes,
                "num_features": d,
                "num_active": active_count,
                "model": model_name,
                **metrics,
            }
        )

        if PRINT_EVERY_TABLE:
            print(
                f"{model_name:24s} | "
                f"acc={format_float(metrics['accuracy'])} | "
                f"bal_acc={format_float(metrics['balanced_accuracy'])} | "
                f"f1={format_float(metrics['macro_f1'])} | "
                f"auc={format_float(metrics['auc'])} | "
                f"logloss={format_float(metrics['logloss'])}"
            )

    subset_rows = []

    if RUN_SUBSET_CHECKS:
        subset_definitions = {
            "all_features": np.arange(
                d,
                dtype=int,
            ),

            "active_only": active_indices,

            "inactive_only": inactive_indices,

            "continuous_only": continuous_indices,

            "categorical_only": categorical_indices,

            "remove_strongest_feature": (
                remaining_without_strongest
            ),

            "top_strength_quarter": (
                top_strength_indices
            ),

            "remove_top_strength_quarter": (
                remaining_without_top
            ),
        }

        if PRINT_EVERY_TABLE:
            print()
            print(
                "Random Forest feature-subset checks:"
            )

        for subset_name, selected_indices in subset_definitions.items():
            metrics = evaluate_subset(
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
                    "num_features": d,
                    "num_active": active_count,
                    "subset": subset_name,
                    "num_selected": int(
                        len(selected_indices)
                    ),
                    **metrics,
                }
            )

            if PRINT_EVERY_TABLE:
                print(
                    f"{subset_name:30s} | "
                    f"features={len(selected_indices):2d} | "
                    f"bal_acc="
                    f"{format_float(metrics['balanced_accuracy'])} | "
                    f"auc={format_float(metrics['auc'])}"
                )

    return (
        table_row,
        model_rows,
        subset_rows,
        feature_rows,
    )


# ============================================================================
# Summary functions
# ============================================================================

def print_generator_summary(
    table_df: pd.DataFrame,
):
    print()
    print("=" * 118)
    print("GENERATOR SUMMARY")
    print("=" * 118)

    columns = [
        "n_total",
        "num_features",
        "continuous_features",
        "categorical_features",
        "categorical_ratio",
        "active_features",
        "inactive_features",
        "active_ratio",
        "missing_rate",
        "mean_active_strength",
        "max_strength",
        "num_validation_errors",
        "num_validation_warnings",
    ]

    summary = table_df[
        columns
    ].agg(
        [
            "mean",
            "std",
            "median",
            "min",
            "max",
        ]
    ).T

    print(summary)

    error_tables = int(
        (
            table_df[
                "num_validation_errors"
            ]
            > 0
        ).sum()
    )

    warning_tables = int(
        (
            table_df[
                "num_validation_warnings"
            ]
            > 0
        ).sum()
    )

    print()
    print(
        f"Tables with validation errors: "
        f"{error_tables}/{len(table_df)}"
    )

    print(
        f"Tables with validation warnings: "
        f"{warning_tables}/{len(table_df)}"
    )


def print_model_summary(
    model_df: pd.DataFrame,
):
    print()
    print("=" * 118)
    print("MODEL SUMMARY")
    print("=" * 118)

    summary = (
        model_df
        .groupby("model")[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "auc",
                "logloss",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(summary)

    balanced_pivot = model_df.pivot(
        index="table_id",
        columns="model",
        values="balanced_accuracy",
    )

    if "dummy_prior" not in balanced_pivot.columns:
        return

    for model_name in (
        "logistic_regression",
        "random_forest",
        "extra_trees",
    ):
        if model_name not in balanced_pivot.columns:
            continue

        lift = (
            balanced_pivot[model_name]
            - balanced_pivot["dummy_prior"]
        )

        print()
        print(
            f"{model_name} versus dummy:"
        )

        print(
            f"  mean balanced-accuracy lift "
            f"= {lift.mean():.4f}"
        )

        print(
            f"  median lift                 "
            f"= {lift.median():.4f}"
        )

        print(
            f"  minimum lift                "
            f"= {lift.min():.4f}"
        )

        print(
            f"  maximum lift                "
            f"= {lift.max():.4f}"
        )

        print(
            f"  beats dummy                 "
            f"= {(lift > 0).mean():.2%}"
        )

        print(
            f"  beats dummy by >= 0.05      "
            f"= {(lift >= 0.05).mean():.2%}"
        )

        print(
            f"  beats dummy by >= 0.10      "
            f"= {(lift >= 0.10).mean():.2%}"
        )


def print_subset_summary(
    subset_df: pd.DataFrame,
):
    if subset_df.empty:
        return

    print()
    print("=" * 118)
    print("FEATURE SUBSET SUMMARY")
    print("=" * 118)

    summary = (
        subset_df
        .groupby("subset")[
            [
                "num_selected",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "auc",
                "logloss",
            ]
        ]
        .agg(
            [
                "mean",
                "std",
                "median",
            ]
        )
    )

    print(summary)

    pivot = subset_df.pivot(
        index="table_id",
        columns="subset",
        values="balanced_accuracy",
    )

    if {
        "active_only",
        "inactive_only",
    }.issubset(pivot.columns):
        difference = (
            pivot["active_only"]
            - pivot["inactive_only"]
        )

        comparable = difference.dropna()

        print()
        print(
            "Active-only minus inactive-only balanced accuracy:"
        )

        print(
            f"  mean   = {comparable.mean():.4f}"
        )

        print(
            f"  median = {comparable.median():.4f}"
        )

        print(
            "  active-only performs better on "
            f"{(comparable > 0).mean():.2%} "
            "of comparable tables"
        )

    if {
        "all_features",
        "remove_strongest_feature",
    }.issubset(pivot.columns):
        drop = (
            pivot["all_features"]
            - pivot["remove_strongest_feature"]
        )

        print()
        print(
            "Balanced-accuracy drop after removing strongest feature:"
        )

        print(
            f"  mean   = {drop.mean():.4f}"
        )

        print(
            f"  median = {drop.median():.4f}"
        )

        print(
            "  performance drops on "
            f"{(drop > 0).mean():.2%} of tables"
        )

    if {
        "all_features",
        "remove_top_strength_quarter",
    }.issubset(pivot.columns):
        drop = (
            pivot["all_features"]
            - pivot["remove_top_strength_quarter"]
        )

        print()
        print(
            "Balanced-accuracy drop after removing top-strength quarter:"
        )

        print(
            f"  mean   = {drop.mean():.4f}"
        )

        print(
            f"  median = {drop.median():.4f}"
        )

        print(
            "  performance drops on "
            f"{(drop > 0).mean():.2%} of tables"
        )


def print_feature_summary(
    feature_df: pd.DataFrame,
):
    print()
    print("=" * 118)
    print("FEATURE SUMMARY")
    print("=" * 118)

    print(
        feature_df.groupby(
            [
                "feature_type",
                "sampled_active",
            ]
        ).size()
    )

    print()
    print(
        "Feature strength by active status:"
    )

    print(
        feature_df.groupby(
            "sampled_active"
        )[
            "feature_strength"
        ].describe()
    )

    print()
    print(
        "Feature strength by feature type:"
    )

    print(
        feature_df.groupby(
            "feature_type"
        )[
            "feature_strength"
        ].describe()
    )

    categorical = feature_df[
        feature_df["feature_type"]
        == "categorical"
    ]

    if not categorical.empty:
        print()
        print(
            "Categorical cardinality distribution:"
        )

        print(
            categorical[
                "cardinality"
            ].value_counts().sort_index()
        )

        print()
        print(
            "Smallest observed category fraction:"
        )

        print(
            categorical[
                "smallest_category_fraction"
            ].describe()
        )


# ============================================================================
# Main
# ============================================================================

def main():
    warnings.filterwarnings(
        "ignore"
    )

    table_rows = []
    model_rows = []
    subset_rows = []
    feature_rows = []

    generation_times = []

    class_generator = torch.Generator(
        device="cpu"
    )

    class_generator.manual_seed(
        BASE_SEED + 999_999
    )

    for table_id in range(
        NUM_TABLES
    ):
        num_classes = int(
            torch.randint(
                MIN_CLASSES,
                MAX_CLASSES + 1,
                (1,),
                generator=class_generator,
            ).item()
        )

        start = time.perf_counter()

        task = build_task(
            table_id=table_id,
            num_classes=num_classes,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        generation_times.append(
            elapsed
        )

        (
            table_row,
            current_model_rows,
            current_subset_rows,
            current_feature_rows,
        ) = evaluate_one_table(
            task=task,
            table_id=table_id,
        )

        table_row[
            "generation_time_seconds"
        ] = elapsed

        table_rows.append(
            table_row
        )

        model_rows.extend(
            current_model_rows
        )

        subset_rows.extend(
            current_subset_rows
        )

        feature_rows.extend(
            current_feature_rows
        )

    table_df = pd.DataFrame(
        table_rows
    )

    model_df = pd.DataFrame(
        model_rows
    )

    subset_df = pd.DataFrame(
        subset_rows
    )

    feature_df = pd.DataFrame(
        feature_rows
    )

    table_df.to_csv(
        TABLE_CSV,
        index=False,
    )

    model_df.to_csv(
        MODEL_CSV,
        index=False,
    )

    subset_df.to_csv(
        SUBSET_CSV,
        index=False,
    )

    feature_df.to_csv(
        FEATURE_CSV,
        index=False,
    )

    print_generator_summary(
        table_df
    )

    print_model_summary(
        model_df
    )

    print_subset_summary(
        subset_df
    )

    print_feature_summary(
        feature_df
    )

    times = np.asarray(
        generation_times,
        dtype=float,
    )

    print()
    print("=" * 118)
    print("GENERATION TIME")
    print("=" * 118)

    print(
        f"Mean   : {times.mean():.4f}s"
    )

    print(
        f"Median : {np.median(times):.4f}s"
    )

    print(
        f"Min    : {times.min():.4f}s"
    )

    print(
        f"Max    : {times.max():.4f}s"
    )

    print()
    print("Saved:")
    print(f"  {TABLE_CSV}")
    print(f"  {MODEL_CSV}")
    print(f"  {SUBSET_CSV}")
    print(f"  {FEATURE_CSV}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

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

from Trash.scm_task_v7 import MixedLatentSCMTask


# =============================================================================
# Configuration
# =============================================================================

NUM_TABLES = 50
BASE_SEED = 0

MIN_CLASSES = 2
MAX_CLASSES = 4

N_ESTIMATORS = 300

OUTPUT_DIR = Path("scm_v7_sanity_results")

TABLE_CSV = OUTPUT_DIR / "table_summary.csv"
FEATURE_CSV = OUTPUT_DIR / "feature_summary.csv"
PREDICTION_CSV = OUTPUT_DIR / "prediction_metrics.csv"
SUBSET_CSV = OUTPUT_DIR / "feature_subset_metrics.csv"
MECHANISM_CSV = OUTPUT_DIR / "mechanism_summary.csv"


# =============================================================================
# Basic utilities
# =============================================================================


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def get_task_outputs(task):
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
        raise AttributeError(
            "Task object is missing required attributes: "
            f"{missing}"
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
    Support both older and newer sklearn versions.
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


def safe_mean(values) -> float:
    array = np.asarray(values, dtype=float)

    if array.size == 0:
        return float("nan")

    return float(np.nanmean(array))


def safe_std(values) -> float:
    array = np.asarray(values, dtype=float)

    if array.size <= 1:
        return float("nan")

    return float(np.nanstd(array, ddof=1))


# =============================================================================
# Task construction
# =============================================================================


def build_task(
    device: torch.device,
    num_classes: int,
    dag_seed: int,
    aleatoric_seed: int,
    x_seed: int,
) -> MixedLatentSCMTask:
    """
    Shared generator configuration.

    Change generator parameters here when running ablations.
    """
    return MixedLatentSCMTask(
        num_classes=num_classes,

        n_min=400,
        n_max=512,
        d_min=8,
        d_max=16,

        test_frac=0.15,
        p_missing=0.05,

        num_roots=3,
        num_layers=5,
        max_nodes_per_layer=8,
        latent_dim=6,

        latent_noise_scale=0.05,
        observation_noise_scale=0.05,

        edge_beta_alpha=2.0,
        edge_beta_beta=5.0,
        edge_prob_min=0.05,
        edge_prob_max=0.95,
        min_parents_per_node=1,

        root_prior_probs=(
            0.45,  # Gaussian
            0.20,  # Uniform
            0.15,  # Heavy-tailed
            0.05,  # Skewed
            0.15,  # Mixture
        ),

        root_mixture_component_probs=(
            0.40,  # M=2
            0.30,  # M=3
            0.18,  # M=4
            0.08,  # M=5
            0.04,  # M=6
        ),

        root_mixture_separation_min=1.5,
        root_mixture_separation_max=3.0,
        root_mixture_scale_min=0.40,
        root_mixture_scale_max=0.90,

        dominant_parent_prob=0.40,
        dominant_parent_weight=0.75,

        # continuous / prototype / threshold-binning
        observation_type_probs=(
            0.50,
            0.25,
            0.25,
        ),

        categorical_cardinalities=(
            2,
            3,
            4,
            5,
            6,
        ),

        categorical_cardinality_probs=(
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),

        min_samples_per_category=8,
        min_component_weight=0.05,

        prototype_max_attempts=8,
        prototype_min_separation=1.0,

        binning_jitter=0.20,

        linear_activation_prob=0.60,
        small_mlp_prob=0.25,
        soft_tree_prob=0.15,
        soft_tree_depth=2,
        soft_tree_temperature=0.5,

        dag_seed=dag_seed,
        aleatoric_seed=aleatoric_seed,
        x_seed=x_seed,

        device=device,
    )


def sample_num_classes(
    table_id: int,
) -> int:
    """
    Deterministically sample 2, 3, or 4 classes per table.
    """
    generator = np.random.default_rng(
        BASE_SEED + 1_000_000 + table_id
    )

    return int(
        generator.integers(
            MIN_CLASSES,
            MAX_CLASSES + 1,
        )
    )


# =============================================================================
# Generator validation
# =============================================================================


def validate_task_output(
    task,
    table_id: int,
) -> None:
    """
    Hard checks for malformed outputs.

    These checks raise immediately instead of silently allowing invalid tables.
    """
    (
        X_train,
        y_train,
        X_test,
        y_test,
        info,
    ) = get_task_outputs(task)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise AssertionError(
            f"Table {table_id}: X must be two-dimensional."
        )

    if y_train.ndim != 1 or y_test.ndim != 1:
        raise AssertionError(
            f"Table {table_id}: y must be one-dimensional."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise AssertionError(
            f"Table {table_id}: train/test feature counts differ."
        )

    if X_train.shape[0] != y_train.shape[0]:
        raise AssertionError(
            f"Table {table_id}: X_train/y_train sizes differ."
        )

    if X_test.shape[0] != y_test.shape[0]:
        raise AssertionError(
            f"Table {table_id}: X_test/y_test sizes differ."
        )

    required_info_keys = (
        "feature_type",
        "cardinality",
        "feature_observation_type_ids",
        "feature_observation_type_names",
        "feature_observation_quality",
        "feature_prototypes",
        "feature_thresholds",
        "feature_projections",
        "feature_ids",
        "target_id",
        "task_edge_prob",
        "latent_dim",
        "root_prior_types",
        "root_prior_type_ids",
        "root_mixture_components",
    )

    missing_info_keys = [
        key
        for key in required_info_keys
        if key not in info
    ]

    if missing_info_keys:
        raise KeyError(
            f"Table {table_id}: info is missing keys "
            f"{missing_info_keys}"
        )

    d = int(X_train.shape[1])

    feature_type = info["feature_type"]
    cardinality = info["cardinality"]
    mechanism_names = info[
        "feature_observation_type_names"
    ]

    if int(feature_type.numel()) != d:
        raise AssertionError(
            f"Table {table_id}: feature_type length does not match d."
        )

    if int(cardinality.numel()) != d:
        raise AssertionError(
            f"Table {table_id}: cardinality length does not match d."
        )

    if len(mechanism_names) != d:
        raise AssertionError(
            f"Table {table_id}: mechanism-name length does not match d."
        )

    allowed_names = {
        "continuous_projection",
        "prototype_discretization",
        "threshold_binning",
        "continuous_fallback_from_prototype",
        "continuous_fallback_from_binning",
    }

    unknown_names = set(mechanism_names) - allowed_names

    if unknown_names:
        raise AssertionError(
            f"Table {table_id}: unknown observation mechanisms: "
            f"{unknown_names}"
        )

    feature_type_cpu = feature_type.detach().cpu()
    cardinality_cpu = cardinality.detach().cpu()

    for col, mechanism_name in enumerate(mechanism_names):
        is_categorical = (
            int(feature_type_cpu[col].item())
            == MixedLatentSCMTask.CATEGORICAL
        )

        k = int(
            cardinality_cpu[col].item()
        )

        if is_categorical:
            if k < 2:
                raise AssertionError(
                    f"Table {table_id}, feature {col}: "
                    f"categorical feature has K={k}."
                )

            if mechanism_name not in {
                "prototype_discretization",
                "threshold_binning",
            }:
                raise AssertionError(
                    f"Table {table_id}, feature {col}: "
                    f"categorical feature has mechanism "
                    f"{mechanism_name}."
                )
        else:
            if k != 0:
                raise AssertionError(
                    f"Table {table_id}, feature {col}: "
                    f"continuous feature has K={k}."
                )

    all_y = torch.cat(
        [y_train, y_test],
        dim=0,
    ).long()

    observed_classes = torch.unique(
        all_y
    )

    if int(observed_classes.numel()) != int(task.num_classes):
        raise AssertionError(
            f"Table {table_id}: expected {task.num_classes} classes, "
            f"observed {observed_classes.tolist()}."
        )


# =============================================================================
# Preprocessing
# =============================================================================


def make_preprocessor(
    feature_type: np.ndarray,
) -> ColumnTransformer:
    feature_type = np.asarray(
        feature_type,
        dtype=int,
    )

    continuous_indices = np.where(
        feature_type
        == MixedLatentSCMTask.CONTINUOUS
    )[0].tolist()

    categorical_indices = np.where(
        feature_type
        == MixedLatentSCMTask.CATEGORICAL
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
            "No usable features were selected."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# =============================================================================
# Models
# =============================================================================


def make_models(
    feature_type: np.ndarray,
    seed: int,
):
    return {
        "dummy_prior": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(
                        feature_type
                    ),
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
                    make_preprocessor(
                        feature_type
                    ),
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
                    make_preprocessor(
                        feature_type
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
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
                    make_preprocessor(
                        feature_type
                    ),
                ),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=N_ESTIMATORS,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# =============================================================================
# Prediction metrics
# =============================================================================


def get_model_classes(
    model,
) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(
            model.classes_
        )

    if isinstance(model, Pipeline):
        classifier = model.named_steps[
            "classifier"
        ]

        if hasattr(
            classifier,
            "classes_",
        ):
            return np.asarray(
                classifier.classes_
            )

    raise AttributeError(
        "Fitted model does not expose classes_."
    )


def compute_auc(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    try:
        probabilities = model.predict_proba(
            X_test
        )

        model_classes = get_model_classes(
            model
        )

        observed_test_classes = np.unique(
            y_test
        )

        if len(model_classes) == 2:
            if len(observed_test_classes) < 2:
                return float("nan")

            positive_class = model_classes[1]

            binary_target = (
                y_test == positive_class
            ).astype(int)

            return float(
                roc_auc_score(
                    binary_target,
                    probabilities[:, 1],
                )
            )

        if (
            len(observed_test_classes)
            != len(model_classes)
        ):
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

    except (
        AttributeError,
        ValueError,
        IndexError,
    ):
        return float("nan")


def compute_log_loss(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    try:
        probabilities = model.predict_proba(
            X_test
        )

        classes = get_model_classes(
            model
        )

        return float(
            log_loss(
                y_test,
                probabilities,
                labels=classes,
            )
        )

    except (
        AttributeError,
        ValueError,
        IndexError,
    ):
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

        "log_loss": compute_log_loss(
            model,
            X_test,
            y_test,
        ),
    }


# =============================================================================
# Feature-subset tests
# =============================================================================


def build_subset_definitions(
    feature_type: np.ndarray,
    mechanism_names: list[str],
) -> dict[str, np.ndarray]:
    feature_type = np.asarray(
        feature_type,
        dtype=int,
    )

    mechanism_names_array = np.asarray(
        mechanism_names,
        dtype=object,
    )

    all_indices = np.arange(
        len(feature_type)
    )

    continuous_indices = np.where(
        feature_type
        == MixedLatentSCMTask.CONTINUOUS
    )[0]

    categorical_indices = np.where(
        feature_type
        == MixedLatentSCMTask.CATEGORICAL
    )[0]

    projection_indices = np.where(
        mechanism_names_array
        == "continuous_projection"
    )[0]

    prototype_indices = np.where(
        mechanism_names_array
        == "prototype_discretization"
    )[0]

    binning_indices = np.where(
        mechanism_names_array
        == "threshold_binning"
    )[0]

    fallback_indices = np.where(
        np.char.find(
            mechanism_names_array.astype(str),
            "fallback",
        )
        >= 0
    )[0]

    non_prototype_indices = np.where(
        mechanism_names_array
        != "prototype_discretization"
    )[0]

    non_binning_indices = np.where(
        mechanism_names_array
        != "threshold_binning"
    )[0]

    return {
        "all_features": all_indices,
        "continuous_features": continuous_indices,
        "categorical_features": categorical_indices,
        "continuous_projection_only": projection_indices,
        "prototype_only": prototype_indices,
        "binning_only": binning_indices,
        "fallback_only": fallback_indices,
        "remove_prototype": non_prototype_indices,
        "remove_binning": non_binning_indices,
    }


def evaluate_feature_subset(
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
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auc": float("nan"),
            "log_loss": float("nan"),
        }

    X_train_subset = X_train[
        :,
        selected_indices,
    ]

    X_test_subset = X_test[
        :,
        selected_indices,
    ]

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


# =============================================================================
# Per-feature diagnostics
# =============================================================================


def collect_feature_rows(
    table_id: int,
    X_full: np.ndarray,
    info: dict,
) -> list[dict]:
    feature_type = to_numpy(
        info["feature_type"]
    ).astype(int)

    cardinality = to_numpy(
        info["cardinality"]
    ).astype(int)

    quality = to_numpy(
        info["feature_observation_quality"]
    ).astype(float)

    feature_ids = to_numpy(
        info["feature_ids"]
    ).astype(int)

    mechanism_names = info[
        "feature_observation_type_names"
    ]

    rows = []

    for col in range(
        X_full.shape[1]
    ):
        column = X_full[
            :,
            col,
        ]

        valid = column[
            ~np.isnan(column)
        ]

        mechanism_name = mechanism_names[
            col
        ]

        is_categorical = (
            feature_type[col]
            == MixedLatentSCMTask.CATEGORICAL
        )

        row = {
            "table_id": table_id,
            "feature_index": col,
            "source_node_id": int(
                feature_ids[col]
            ),
            "mechanism": mechanism_name,
            "is_categorical": int(
                is_categorical
            ),
            "cardinality": int(
                cardinality[col]
            ),
            "quality_score": float(
                quality[col]
            ),
            "missing_rate": float(
                np.isnan(column).mean()
            ),
            "num_observed": int(
                valid.size
            ),
        }

        if valid.size == 0:
            row.update(
                {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "minimum_category_count": float("nan"),
                    "minimum_category_fraction": float("nan"),
                }
            )

        elif is_categorical:
            k = int(
                cardinality[col]
            )

            counts = np.bincount(
                valid.astype(int),
                minlength=k,
            )

            row.update(
                {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float(
                        valid.min()
                    ),
                    "max": float(
                        valid.max()
                    ),
                    "minimum_category_count": int(
                        counts.min()
                    ),
                    "minimum_category_fraction": float(
                        counts.min()
                        / counts.sum()
                    ),
                    "category_counts": str(
                        counts.tolist()
                    ),
                }
            )

        else:
            row.update(
                {
                    "mean": float(
                        valid.mean()
                    ),
                    "std": float(
                        valid.std()
                    ),
                    "min": float(
                        valid.min()
                    ),
                    "max": float(
                        valid.max()
                    ),
                    "minimum_category_count": float("nan"),
                    "minimum_category_fraction": float("nan"),
                }
            )

        rows.append(
            row
        )

    return rows


# =============================================================================
# Evaluate one table
# =============================================================================


def evaluate_one_table(
    task,
    table_id: int,
):
    validate_task_output(
        task,
        table_id,
    )

    (
        X_train_t,
        y_train_t,
        X_test_t,
        y_test_t,
        info,
    ) = get_task_outputs(
        task
    )

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
        [
            X_train,
            X_test,
        ],
        axis=0,
    )

    y_full = np.concatenate(
        [
            y_train,
            y_test,
        ],
        axis=0,
    )

    feature_type = to_numpy(
        info["feature_type"]
    ).astype(int)

    cardinality = to_numpy(
        info["cardinality"]
    ).astype(int)

    mechanism_names = list(
        info[
            "feature_observation_type_names"
        ]
    )

    num_features = int(
        X_train.shape[1]
    )

    num_classes = int(
        len(
            np.unique(
                y_full
            )
        )
    )

    num_categorical = int(
        (
            feature_type
            == MixedLatentSCMTask.CATEGORICAL
        ).sum()
    )

    num_continuous = (
        num_features
        - num_categorical
    )

    mechanism_counter = Counter(
        mechanism_names
    )

    root_prior_types = list(
        info["root_prior_types"]
    )

    root_mixture_components = to_numpy(
        info["root_mixture_components"]
    ).astype(int)

    table_row = {
        "table_id": table_id,
        "num_samples": int(
            X_full.shape[0]
        ),
        "num_train": int(
            X_train.shape[0]
        ),
        "num_test": int(
            X_test.shape[0]
        ),
        "num_features": num_features,
        "num_classes": num_classes,
        "num_continuous": num_continuous,
        "num_categorical": num_categorical,
        "categorical_ratio": (
            num_categorical
            / num_features
        ),
        "missing_rate": float(
            np.isnan(
                X_full
            ).mean()
        ),
        "task_edge_prob": float(
            to_numpy(
                info["task_edge_prob"]
            ).item()
        ),
        "latent_dim": int(
            to_numpy(
                info["latent_dim"]
            ).item()
        ),
        "root_prior_types": str(
            root_prior_types
        ),
        "root_mixture_components": str(
            root_mixture_components.tolist()
        ),
        "continuous_projection_count": mechanism_counter[
            "continuous_projection"
        ],
        "prototype_count": mechanism_counter[
            "prototype_discretization"
        ],
        "binning_count": mechanism_counter[
            "threshold_binning"
        ],
        "prototype_fallback_count": mechanism_counter[
            "continuous_fallback_from_prototype"
        ],
        "binning_fallback_count": mechanism_counter[
            "continuous_fallback_from_binning"
        ],
    }

    print()
    print("=" * 110)
    print(
        f"TABLE {table_id:03d} | "
        f"classes={num_classes} | "
        f"features={num_features} | "
        f"continuous={num_continuous} | "
        f"categorical={num_categorical} | "
        f"missing={table_row['missing_rate']:.2%}"
    )
    print("=" * 110)

    print(
        "Roots:",
        root_prior_types,
    )

    print(
        "Root mixture components:",
        root_mixture_components.tolist(),
    )

    print(
        "Mechanisms:",
        dict(
            mechanism_counter
        ),
    )

    class_values, class_counts = np.unique(
        y_full,
        return_counts=True,
    )

    print(
        "Target class counts:",
        dict(
            zip(
                class_values.tolist(),
                class_counts.tolist(),
            )
        ),
    )

    prediction_rows = []

    models = make_models(
        feature_type=feature_type,
        seed=BASE_SEED + table_id,
    )

    for model_name, model in models.items():
        start = time.perf_counter()

        metrics = fit_and_evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        prediction_rows.append(
            {
                "table_id": table_id,
                "num_classes": num_classes,
                "num_features": num_features,
                "num_continuous": num_continuous,
                "num_categorical": num_categorical,
                "model": model_name,
                "fit_evaluation_time": elapsed,
                **metrics,
            }
        )

        print(
            f"{model_name:22s} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"f1={metrics['macro_f1']:.4f} | "
            f"auc={metrics['auc']:.4f} | "
            f"logloss={metrics['log_loss']:.4f}"
        )

    subset_rows = []

    subset_definitions = build_subset_definitions(
        feature_type=feature_type,
        mechanism_names=mechanism_names,
    )

    print()
    print(
        "Random Forest feature-subset checks:"
    )

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
                "subset": subset_name,
                "num_selected": int(
                    len(
                        selected_indices
                    )
                ),
                **metrics,
            }
        )

        print(
            f"{subset_name:28s} | "
            f"features={len(selected_indices):2d} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"auc={metrics['auc']:.4f}"
        )

    feature_rows = collect_feature_rows(
        table_id=table_id,
        X_full=X_full,
        info=info,
    )

    return (
        table_row,
        feature_rows,
        prediction_rows,
        subset_rows,
    )


# =============================================================================
# Aggregate mechanism statistics
# =============================================================================


def build_mechanism_summary(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for mechanism_name, group in feature_df.groupby(
        "mechanism"
    ):
        categorical_group = group[
            group["is_categorical"]
            == 1
        ]

        rows.append(
            {
                "mechanism": mechanism_name,
                "feature_count": int(
                    len(group)
                ),
                "categorical_count": int(
                    group[
                        "is_categorical"
                    ].sum()
                ),
                "categorical_rate": float(
                    group[
                        "is_categorical"
                    ].mean()
                ),
                "mean_quality_score": float(
                    group[
                        "quality_score"
                    ].mean()
                ),
                "mean_missing_rate": float(
                    group[
                        "missing_rate"
                    ].mean()
                ),
                "mean_cardinality": (
                    float(
                        categorical_group[
                            "cardinality"
                        ].mean()
                    )
                    if len(
                        categorical_group
                    )
                    > 0
                    else float("nan")
                ),
                "mean_minimum_category_fraction": (
                    float(
                        categorical_group[
                            "minimum_category_fraction"
                        ].mean()
                    )
                    if len(
                        categorical_group
                    )
                    > 0
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Summaries
# =============================================================================


def print_prediction_summary(
    prediction_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 110)
    print("PREDICTION SUMMARY")
    print("=" * 110)

    columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auc",
        "log_loss",
    ]

    summary = (
        prediction_df
        .groupby("model")[
            columns
        ]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        summary
    )


def print_subset_summary(
    subset_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 110)
    print("FEATURE-SUBSET SUMMARY")
    print("=" * 110)

    columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "auc",
        "log_loss",
    ]

    summary = (
        subset_df
        .groupby("subset")[
            columns
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    print(
        summary
    )


def print_generator_summary(
    table_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
) -> None:
    print()
    print("=" * 110)
    print("GENERATOR SUMMARY")
    print("=" * 110)

    print(
        f"Tables: {len(table_df)}"
    )

    print(
        "Mean samples per table: "
        f"{table_df['num_samples'].mean():.2f}"
    )

    print(
        "Mean features per table: "
        f"{table_df['num_features'].mean():.2f}"
    )

    print(
        "Overall missing rate: "
        f"{feature_df['missing_rate'].mean():.2%}"
    )

    print(
        "Mean categorical ratio per table: "
        f"{table_df['categorical_ratio'].mean():.2%}"
    )

    print(
        "Overall categorical ratio: "
        f"{feature_df['is_categorical'].mean():.2%}"
    )

    categorical_features = feature_df[
        feature_df["is_categorical"]
        == 1
    ]

    if len(
        categorical_features
    ) > 0:
        cardinality_counts = (
            categorical_features[
                "cardinality"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            "Categorical cardinality distribution:",
            cardinality_counts,
        )

    print()
    print(
        "Observation-mechanism summary:"
    )

    print(
        mechanism_df.to_string(
            index=False
        )
    )

    prototype_selected = int(
        (
            feature_df["mechanism"]
            == "prototype_discretization"
        ).sum()
        + (
            feature_df["mechanism"]
            == "continuous_fallback_from_prototype"
        ).sum()
    )

    prototype_successful = int(
        (
            feature_df["mechanism"]
            == "prototype_discretization"
        ).sum()
    )

    prototype_fallback = int(
        (
            feature_df["mechanism"]
            == "continuous_fallback_from_prototype"
        ).sum()
    )

    if prototype_selected > 0:
        print()
        print(
            "Prototype observation:"
        )

        print(
            f"  selected:   {prototype_selected}"
        )

        print(
            f"  successful: {prototype_successful}"
        )

        print(
            f"  fallback:   {prototype_fallback}"
        )

        print(
            "  success rate: "
            f"{prototype_successful / prototype_selected:.2%}"
        )

        print(
            "  fallback rate: "
            f"{prototype_fallback / prototype_selected:.2%}"
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    warnings.filterwarnings(
        "ignore"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cpu"
    )

    print(
        "========== SCM v7 predictive sanity check =========="
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Number of tables: {NUM_TABLES}"
    )

    print(
        f"Output directory: {OUTPUT_DIR.resolve()}"
    )

    table_rows = []
    feature_rows = []
    prediction_rows = []
    subset_rows = []

    generation_times = []

    total_start = time.perf_counter()

    for table_id in range(
        NUM_TABLES
    ):
        num_classes = sample_num_classes(
            table_id
        )

        generation_start = time.perf_counter()

        task = build_task(
            device=device,
            num_classes=num_classes,
            dag_seed=BASE_SEED + table_id,
            aleatoric_seed=10_000 + BASE_SEED + table_id,
            x_seed=20_000 + BASE_SEED + table_id,
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        generation_times.append(
            generation_time
        )

        (
            table_row,
            current_feature_rows,
            current_prediction_rows,
            current_subset_rows,
        ) = evaluate_one_table(
            task=task,
            table_id=table_id,
        )

        table_row[
            "generation_time"
        ] = generation_time

        table_rows.append(
            table_row
        )

        feature_rows.extend(
            current_feature_rows
        )

        prediction_rows.extend(
            current_prediction_rows
        )

        subset_rows.extend(
            current_subset_rows
        )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    table_df = pd.DataFrame(
        table_rows
    )

    feature_df = pd.DataFrame(
        feature_rows
    )

    prediction_df = pd.DataFrame(
        prediction_rows
    )

    subset_df = pd.DataFrame(
        subset_rows
    )

    mechanism_df = build_mechanism_summary(
        feature_df
    )

    table_df.to_csv(
        TABLE_CSV,
        index=False,
    )

    feature_df.to_csv(
        FEATURE_CSV,
        index=False,
    )

    prediction_df.to_csv(
        PREDICTION_CSV,
        index=False,
    )

    subset_df.to_csv(
        SUBSET_CSV,
        index=False,
    )

    mechanism_df.to_csv(
        MECHANISM_CSV,
        index=False,
    )

    print_generator_summary(
        table_df=table_df,
        feature_df=feature_df,
        mechanism_df=mechanism_df,
    )

    print_prediction_summary(
        prediction_df
    )

    print_subset_summary(
        subset_df
    )

    generation_times_array = np.asarray(
        generation_times,
        dtype=float,
    )

    print()
    print("=" * 110)
    print("RUNTIME")
    print("=" * 110)

    print(
        "Mean generation time: "
        f"{generation_times_array.mean():.4f}s"
    )

    print(
        "Median generation time: "
        f"{np.median(generation_times_array):.4f}s"
    )

    print(
        "Minimum generation time: "
        f"{generation_times_array.min():.4f}s"
    )

    print(
        "Maximum generation time: "
        f"{generation_times_array.max():.4f}s"
    )

    print(
        "Total execution time: "
        f"{total_elapsed:.2f}s"
    )

    print()
    print("=" * 110)
    print("SAVED FILES")
    print("=" * 110)

    print(
        f"  {TABLE_CSV}"
    )

    print(
        f"  {FEATURE_CSV}"
    )

    print(
        f"  {PREDICTION_CSV}"
    )

    print(
        f"  {SUBSET_CSV}"
    )

    print(
        f"  {MECHANISM_CSV}"
    )


if __name__ == "__main__":
    main()
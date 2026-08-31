# sanity_check/weighted_scm_validation.py

from __future__ import annotations

import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# IMPORTANT:
# Replace this import with the actual module containing your current generator.
# ---------------------------------------------------------------------------

from Trash.scm_task_v8 import WeightedMixedLatentSCMTask


# ============================================================================
# Configuration
# ============================================================================


NUM_TABLES = 25
BASE_SEED = 0

MIN_CLASSES = 2
MAX_CLASSES = 4

DEVICE = torch.device("cpu")

PRINT_EVERY_TABLE = True
RUN_FEATURE_SUBSET_TESTS = True

MODEL_N_ESTIMATORS = 300

TABLE_CSV = "weighted_scm_table_diagnostics.csv"
MODEL_CSV = "weighted_scm_model_metrics.csv"
SUBSET_CSV = "weighted_scm_subset_metrics.csv"
FEATURE_CSV = "weighted_scm_feature_diagnostics.csv"


# ============================================================================
# Task configuration
# ============================================================================


TASK_KWARGS = dict(
    n_min=400,
    n_max=512,
    d_min=8,
    d_max=16,
    test_frac=0.15,
    p_missing=0.05,

    num_roots=4,
    num_layers=5,
    hidden_width_min=8,
    hidden_width_max=12,
    final_width=1,
    latent_dim=4,

    connection_probs=(0.30, 0.45, 0.65, 0.85),
    min_parents_per_node=2,
    edge_weight_concentration=0.60,

    latent_noise_scale=0,
    observation_noise_scale=0.03,

    dominant_mass_threshold=0.70,
    dominant_feature_fraction=0.70,

    observation_type_probs=(1.0, 0, 0),

    categorical_cardinalities=(2, 3, 4, 5, 6),
    categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),

    min_samples_per_category=8,
    min_component_weight=0.05,

    prototype_max_attempts=8,
    prototype_min_separation=1.0,
    binning_jitter=0.20,

    root_prior_probs=(0.45, 0.20, 0.15, 0, 0),
    root_mixture_component_probs=(0.40, 0.30, 0.18, 0.08, 0.04),

    root_mixture_separation_min=1.5,
    root_mixture_separation_max=3.0,
    root_mixture_scale_min=0.40,
    root_mixture_scale_max=0.90,

    linear_activation_prob=0.60,
    small_mlp_prob=0.25,
    soft_tree_prob=0.15,

    small_mlp_hidden_dim=None,
    soft_tree_depth=2,
    soft_tree_temperature=0.5,

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
    if isinstance(task, (tuple, list)):
        if len(task) != 5:
            raise ValueError(
                "Expected task tuple "
                "(X_train, y_train, X_test, y_test, info)."
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


def global_node_id_to_layer(
    global_id: int,
    widths: list[int],
) -> tuple[int, int]:
    """
    Convert flattened global node id to:
        (layer index, node index inside layer)
    """
    start = 0

    for layer_idx, width in enumerate(widths):
        end = start + width

        if start <= global_id < end:
            return layer_idx, global_id - start

        start = end

    raise ValueError(
        f"global_id={global_id} is outside graph with widths={widths}."
    )


def safe_mean(values) -> float:
    array = np.asarray(values, dtype=float)

    if array.size == 0 or np.isnan(array).all():
        return float("nan")

    return float(np.nanmean(array))


# ============================================================================
# Build one task
# ============================================================================


def build_task(
    table_id: int,
    num_classes: int,
) -> WeightedMixedLatentSCMTask:
    return WeightedMixedLatentSCMTask(
        num_classes=num_classes,
        dag_seed=BASE_SEED + table_id,
        aleatoric_seed=100_000 + BASE_SEED + table_id,
        x_seed=200_000 + BASE_SEED + table_id,
        **TASK_KWARGS,
    )


# ============================================================================
# Data validation
# ============================================================================


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]


def validate_task_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    info: dict,
) -> ValidationResult:
    errors: list[str] = []
    warnings_list: list[str] = []

    if X_train.ndim != 2 or X_test.ndim != 2:
        errors.append("X_train and X_test must both be 2-dimensional.")

    if y_train.ndim != 1 or y_test.ndim != 1:
        errors.append("y_train and y_test must both be 1-dimensional.")

    if X_train.shape[1] != X_test.shape[1]:
        errors.append(
            "Train and test feature counts differ: "
            f"{X_train.shape[1]} versus {X_test.shape[1]}."
        )

    if X_train.shape[0] != y_train.shape[0]:
        errors.append("X_train and y_train lengths differ.")

    if X_test.shape[0] != y_test.shape[0]:
        errors.append("X_test and y_test lengths differ.")

    d = X_train.shape[1]

    feature_type = to_numpy(info["feature_type"]).astype(int)
    cardinality = to_numpy(info["cardinality"]).astype(int)
    type_ids = to_numpy(
        info["feature_observation_type_ids"]
    ).astype(int)

    if len(feature_type) != d:
        errors.append(
            f"feature_type length={len(feature_type)}, expected {d}."
        )

    if len(cardinality) != d:
        errors.append(
            f"cardinality length={len(cardinality)}, expected {d}."
        )

    if len(type_ids) != d:
        errors.append(
            f"observation type length={len(type_ids)}, expected {d}."
        )

    X_full = np.concatenate(
        [X_train, X_test],
        axis=0,
    )

    if np.isinf(X_full).any():
        errors.append("X contains positive or negative infinity.")

    if not np.isfinite(y_train).all():
        errors.append("y_train contains non-finite values.")

    if not np.isfinite(y_test).all():
        errors.append("y_test contains non-finite values.")

    classes = np.unique(
        np.concatenate([y_train, y_test])
    )

    expected_classes = np.arange(len(classes))

    if not np.array_equal(classes, expected_classes):
        warnings_list.append(
            f"Class labels are {classes.tolist()}, "
            f"not consecutive labels {expected_classes.tolist()}."
        )

    for col in range(d):
        values = X_full[:, col]
        valid = values[~np.isnan(values)]

        if valid.size == 0:
            errors.append(f"Feature {col} is entirely missing.")
            continue

        if feature_type[col] == 0:
            if cardinality[col] != 0:
                errors.append(
                    f"Continuous feature {col} has "
                    f"cardinality={cardinality[col]}."
                )

            std = float(np.std(valid))

            if std <= 1e-8:
                warnings_list.append(
                    f"Continuous feature {col} is nearly constant."
                )

        elif feature_type[col] == 1:
            k = int(cardinality[col])

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
                    f"Categorical feature {col} contains non-integer values."
                )

            integer_values = rounded.astype(int)

            if integer_values.min() < 0:
                errors.append(
                    f"Categorical feature {col} contains negative labels."
                )

            if integer_values.max() >= k:
                errors.append(
                    f"Categorical feature {col} contains label "
                    f"{integer_values.max()}, but K={k}."
                )

            observed_categories = np.unique(integer_values)

            if len(observed_categories) < 2:
                warnings_list.append(
                    f"Categorical feature {col} has only one observed category."
                )

        else:
            errors.append(
                f"Feature {col} has unknown feature_type={feature_type[col]}."
            )

    return ValidationResult(
        errors=errors,
        warnings=warnings_list,
    )


# ============================================================================
# Structural checks
# ============================================================================


def validate_graph_structure(task) -> ValidationResult:
    errors: list[str] = []
    warnings_list: list[str] = []

    scm = task.scm
    widths = list(scm.widths)

    if widths[-1] != 1:
        warnings_list.append(
            f"Final layer width is {widths[-1]}, not 1."
        )

    for layer_idx, connection in enumerate(scm.connections):
        adjacency = connection.adj.detach().cpu()
        weights = connection.weights.detach().cpu()

        expected_shape = (
            widths[layer_idx],
            widths[layer_idx + 1],
        )

        if tuple(adjacency.shape) != expected_shape:
            errors.append(
                f"Layer {layer_idx} adjacency shape "
                f"{tuple(adjacency.shape)} != {expected_shape}."
            )

        if tuple(weights.shape) != expected_shape:
            errors.append(
                f"Layer {layer_idx} weight shape "
                f"{tuple(weights.shape)} != {expected_shape}."
            )

        if bool((weights < -1e-8).any()):
            errors.append(
                f"Layer {layer_idx} contains negative edge weights."
            )

        disconnected_weight = weights[~adjacency]

        if disconnected_weight.numel() > 0:
            max_disconnected = float(
                disconnected_weight.abs().max().item()
            )

            if max_disconnected > 1e-7:
                errors.append(
                    f"Layer {layer_idx} has nonzero weights "
                    "for disconnected edges."
                )

        for child in range(connection.out_width):
            parent_mask = adjacency[:, child]
            parent_count = int(parent_mask.sum().item())

            if parent_count == 0:
                errors.append(
                    f"Layer {layer_idx}, child {child} has no parent."
                )
                continue

            weight_sum = float(
                weights[parent_mask, child].sum().item()
            )

            if not np.isclose(
                weight_sum,
                1.0,
                atol=1e-5,
            ):
                errors.append(
                    f"Layer {layer_idx}, child {child} "
                    f"has parent weight sum={weight_sum:.8f}."
                )

    influence = scm.compute_node_influence(
        target_node_idx=0
    )

    if len(influence) != len(widths):
        errors.append(
            "Influence output length does not match number of layers."
        )
        return ValidationResult(errors, warnings_list)

    if not torch.allclose(
        influence[-1],
        torch.tensor(
            [1.0],
            device=influence[-1].device,
            dtype=influence[-1].dtype,
        ),
    ):
        errors.append(
            "Final target influence is not exactly [1.0]."
        )

    for layer in range(len(widths) - 1):
        recomputed = (
            scm.connections[layer].weights
            @ influence[layer + 1]
        )

        if not torch.allclose(
            recomputed,
            influence[layer],
            atol=1e-6,
            rtol=1e-5,
        ):
            errors.append(
                f"Influence recurrence failed at layer {layer}."
            )

    return ValidationResult(
        errors=errors,
        warnings=warnings_list,
    )


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
        feature_type == 0
    )[0].tolist()

    categorical_indices = np.where(
        feature_type == 1
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
            "No features are available for preprocessing."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================================
# Models and metrics
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
                        n_estimators=MODEL_N_ESTIMATORS,
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
                        n_estimators=MODEL_N_ESTIMATORS,
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


def compute_auc(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    try:
        probabilities = model.predict_proba(X_test)
        model_classes = np.asarray(model.classes_)

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
        probabilities = model.predict_proba(X_test)

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
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auc": float("nan"),
            "logloss": float("nan"),
        }

    subset_type = feature_type[
        selected_indices
    ]

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    subset_type
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=MODEL_N_ESTIMATORS,
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
# One-table evaluation
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

    feature_ids = to_numpy(
        info["feature_ids"]
    ).astype(int)

    feature_strength = to_numpy(
        info["feature_strength"]
    ).astype(float)

    selected_dominant = to_numpy(
        info["selected_from_dominant_group"]
    ).astype(bool)

    observation_names = list(
        info["feature_observation_type_names"]
    )

    widths = to_numpy(
        info["layer_widths"]
    ).astype(int).tolist()

    data_validation = validate_task_data(
        X_train,
        y_train,
        X_test,
        y_test,
        info,
    )

    graph_validation = validate_graph_structure(
        task
    )

    all_errors = (
        data_validation.errors
        + graph_validation.errors
    )

    all_warnings = (
        data_validation.warnings
        + graph_validation.warnings
    )

    d = X_train.shape[1]
    num_classes = int(
        np.unique(y_full).size
    )

    continuous_count = int(
        (feature_type == 0).sum()
    )

    categorical_count = int(
        (feature_type == 1).sum()
    )

    missing_rate = float(
        np.isnan(X_full).mean()
    )

    dominant_selected_count = int(
        selected_dominant.sum()
    )

    dominant_selected_ratio = float(
        selected_dominant.mean()
    )

    mechanism_counts = Counter(
        observation_names
    )

    feature_layers = []
    feature_nodes_in_layer = []

    for global_id in feature_ids:
        layer_idx, node_idx = global_node_id_to_layer(
            int(global_id),
            widths,
        )

        feature_layers.append(layer_idx)
        feature_nodes_in_layer.append(node_idx)

    layer_counts = Counter(
        feature_layers
    )

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
        "missing_rate": missing_rate,
        "dominant_selected_count": dominant_selected_count,
        "dominant_selected_ratio": dominant_selected_ratio,
        "dominant_group_size": int(
            info["dominant_group_ids"].numel()
        ),
        "mean_selected_strength": float(
            feature_strength.mean()
        ),
        "max_selected_strength": float(
            feature_strength.max()
        ),
        "min_selected_strength": float(
            feature_strength.min()
        ),
        "num_validation_errors": len(all_errors),
        "num_validation_warnings": len(all_warnings),
        "root_prior_types": "|".join(
            info["root_prior_types"]
        ),
        "layer_widths": "|".join(
            str(v)
            for v in widths
        ),
    }

    for mechanism_name in (
        "continuous_projection",
        "prototype_discretization",
        "threshold_binning",
        "continuous_fallback_from_prototype",
        "continuous_fallback_from_binning",
    ):
        table_row[
            f"count_{mechanism_name}"
        ] = mechanism_counts.get(
            mechanism_name,
            0,
        )

    for layer_idx in range(
        len(widths) - 1
    ):
        table_row[
            f"selected_layer_{layer_idx}"
        ] = layer_counts.get(
            layer_idx,
            0,
        )

    feature_rows = []

    for col in range(d):
        values = X_full[:, col]
        valid = values[
            ~np.isnan(values)
        ]

        row = {
            "table_id": table_id,
            "feature_index": col,
            "global_node_id": int(
                feature_ids[col]
            ),
            "source_layer": int(
                feature_layers[col]
            ),
            "source_node": int(
                feature_nodes_in_layer[col]
            ),
            "feature_type": (
                "categorical"
                if feature_type[col] == 1
                else "continuous"
            ),
            "mechanism": observation_names[col],
            "cardinality": int(
                cardinality[col]
            ),
            "selected_from_dominant": bool(
                selected_dominant[col]
            ),
            "structural_path_score": float(
                feature_strength[col]
            ),
            "missing_rate": float(
                np.isnan(values).mean()
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

        if feature_type[col] == 1:
            counts = np.bincount(
                valid.astype(int),
                minlength=int(cardinality[col]),
            )

            row["smallest_category_count"] = int(
                counts.min()
            )

            row["smallest_category_fraction"] = float(
                counts.min()
                / counts.sum()
            )

            row["category_counts"] = "|".join(
                str(int(v))
                for v in counts
            )

        else:
            row["smallest_category_count"] = np.nan
            row["smallest_category_fraction"] = np.nan
            row["category_counts"] = ""

        feature_rows.append(row)

    if PRINT_EVERY_TABLE:
        print()
        print("=" * 118)
        print(
            f"TABLE {table_id:03d} | "
            f"classes={num_classes} | "
            f"features={d} | "
            f"continuous={continuous_count} | "
            f"categorical={categorical_count} | "
            f"missing={missing_rate:.2%}"
        )
        print("=" * 118)

        print(
            f"Layer widths: {widths}"
        )

        print(
            "Roots:",
            info["root_prior_types"],
        )

        print(
            "Mechanisms:",
            dict(mechanism_counts),
        )

        print(
            "Selected feature layers:",
            dict(sorted(layer_counts.items())),
        )

        print(
            "Dominant selected: "
            f"{dominant_selected_count}/{d} "
            f"({dominant_selected_ratio:.2%})"
        )

        print(
            "Target class counts:",
            dict(
                zip(
                    *np.unique(
                        y_full,
                        return_counts=True,
                    )
                )
            ),
        )

        if all_errors:
            print("\nVALIDATION ERRORS:")

            for message in all_errors:
                print(f"  ERROR: {message}")

        if all_warnings:
            print("\nVALIDATION WARNINGS:")

            for message in all_warnings:
                print(f"  WARNING: {message}")

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
                "model": model_name,
                **metrics,
            }
        )

        if PRINT_EVERY_TABLE:
            print(
                f"{model_name:24s} | "
                f"acc={metrics['accuracy']:.4f} | "
                f"bal_acc={metrics['balanced_accuracy']:.4f} | "
                f"f1={metrics['macro_f1']:.4f} | "
                f"auc={metrics['auc']:.4f} | "
                f"logloss={metrics['logloss']:.4f}"
            )

    subset_rows = []

    if RUN_FEATURE_SUBSET_TESTS:
        all_indices = np.arange(
            d,
            dtype=int,
        )

        continuous_indices = np.where(
            feature_type == 0
        )[0]

        categorical_indices = np.where(
            feature_type == 1
        )[0]

        dominant_indices = np.where(
            selected_dominant
        )[0]

        non_dominant_indices = np.where(
            ~selected_dominant
        )[0]

        prototype_indices = np.asarray(
            [
                i
                for i, name in enumerate(observation_names)
                if name == "prototype_discretization"
            ],
            dtype=int,
        )

        binning_indices = np.asarray(
            [
                i
                for i, name in enumerate(observation_names)
                if name == "threshold_binning"
            ],
            dtype=int,
        )

        continuous_projection_indices = np.asarray(
            [
                i
                for i, name in enumerate(observation_names)
                if name == "continuous_projection"
            ],
            dtype=int,
        )

        fallback_indices = np.asarray(
            [
                i
                for i, name in enumerate(observation_names)
                if "fallback" in name
            ],
            dtype=int,
        )

        remove_dominant_indices = np.where(
            ~selected_dominant
        )[0]

        remove_prototype_indices = np.asarray(
            [
                i
                for i, name in enumerate(observation_names)
                if name != "prototype_discretization"
            ],
            dtype=int,
        )

        remove_binning_indices = np.asarray(
            [
                i
                for i, name in enumerate(observation_names)
                if name != "threshold_binning"
            ],
            dtype=int,
        )

        subset_definitions = {
            "all_features": all_indices,
            "dominant_features": dominant_indices,
            "non_dominant_features": non_dominant_indices,
            "remove_dominant": remove_dominant_indices,
            "continuous_features": continuous_indices,
            "categorical_features": categorical_indices,
            "continuous_projection_only": continuous_projection_indices,
            "prototype_only": prototype_indices,
            "binning_only": binning_indices,
            "fallback_only": fallback_indices,
            "remove_prototype": remove_prototype_indices,
            "remove_binning": remove_binning_indices,
        }

        if PRINT_EVERY_TABLE:
            print(
                "\nRandom Forest feature-subset checks:"
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
                    "subset": subset_name,
                    "num_selected": int(
                        len(selected_indices)
                    ),
                    **metrics,
                }
            )

            if PRINT_EVERY_TABLE:
                print(
                    f"{subset_name:28s} | "
                    f"features={len(selected_indices):2d} | "
                    f"bal_acc={metrics['balanced_accuracy']:.4f} | "
                    f"auc={metrics['auc']:.4f}"
                )

    return (
        table_row,
        model_rows,
        subset_rows,
        feature_rows,
    )


# ============================================================================
# Summary reporting
# ============================================================================


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
            ["mean", "std", "median", "min", "max"]
        )
    )

    print(summary)

    pivot = model_df.pivot(
        index="table_id",
        columns="model",
        values="balanced_accuracy",
    )

    if (
        "dummy_prior" in pivot.columns
        and "random_forest" in pivot.columns
    ):
        lift = (
            pivot["random_forest"]
            - pivot["dummy_prior"]
        )

        print()
        print("Random Forest balanced-accuracy lift over dummy:")
        print(
            f"  mean   = {lift.mean():.4f}"
        )
        print(
            f"  median = {lift.median():.4f}"
        )
        print(
            f"  min    = {lift.min():.4f}"
        )
        print(
            f"  max    = {lift.max():.4f}"
        )
        print(
            "  RF beats dummy on "
            f"{(lift > 0).mean():.2%} of tables"
        )
        print(
            "  RF beats dummy by >= 0.05 on "
            f"{(lift >= 0.05).mean():.2%} of tables"
        )
        print(
            "  RF beats dummy by >= 0.10 on "
            f"{(lift >= 0.10).mean():.2%} of tables"
        )

    for model_name in (
        "logistic_regression",
        "random_forest",
        "extra_trees",
    ):
        if (
            "dummy_prior" in pivot.columns
            and model_name in pivot.columns
        ):
            lift = (
                pivot[model_name]
                - pivot["dummy_prior"]
            )

            print()
            print(
                f"{model_name} versus dummy:"
            )
            print(
                f"  mean balanced-accuracy lift = {lift.mean():.4f}"
            )
            print(
                f"  tables above dummy           = {(lift > 0).mean():.2%}"
            )


def print_table_summary(
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
        "missing_rate",
        "dominant_group_size",
        "dominant_selected_count",
        "dominant_selected_ratio",
        "mean_selected_strength",
        "num_validation_errors",
        "num_validation_warnings",
    ]

    print(
        table_df[
            columns
        ].agg(
            ["mean", "std", "median", "min", "max"]
        ).T
    )

    print()
    print(
        "Tables with validation errors:",
        int(
            (
                table_df[
                    "num_validation_errors"
                ]
                > 0
            ).sum()
        ),
        "/",
        len(table_df),
    )

    print(
        "Tables with validation warnings:",
        int(
            (
                table_df[
                    "num_validation_warnings"
                ]
                > 0
            ).sum()
        ),
        "/",
        len(table_df),
    )

    mechanism_columns = [
        column
        for column in table_df.columns
        if column.startswith("count_")
    ]

    mechanism_totals = (
        table_df[
            mechanism_columns
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print()
    print("Mechanism totals:")

    for column, count in mechanism_totals.items():
        print(
            f"  {column.removeprefix('count_')}: {int(count)}"
        )

    total_features = int(
        table_df[
            "num_features"
        ].sum()
    )

    print()
    print(
        "Overall categorical ratio:",
        f"{table_df['categorical_features'].sum() / total_features:.2%}",
    )

    print(
        "Overall dominant-selected ratio:",
        f"{table_df['dominant_selected_count'].sum() / total_features:.2%}",
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
            ["mean", "std", "median"]
        )
    )

    print(summary)

    pivot = subset_df.pivot(
        index="table_id",
        columns="subset",
        values="balanced_accuracy",
    )

    if {
        "dominant_features",
        "non_dominant_features",
    }.issubset(pivot.columns):
        difference = (
            pivot["dominant_features"]
            - pivot["non_dominant_features"]
        )

        print()
        print(
            "Dominant-only minus non-dominant-only "
            "balanced accuracy:"
        )
        print(
            f"  mean   = {difference.mean():.4f}"
        )
        print(
            f"  median = {difference.median():.4f}"
        )
        print(
            "  dominant subset performs better on "
            f"{(difference > 0).mean():.2%} of comparable tables"
        )

    if {
        "all_features",
        "remove_dominant",
    }.issubset(pivot.columns):
        drop = (
            pivot["all_features"]
            - pivot["remove_dominant"]
        )

        print()
        print(
            "Balanced-accuracy drop after removing dominant features:"
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
                "mechanism",
            ]
        ).size()
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
            "Smallest category fraction:"
        )

        print(
            categorical[
                "smallest_category_fraction"
            ].describe()
        )

    print()
    print(
        "Selected source-layer distribution:"
    )

    print(
        feature_df[
            "source_layer"
        ].value_counts().sort_index()
    )

    print()
    print(
        "Mean structural path score by source layer:"
    )

    print(
        feature_df.groupby(
            "source_layer"
        )[
            "structural_path_score"
        ].mean()
    )

    print()
    print(
        "Mean structural path score by dominant membership:"
    )

    print(
        feature_df.groupby(
            "selected_from_dominant"
        )[
            "structural_path_score"
        ].mean()
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

        generation_time = (
            time.perf_counter()
            - start
        )

        generation_times.append(
            generation_time
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
        ] = generation_time

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

    print_table_summary(
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
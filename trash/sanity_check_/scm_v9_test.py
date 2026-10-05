from __future__ import annotations

import inspect
import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from Trash.scm_task_v9 import WeightedLayeredScalarSCM, WeightedMixedScalarSCMTask

BASE_SEED = 0
NUM_TABLES_PER_EXPERIMENT = 5
MIN_CLASSES = 2
MAX_CLASSES = 4
DEVICE = torch.device("cpu")
MODEL_N_ESTIMATORS = 300
PRINT_EVERY_TABLE = True
RUN_SHUFFLED_LABEL_CONTROL = True
RUN_LATENT_SHAPE_CHECK = True
RUN_FEATURE_SUBSET_CHECK = True

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CSV = OUTPUT_DIR / "scalar_scm_model_metrics.csv"
TABLE_CSV = OUTPUT_DIR / "scalar_scm_table_diagnostics.csv"
SUMMARY_CSV = OUTPUT_DIR / "scalar_scm_experiment_summary.csv"
SUBSET_CSV = OUTPUT_DIR / "scalar_scm_subset_metrics.csv"
CONFIG_JSON = OUTPUT_DIR / "scalar_scm_validation_config.json"

BASE_TASK_KWARGS = dict(
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
    connection_probs=(0.30, 0.45, 0.65, 0.85),
    min_parents_per_node=2,
    edge_weight_concentration=0.60,
    latent_noise_scale=0.0,
    observation_noise_scale=0.03,
    dominant_mass_threshold=0.70,
    dominant_feature_fraction=0.50,
    observation_type_probs=(0.70, 0.15, 0.15),
    categorical_cardinalities=(2, 3, 4, 5, 6),
    categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    min_samples_per_category=8,
    min_component_weight=0.05,
    prototype_max_attempts=8,
    prototype_min_separation=1.0,
    binning_jitter=0.20,
    root_prior_probs=(0.45, 0.20, 0.15, 0.05, 0.15),
    root_mixture_component_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    root_mixture_separation_min=1.5,
    root_mixture_separation_max=3.0,
    root_mixture_scale_min=0.40,
    root_mixture_scale_max=0.90,
    small_mlp_hidden_dim=None,
    soft_tree_depth=2,
    soft_tree_temperature=0.5,
    device=DEVICE,
)

EXPERIMENTS = {
    "linear_only": dict(linear_activation_prob=1.0, small_mlp_prob=0.0, soft_tree_prob=0.0),
    "mlp_only": dict(linear_activation_prob=0.0, small_mlp_prob=1.0, soft_tree_prob=0.0),
    "soft_tree_only": dict(linear_activation_prob=0.0, small_mlp_prob=0.0, soft_tree_prob=1.0),
    "mixed": dict(linear_activation_prob=0.60, small_mlp_prob=0.25, soft_tree_prob=0.15),
}


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def class_count_dict(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def global_node_id_to_layer(global_id: int, widths: list[int]) -> tuple[int, int]:
    start = 0
    for layer_idx, width in enumerate(widths):
        end = start + int(width)
        if start <= global_id < end:
            return layer_idx, global_id - start
        start = end
    raise ValueError(f"global node id {global_id} is outside widths={widths}.")


def build_task(experiment_name: str, table_id: int, num_classes: int) -> WeightedMixedScalarSCMTask:
    kwargs = dict(BASE_TASK_KWARGS)
    kwargs.update(EXPERIMENTS[experiment_name])
    experiment_position = list(EXPERIMENTS).index(experiment_name)
    offset = experiment_position * 1_000_000
    return WeightedMixedScalarSCMTask(
        num_classes=num_classes,
        dag_seed=BASE_SEED + offset + table_id,
        aleatoric_seed=100_000 + BASE_SEED + offset + table_id,
        x_seed=200_000 + BASE_SEED + offset + table_id,
        **kwargs,
    )


def print_import_debug():
    print("\n" + "=" * 120)
    print("IMPORT DEBUG")
    print("=" * 120)
    print("Task class:")
    print(WeightedMixedScalarSCMTask)
    print("\nTask class file:")
    print(inspect.getfile(WeightedMixedScalarSCMTask))
    print("\nSCM class:")
    print(WeightedLayeredScalarSCM)
    print("\nSCM class file:")
    print(inspect.getfile(WeightedLayeredScalarSCM))


def validate_latent_shapes(task: WeightedMixedScalarSCMTask, n_samples: int = 32) -> list[str]:
    errors: list[str] = []
    latents = task.scm.forward(n_samples, latent_noise_scale=0.0)
    if len(latents) != len(task.scm.widths):
        return ["Number of latent layers does not match scm.widths."]
    for layer_idx, layer in enumerate(latents):
        expected_width = int(task.scm.widths[layer_idx])
        if len(layer) != expected_width:
            errors.append(f"Layer {layer_idx} contains {len(layer)} nodes, expected {expected_width}.")
        for node_idx, node in enumerate(layer):
            expected_shape = (n_samples, 1)
            if tuple(node.shape) != expected_shape:
                errors.append(
                    f"Layer {layer_idx}, node {node_idx} has shape {tuple(node.shape)}, expected {expected_shape}."
                )
            if not torch.isfinite(node).all():
                errors.append(f"Layer {layer_idx}, node {node_idx} contains non-finite values.")
    return errors


def validate_joint_mlp_parameters(parameters: dict, num_parents: int, layer_idx: int, child: int) -> list[str]:
    errors: list[str] = []
    required = {"W1", "b1", "W2", "b2"}
    missing = required.difference(parameters)
    if missing:
        return [f"Connection {layer_idx}, child {child}: joint MLP is missing keys {sorted(missing)}."]
    W1 = parameters["W1"]
    if W1.ndim != 2:
        return [f"Connection {layer_idx}, child {child}: joint MLP W1 has ndim={W1.ndim}, expected 2."]
    hidden = int(W1.shape[0])
    expected_shapes = {
        "W1": (hidden, num_parents),
        "b1": (hidden,),
        "W2": (1, hidden),
        "b2": (1,),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(parameters[name].shape)
        if actual_shape != expected_shape:
            errors.append(
                f"Connection {layer_idx}, child {child}: joint MLP {name} has shape {actual_shape}, expected {expected_shape}."
            )
        if not torch.isfinite(parameters[name]).all():
            errors.append(f"Connection {layer_idx}, child {child}: joint MLP {name} contains non-finite values.")
    return errors


def validate_graph_structure(task: WeightedMixedScalarSCMTask) -> list[str]:
    errors: list[str] = []
    scm = task.scm
    widths = list(scm.widths)

    if len(widths) != scm.num_layers:
        errors.append("len(scm.widths) does not equal scm.num_layers.")
    if widths[-1] != 1:
        errors.append(f"Final width is {widths[-1]}, expected 1.")
    if len(scm.connections) != len(widths) - 1:
        errors.append("Number of connections does not equal number of layer transitions.")
        return errors

    for layer_idx, connection in enumerate(scm.connections):
        expected_shape = (widths[layer_idx], widths[layer_idx + 1])
        if tuple(connection.adj.shape) != expected_shape:
            errors.append(
                f"Connection {layer_idx} adjacency shape is {tuple(connection.adj.shape)}, expected {expected_shape}."
            )
        if tuple(connection.weights.shape) != expected_shape:
            errors.append(
                f"Connection {layer_idx} weight shape is {tuple(connection.weights.shape)}, expected {expected_shape}."
            )
        if tuple(connection.child_methods.shape) != (connection.out_width,):
            errors.append(
                f"Connection {layer_idx} child_methods shape is {tuple(connection.child_methods.shape)}, expected {(connection.out_width,)}."
            )
        if len(connection.child_scalar_edges) != connection.out_width:
            errors.append(
                f"Connection {layer_idx}: child_scalar_edges length is {len(connection.child_scalar_edges)}, expected {connection.out_width}."
            )
        if len(connection.child_joint_mlps) != connection.out_width:
            errors.append(
                f"Connection {layer_idx}: child_joint_mlps length is {len(connection.child_joint_mlps)}, expected {connection.out_width}."
            )
        if bool((connection.weights < -1e-8).any()):
            errors.append(f"Connection {layer_idx} contains negative weights.")

        disconnected_weights = connection.weights[~connection.adj]
        if disconnected_weights.numel() > 0 and float(disconnected_weights.abs().max().item()) > 1e-7:
            errors.append(f"Connection {layer_idx} contains nonzero disconnected weights.")

        for child in range(connection.out_width):
            parent_mask = connection.adj[:, child]
            parents = torch.where(parent_mask)[0]
            num_parents = int(parents.numel())
            if num_parents == 0:
                errors.append(f"Connection {layer_idx}, child {child} has no parents.")
                continue

            weight_sum = float(connection.weights[parent_mask, child].sum().item())
            if not np.isclose(weight_sum, 1.0, atol=1e-5):
                errors.append(
                    f"Connection {layer_idx}, child {child}: parent weights sum to {weight_sum:.8f}, expected 1."
                )

            method = int(connection.child_methods[child].item())

            if method == 0:
                for parent in range(connection.in_width):
                    edge = connection.edges[parent][child]
                    connected = bool(connection.adj[parent, child].item())
                    if connected and edge is None:
                        errors.append(
                            f"Connection {layer_idx}, child {child}: edgewise method has missing edge {parent}->{child}."
                        )
                    if not connected and edge is not None:
                        errors.append(
                            f"Connection {layer_idx}, child {child}: disconnected edge {parent}->{child} has an edge object."
                        )
                if connection.child_scalar_edges[child] is not None:
                    errors.append(
                        f"Connection {layer_idx}, child {child}: edgewise method unexpectedly has a child scalar edge."
                    )
                if connection.child_joint_mlps[child] is not None:
                    errors.append(
                        f"Connection {layer_idx}, child {child}: edgewise method unexpectedly has joint MLP parameters."
                    )

            elif method == 1:
                for parent in range(connection.in_width):
                    if connection.edges[parent][child] is not None:
                        errors.append(
                            f"Connection {layer_idx}, child {child}: post-aggregate method unexpectedly has edge object {parent}->{child}."
                        )
                if connection.child_scalar_edges[child] is None:
                    errors.append(
                        f"Connection {layer_idx}, child {child}: post-aggregate method is missing its child scalar function."
                    )
                if connection.child_joint_mlps[child] is not None:
                    errors.append(
                        f"Connection {layer_idx}, child {child}: post-aggregate method unexpectedly has joint MLP parameters."
                    )

            elif method == 2:
                for parent in range(connection.in_width):
                    if connection.edges[parent][child] is not None:
                        errors.append(
                            f"Connection {layer_idx}, child {child}: joint-MLP method unexpectedly has edge object {parent}->{child}."
                        )
                if connection.child_scalar_edges[child] is not None:
                    errors.append(
                        f"Connection {layer_idx}, child {child}: joint-MLP method unexpectedly has a child scalar edge."
                    )
                parameters = connection.child_joint_mlps[child]
                if parameters is None:
                    errors.append(
                        f"Connection {layer_idx}, child {child}: joint-MLP method is missing parameters."
                    )
                else:
                    errors.extend(validate_joint_mlp_parameters(parameters, num_parents, layer_idx, child))
            else:
                errors.append(f"Connection {layer_idx}, child {child}: unknown child method {method}.")

    influence = scm.compute_sampling_influence(target_node_idx=0)
    if len(influence) != len(widths):
        errors.append("Influence layer count is incorrect.")
        return errors

    expected_target = torch.ones(1, device=influence[-1].device, dtype=influence[-1].dtype)
    if not torch.allclose(influence[-1], expected_target, atol=1e-7):
        errors.append("Final target influence is not [1.0].")

    for layer_idx in range(len(widths) - 1):
        recomputed = scm.connections[layer_idx].weights @ influence[layer_idx + 1]
        if not torch.allclose(recomputed, influence[layer_idx], atol=1e-6, rtol=1e-5):
            errors.append(f"Influence recurrence failed at layer {layer_idx}.")

    return errors


def validate_generated_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    info: dict,
) -> list[str]:
    errors: list[str] = []
    if X_train.ndim != 2:
        errors.append(f"X_train shape is {X_train.shape}; expected 2D.")
    if X_test.ndim != 2:
        errors.append(f"X_test shape is {X_test.shape}; expected 2D.")
    if y_train.ndim != 1:
        errors.append(f"y_train shape is {y_train.shape}; expected 1D.")
    if y_test.ndim != 1:
        errors.append(f"y_test shape is {y_test.shape}; expected 1D.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("X_train and y_train have different sample counts.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("X_test and y_test have different sample counts.")
    if X_train.shape[1] != X_test.shape[1]:
        errors.append("Train and test feature counts differ.")

    required = [
        "feature_type",
        "cardinality",
        "feature_ids",
        "feature_observation_type_ids",
        "feature_observation_type_names",
        "selected_from_dominant_group",
        "layer_widths",
    ]
    for key in required:
        if key not in info:
            errors.append(f"info is missing required key: {key}.")
    if errors:
        return errors

    d = X_train.shape[1]
    feature_type = to_numpy(info["feature_type"]).astype(int)
    cardinality = to_numpy(info["cardinality"]).astype(int)
    feature_ids = to_numpy(info["feature_ids"]).astype(int)
    type_ids = to_numpy(info["feature_observation_type_ids"]).astype(int)
    type_names = list(info["feature_observation_type_names"])

    lengths = {
        "feature_type": len(feature_type),
        "cardinality": len(cardinality),
        "feature_ids": len(feature_ids),
        "observation_type_ids": len(type_ids),
        "observation_type_names": len(type_names),
    }
    for name, length in lengths.items():
        if length != d:
            errors.append(f"{name} has length {length}, expected {d}.")

    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)
    if np.isinf(X_full).any():
        errors.append("X contains positive or negative infinity.")
    if not np.isfinite(y_full).all():
        errors.append("y contains non-finite values.")

    classes = np.unique(y_full)
    expected_classes = np.arange(len(classes))
    if not np.array_equal(classes, expected_classes):
        errors.append(
            f"Class labels are {classes.tolist()}, expected consecutive labels {expected_classes.tolist()}."
        )

    for column in range(d):
        values = X_full[:, column]
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            errors.append(f"Feature {column} is entirely missing.")
            continue
        current_type = int(feature_type[column])
        current_cardinality = int(cardinality[column])
        if current_type == 0:
            if current_cardinality != 0:
                errors.append(f"Continuous feature {column} has cardinality={current_cardinality}.")
        elif current_type == 1:
            if current_cardinality < 2:
                errors.append(f"Categorical feature {column} has invalid cardinality={current_cardinality}.")
                continue
            rounded = np.round(valid)
            if not np.allclose(valid, rounded, atol=1e-6):
                errors.append(f"Categorical feature {column} contains non-integer labels.")
            integer_values = rounded.astype(int)
            if integer_values.min() < 0:
                errors.append(f"Categorical feature {column} contains negative labels.")
            if integer_values.max() >= current_cardinality:
                errors.append(
                    f"Categorical feature {column} contains label {integer_values.max()}, but cardinality={current_cardinality}."
                )
        else:
            errors.append(f"Feature {column} has unknown feature_type={current_type}.")
    return errors


def collect_edge_diagnostics(task: WeightedMixedScalarSCMTask):
    function_counts = Counter()
    activation_counts = Counter()
    child_method_counts = Counter()
    method_names = {0: "edgewise", 1: "post_aggregate", 2: "joint_mlp"}

    def record_scalar_edge(edge):
        if edge.edge_type == edge.LINEAR:
            function_counts["linear"] += 1
            activation_counts[edge.activation_name] += 1
        elif edge.edge_type == edge.MLP:
            function_counts["mlp"] += 1
        elif edge.edge_type == edge.SOFT_TREE:
            function_counts["soft_tree"] += 1
        else:
            function_counts["unknown"] += 1

    for connection in task.scm.connections:
        for child in range(connection.out_width):
            method = int(connection.child_methods[child].item())
            child_method_counts[method_names.get(method, "unknown")] += 1
            if method == 0:
                parents = torch.where(connection.adj[:, child])[0]
                for parent in parents.tolist():
                    edge = connection.edges[parent][child]
                    if edge is not None:
                        record_scalar_edge(edge)
            elif method == 1:
                edge = connection.child_scalar_edges[child]
                if edge is not None:
                    record_scalar_edge(edge)
            elif method == 2:
                function_counts["joint_mlp"] += 1

    return dict(function_counts), dict(activation_counts), dict(child_method_counts)


def make_preprocessor(feature_type: np.ndarray) -> ColumnTransformer:
    feature_type = np.asarray(feature_type, dtype=int)
    continuous_indices = np.where(feature_type == 0)[0].tolist()
    categorical_indices = np.where(feature_type == 1)[0].tolist()
    transformers = []
    if continuous_indices:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                continuous_indices,
            )
        )
    if categorical_indices:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", make_one_hot_encoder()),
                    ]
                ),
                categorical_indices,
            )
        )
    if not transformers:
        raise ValueError("No columns were provided to the preprocessor.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models(feature_type: np.ndarray, seed: int):
    return {
        "dummy_prior": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(feature_type)),
                ("classifier", DummyClassifier(strategy="prior", random_state=seed)),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(feature_type)),
                (
                    "classifier",
                    LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(feature_type)),
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
                ("preprocessor", make_preprocessor(feature_type)),
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


def compute_auc(model, X: np.ndarray, y: np.ndarray) -> float:
    try:
        probabilities = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        if classes.size == 2:
            binary_y = (y == classes[1]).astype(int)
            if np.unique(binary_y).size < 2:
                return float("nan")
            return float(roc_auc_score(binary_y, probabilities[:, 1]))
        if np.unique(y).size != classes.size:
            return float("nan")
        return float(
            roc_auc_score(
                y,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        )
    except (AttributeError, ValueError):
        return float("nan")


def compute_logloss(model, X: np.ndarray, y: np.ndarray) -> float:
    try:
        probabilities = model.predict_proba(X)
        return float(log_loss(y, probabilities, labels=np.asarray(model.classes_)))
    except (AttributeError, ValueError):
        return float("nan")


def classification_metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    prediction = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "auc": compute_auc(model, X, y),
        "logloss": compute_logloss(model, X, y),
    }


def fit_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "classes_"):
        raise RuntimeError("Classifier does not have classes_ after fit.")
    return (
        classification_metrics(model, X_train, y_train),
        classification_metrics(model, X_test, y_test),
    )


def shuffled_label_control(X_train, y_train, X_test, y_test, feature_type, seed: int) -> float:
    rng = np.random.default_rng(seed)
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(feature_type)),
            (
                "classifier",
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
            ),
        ]
    )
    model.fit(X_train, rng.permutation(y_train))
    return float(balanced_accuracy_score(y_test, model.predict(X_test)))


def evaluate_subset(X_train, y_train, X_test, y_test, feature_type, selected_indices, seed: int):
    selected_indices = np.asarray(selected_indices, dtype=int)
    if selected_indices.size == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "auc": float("nan"),
            "logloss": float("nan"),
        }
    subset_type = feature_type[selected_indices]
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(subset_type)),
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
    model.fit(X_train[:, selected_indices], y_train)
    return classification_metrics(model, X_test[:, selected_indices], y_test)


def evaluate_one_table(task, experiment_name: str, table_id: int):
    X_train = to_numpy(task.X_train).astype(float)
    X_test = to_numpy(task.X_test).astype(float)
    y_train = to_numpy(task.y_train).astype(int)
    y_test = to_numpy(task.y_test).astype(int)
    info = task.info

    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)
    feature_type = to_numpy(info["feature_type"]).astype(int)
    feature_ids = to_numpy(info["feature_ids"]).astype(int)
    selected_dominant = to_numpy(info["selected_from_dominant_group"]).astype(bool)
    observation_names = list(info["feature_observation_type_names"])
    widths = to_numpy(info["layer_widths"]).astype(int).tolist()

    feature_layers = [global_node_id_to_layer(int(gid), widths)[0] for gid in feature_ids]
    layer_counts = Counter(feature_layers)
    function_counts, activation_counts, child_method_counts = collect_edge_diagnostics(task)

    errors = validate_graph_structure(task)
    errors.extend(validate_generated_data(X_train, y_train, X_test, y_test, info))
    if RUN_LATENT_SHAPE_CHECK:
        errors.extend(validate_latent_shapes(task, n_samples=32))

    shuffled_score = float("nan")
    if RUN_SHUFFLED_LABEL_CONTROL:
        shuffled_score = shuffled_label_control(
            X_train,
            y_train,
            X_test,
            y_test,
            feature_type,
            seed=900_000 + table_id,
        )

    num_classes = int(np.unique(y_full).size)
    d = int(X_train.shape[1])
    n_total = int(X_full.shape[0])
    n_continuous = int((feature_type == 0).sum())
    n_categorical = int((feature_type == 1).sum())
    missing_rate = float(np.isnan(X_full).mean())

    if PRINT_EVERY_TABLE:
        print("\n" + "=" * 120)
        print(
            f"{experiment_name.upper()} | TABLE {table_id:03d} | "
            f"classes={num_classes} | n={n_total} | d={d}"
        )
        print("=" * 120)
        print("Layer widths:", widths)
        print("Root priors:", info["root_prior_types"])
        print("Child methods:", child_method_counts)
        print("Scalar/joint function counts:", function_counts)
        if activation_counts:
            print("Linear activations:", activation_counts)
        print("Feature source layers:", dict(sorted(layer_counts.items())))
        print("Observation mechanisms:", dict(Counter(observation_names)))
        print("Target class counts:", class_count_dict(y_full))
        print(f"Feature types: continuous={n_continuous}, categorical={n_categorical}")
        print(f"Missing rate: {missing_rate:.2%}")
        print(f"Shuffled-label logistic control: bal_acc={shuffled_score:.4f}")
        if errors:
            print("\nVALIDATION ERRORS:")
            for message in errors:
                print(f"ERROR: {message}")
        else:
            print("Validation: PASS")

    model_rows = []
    for model_name, model in make_models(feature_type, seed=BASE_SEED + table_id).items():
        train_metrics, test_metrics = fit_and_evaluate(model, X_train, y_train, X_test, y_test)
        model_rows.append(
            {
                "experiment": experiment_name,
                "table_id": table_id,
                "num_classes": num_classes,
                "num_features": d,
                "model": model_name,
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_auc": test_metrics["auc"],
                "test_logloss": test_metrics["logloss"],
            }
        )
        if PRINT_EVERY_TABLE:
            print(
                f"{model_name:24s} | train_acc={train_metrics['accuracy']:.4f} | "
                f"test_acc={test_metrics['accuracy']:.4f} | "
                f"bal_acc={test_metrics['balanced_accuracy']:.4f} | "
                f"auc={test_metrics['auc']:.4f} | logloss={test_metrics['logloss']:.4f}"
            )

    subset_rows = []
    if RUN_FEATURE_SUBSET_CHECK:
        subset_definitions = {
            "all_features": np.arange(d, dtype=int),
            "dominant_features": np.where(selected_dominant)[0],
            "non_dominant_features": np.where(~selected_dominant)[0],
            "continuous_features": np.where(feature_type == 0)[0],
            "categorical_features": np.where(feature_type == 1)[0],
            "continuous_scalar_only": np.asarray(
                [
                    i
                    for i, name in enumerate(observation_names)
                    if name == "continuous_scalar" or "fallback" in name
                ],
                dtype=int,
            ),
            "prototype_only": np.asarray(
                [i for i, name in enumerate(observation_names) if name == "prototype_discretization"],
                dtype=int,
            ),
            "binning_only": np.asarray(
                [i for i, name in enumerate(observation_names) if name == "threshold_binning"],
                dtype=int,
            ),
        }
        if PRINT_EVERY_TABLE:
            print("\nRandom Forest subset checks:")
        for subset_name, indices in subset_definitions.items():
            metrics = evaluate_subset(
                X_train,
                y_train,
                X_test,
                y_test,
                feature_type,
                indices,
                seed=500_000 + table_id,
            )
            subset_rows.append(
                {
                    "experiment": experiment_name,
                    "table_id": table_id,
                    "num_classes": num_classes,
                    "subset": subset_name,
                    "num_selected": int(len(indices)),
                    **metrics,
                }
            )
            if PRINT_EVERY_TABLE:
                print(
                    f"{subset_name:26s} | features={len(indices):2d} | "
                    f"bal_acc={metrics['balanced_accuracy']:.4f} | auc={metrics['auc']:.4f}"
                )

    table_row = {
        "experiment": experiment_name,
        "table_id": table_id,
        "n_total": n_total,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "num_classes": num_classes,
        "num_features": d,
        "continuous_features": n_continuous,
        "categorical_features": n_categorical,
        "missing_rate": missing_rate,
        "dominant_selected_count": int(selected_dominant.sum()),
        "dominant_selected_ratio": float(selected_dominant.mean()),
        "shuffled_label_balanced_accuracy": shuffled_score,
        "num_validation_errors": len(errors),
        "validation_passed": len(errors) == 0,
        "layer_widths": "|".join(str(value) for value in widths),
        "root_prior_types": "|".join(info["root_prior_types"]),
        "child_edgewise_count": child_method_counts.get("edgewise", 0),
        "child_post_aggregate_count": child_method_counts.get("post_aggregate", 0),
        "child_joint_mlp_count": child_method_counts.get("joint_mlp", 0),
        "function_linear_count": function_counts.get("linear", 0),
        "function_mlp_count": function_counts.get("mlp", 0),
        "function_soft_tree_count": function_counts.get("soft_tree", 0),
        "function_joint_mlp_count": function_counts.get("joint_mlp", 0),
    }
    return table_row, model_rows, subset_rows


def build_experiment_summary(model_df: pd.DataFrame, table_df: pd.DataFrame) -> pd.DataFrame:
    dummy = (
        model_df[model_df["model"] == "dummy_prior"]
        [["experiment", "table_id", "test_balanced_accuracy"]]
        .rename(columns={"test_balanced_accuracy": "dummy_balanced_accuracy"})
    )
    merged = model_df.merge(dummy, on=["experiment", "table_id"], how="left")
    merged["lift_over_dummy"] = merged["test_balanced_accuracy"] - merged["dummy_balanced_accuracy"]

    table_info = (
        table_df.groupby("experiment")
        .agg(
            mean_shuffled_label_balanced_accuracy=("shuffled_label_balanced_accuracy", "mean"),
            tables_with_validation_errors=(
                "num_validation_errors",
                lambda values: int((values > 0).sum()),
            ),
            validation_pass_rate=("validation_passed", "mean"),
        )
        .reset_index()
    )

    summary = (
        merged.groupby(["experiment", "model"])
        .agg(
            mean_balanced_accuracy=("test_balanced_accuracy", "mean"),
            std_balanced_accuracy=("test_balanced_accuracy", "std"),
            median_balanced_accuracy=("test_balanced_accuracy", "median"),
            mean_auc=("test_auc", "mean"),
            mean_logloss=("test_logloss", "mean"),
            mean_lift_over_dummy=("lift_over_dummy", "mean"),
            fraction_above_dummy=("lift_over_dummy", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )
    return summary.merge(table_info, on="experiment", how="left")


def print_final_summary(summary_df: pd.DataFrame):
    print("\n" + "=" * 120)
    print("FINAL SUMMARY")
    print("=" * 120)
    columns = [
        "experiment",
        "model",
        "mean_balanced_accuracy",
        "std_balanced_accuracy",
        "median_balanced_accuracy",
        "mean_auc",
        "mean_logloss",
        "mean_lift_over_dummy",
        "fraction_above_dummy",
        "mean_shuffled_label_balanced_accuracy",
        "tables_with_validation_errors",
        "validation_pass_rate",
    ]
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.float_format",
        lambda value: f"{value:.4f}",
    ):
        print(summary_df[columns].to_string(index=False))


def main():
    warnings.filterwarnings("ignore")
    print_import_debug()
    all_table_rows = []
    all_model_rows = []
    all_subset_rows = []

    class_generator = torch.Generator(device="cpu")
    class_generator.manual_seed(BASE_SEED + 999_999)
    total_start = time.perf_counter()

    for experiment_name in EXPERIMENTS:
        print("\n" + "#" * 120)
        print(f"RUNNING EXPERIMENT: {experiment_name}")
        print("#" * 120)
        for table_id in range(NUM_TABLES_PER_EXPERIMENT):
            num_classes = int(
                torch.randint(
                    MIN_CLASSES,
                    MAX_CLASSES + 1,
                    (),
                    generator=class_generator,
                ).item()
            )
            generation_start = time.perf_counter()
            task = build_task(experiment_name, table_id, num_classes)
            generation_seconds = time.perf_counter() - generation_start
            table_row, model_rows, subset_rows = evaluate_one_table(task, experiment_name, table_id)
            table_row["generation_seconds"] = generation_seconds
            all_table_rows.append(table_row)
            all_model_rows.extend(model_rows)
            all_subset_rows.extend(subset_rows)

    table_df = pd.DataFrame(all_table_rows)
    model_df = pd.DataFrame(all_model_rows)
    subset_df = pd.DataFrame(all_subset_rows)
    summary_df = build_experiment_summary(model_df, table_df)

    model_df.to_csv(MODEL_CSV, index=False)
    table_df.to_csv(TABLE_CSV, index=False)
    subset_df.to_csv(SUBSET_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    config = {
        "base_seed": BASE_SEED,
        "num_tables_per_experiment": NUM_TABLES_PER_EXPERIMENT,
        "min_classes": MIN_CLASSES,
        "max_classes": MAX_CLASSES,
        "model_n_estimators": MODEL_N_ESTIMATORS,
        "latent_node_shape": "[N, 1]",
        "child_methods": {"0": "edgewise", "1": "post_aggregate", "2": "joint_mlp"},
        "base_task_kwargs": {
            key: str(value) if isinstance(value, torch.device) else value
            for key, value in BASE_TASK_KWARGS.items()
        },
        "experiments": EXPERIMENTS,
    }
    with CONFIG_JSON.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print_final_summary(summary_df)
    total_seconds = time.perf_counter() - total_start
    print("\n" + "=" * 120)
    print("RUN COMPLETE")
    print("=" * 120)
    print(f"Total runtime: {total_seconds:.2f} seconds")
    print("\nSaved files:")
    print(f"Model metrics : {MODEL_CSV.resolve()}")
    print(f"Table metrics : {TABLE_CSV.resolve()}")
    print(f"Subset metrics: {SUBSET_CSV.resolve()}")
    print(f"Summary       : {SUMMARY_CSV.resolve()}")
    print(f"Configuration : {CONFIG_JSON.resolve()}")


if __name__ == "__main__":
    main()

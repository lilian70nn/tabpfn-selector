# sanity_check/scm_v9_importance_test.py

import time
from collections import Counter
from pathlib import Path

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

from src.data.scm_task_v10 import WeightedMixedScalarSCMTask


# =============================================================================
# Configuration
# =============================================================================

BASE_SEED = 0

NUM_TABLES = 100

MIN_CLASSES = 2
MAX_CLASSES = 4

DEVICE = torch.device("cpu")

RF_ESTIMATORS = 300

TOP_K = 5

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_CSV = (
    OUTPUT_DIR
    / "scm_v10_importance_tables_residuals_penalty25_imp1.csv"
)

MODEL_CSV = (
    OUTPUT_DIR
    / "scm_v10_importance_models_residuals_penalty25_imp1.csv"
)

FEATURE_CSV = (
    OUTPUT_DIR
    / "scm_v10_feature_importance_residuals_penalty25_imp1.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "scm_v10_importance_summary_residuals_penalty25_imp1.csv"
)


# =============================================================================
# Generator configuration
# =============================================================================
TASK_KWARGS = dict(
    n_min=400,
    n_max=512,
    d_min=8,
    d_max=16,
    test_frac=0.15,
    p_missing=0.05,
    num_roots=8,
    num_layers=5,
    hidden_width_min=6,
    hidden_width_max=10,
    final_width=1,
    connection_probs=(0.20, 0.20, 0.30, 0.85),
    edge_weight_concentration=0.30,
    latent_noise_scale=0.0,
    sampling_penalty=0.25,
    observation_noise_scale=0.03,
    observation_type_probs=(0.70, 0.15, 0.15),
    categorical_cardinalities=(2, 3, 4, 5, 6),
    categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    min_samples_per_category=8,
    min_component_weight=0.05,
    prototype_max_attempts=8,
    prototype_min_separation=1.0,
    binning_jitter=0.20,
    source_prior_probs=(0.45, 0.20, 0.15, 0.05),
    linear_activation_prob=0.60,
    small_mlp_prob=0.25,
    soft_tree_prob=0.15,
    small_mlp_hidden_dim=None,
    soft_tree_depth=2,
    soft_tree_temperature=0.5,
    device=DEVICE,
)


# =============================================================================
# Helpers
# =============================================================================

def to_numpy(value):
    if torch.is_tensor(value):
        return (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(value)


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


def class_count_dict(y):
    labels, counts = np.unique(
        y,
        return_counts=True,
    )

    return {
        int(label): int(count)
        for label, count
        in zip(labels, counts)
    }


def global_node_id_to_layer(
    global_id,
    widths,
):
    start = 0

    for layer_idx, width in enumerate(
        widths
    ):
        end = (
            start
            + int(width)
        )

        if (
            start
            <= global_id
            < end
        ):
            return (
                layer_idx,
                global_id - start,
            )

        start = end

    raise ValueError(
        f"global_id={global_id} "
        f"is outside widths={widths}"
    )


def safe_spearman(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    mask = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    a = a[mask]
    b = b[mask]

    if len(a) < 2:
        return float("nan")

    if (
        np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return float("nan")

    a_rank = pd.Series(
        a
    ).rank(
        method="average"
    )

    b_rank = pd.Series(
        b
    ).rank(
        method="average"
    )

    return float(
        a_rank.corr(
            b_rank,
            method="pearson",
        )
    )


def top_k_overlap(
    a,
    b,
    k,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    indices = np.where(
        valid
    )[0]

    if indices.size == 0:
        return float("nan")

    k = min(
        int(k),
        indices.size,
    )

    if k <= 0:
        return float("nan")

    a_valid = a[
        indices
    ]

    b_valid = b[
        indices
    ]

    a_top = set(
        indices[
            np.argsort(
                a_valid
            )[-k:]
        ].tolist()
    )

    b_top = set(
        indices[
            np.argsort(
                b_valid
            )[-k:]
        ].tolist()
    )

    return (
        len(
            a_top
            & b_top
        )
        / k
    )


# =============================================================================
# Preprocessing
# =============================================================================

def make_preprocessor(
    feature_type,
):
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
            "No valid feature columns."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# =============================================================================
# Models
# =============================================================================

def make_rf(
    feature_type,
    seed,
):
    return Pipeline(
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
                    n_estimators=RF_ESTIMATORS,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_models(
    feature_type,
    seed,
):
    return {
        "dummy": Pipeline(
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
                    ),
                ),
            ]
        ),

        "logistic": Pipeline(
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
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),

        "random_forest": make_rf(
            feature_type,
            seed,
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
                        n_estimators=RF_ESTIMATORS,
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


# =============================================================================
# Metrics
# =============================================================================

def compute_auc(
    model,
    X,
    y,
):
    try:
        probabilities = model.predict_proba(
            X
        )

        classes = np.asarray(
            model.classes_
        )

        if classes.size == 2:
            positive = classes[1]

            binary_y = (
                y == positive
            ).astype(int)

            if np.unique(
                binary_y
            ).size < 2:
                return float("nan")

            return float(
                roc_auc_score(
                    binary_y,
                    probabilities[:, 1],
                )
            )

        if (
            np.unique(y).size
            != classes.size
        ):
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

    except (
        ValueError,
        AttributeError,
    ):
        return float("nan")


def compute_logloss(
    model,
    X,
    y,
):
    try:
        probabilities = model.predict_proba(
            X
        )

        return float(
            log_loss(
                y,
                probabilities,
                labels=np.asarray(
                    model.classes_
                ),
            )
        )

    except (
        ValueError,
        AttributeError,
    ):
        return float("nan")


def evaluate_model(
    model,
    X,
    y,
):
    prediction = model.predict(
        X
    )

    return {
        "accuracy": float(
            accuracy_score(
                y,
                prediction,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                prediction,
            )
        ),

        "macro_f1": float(
            f1_score(
                y,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),

        "auc": compute_auc(
            model,
            X,
            y,
        ),

        "logloss": compute_logloss(
            model,
            X,
            y,
        ),
    }


# =============================================================================
# Empirical feature importance
# =============================================================================

def single_feature_relevance(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_type,
    seed,
):
    d = X_train.shape[1]

    scores = np.full(
        d,
        np.nan,
        dtype=float,
    )

    for column in range(d):
        selected_type = np.asarray(
            [
                feature_type[
                    column
                ]
            ],
            dtype=int,
        )

        model = make_rf(
            selected_type,
            seed + column,
        )

        model.fit(
            X_train[
                :,
                [column],
            ],
            y_train,
        )

        prediction = model.predict(
            X_test[
                :,
                [column],
            ]
        )

        scores[column] = (
            balanced_accuracy_score(
                y_test,
                prediction,
            )
        )

    return scores


def permutation_importance_raw(
    fitted_model,
    X_test,
    y_test,
    seed,
    repeats=5,
):
    rng = np.random.default_rng(
        seed
    )

    baseline_prediction = (
        fitted_model.predict(
            X_test
        )
    )

    baseline_score = (
        balanced_accuracy_score(
            y_test,
            baseline_prediction,
        )
    )

    d = X_test.shape[1]

    importances = np.zeros(
        d,
        dtype=float,
    )

    for column in range(d):
        drops = []

        for _ in range(
            repeats
        ):
            X_permuted = (
                X_test.copy()
            )

            permutation = rng.permutation(
                X_permuted.shape[0]
            )

            X_permuted[
                :,
                column,
            ] = X_permuted[
                permutation,
                column,
            ]

            prediction = (
                fitted_model.predict(
                    X_permuted
                )
            )

            score = (
                balanced_accuracy_score(
                    y_test,
                    prediction,
                )
            )

            drops.append(
                baseline_score
                - score
            )

        importances[
            column
        ] = np.mean(
            drops
        )

    return (
        importances,
        baseline_score,
    )


def drop_column_importance(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_type,
    baseline_score,
    seed,
):
    d = X_train.shape[1]

    importances = np.zeros(
        d,
        dtype=float,
    )

    if d <= 1:
        return np.full(
            d,
            np.nan,
            dtype=float,
        )

    all_indices = np.arange(
        d
    )

    for column in range(d):
        keep = all_indices[
            all_indices != column
        ]

        reduced_feature_type = (
            feature_type[
                keep
            ]
        )

        model = make_rf(
            reduced_feature_type,
            seed + column,
        )

        model.fit(
            X_train[
                :,
                keep,
            ],
            y_train,
        )

        prediction = model.predict(
            X_test[
                :,
                keep,
            ]
        )

        score = (
            balanced_accuracy_score(
                y_test,
                prediction,
            )
        )

        importances[
            column
        ] = (
            baseline_score
            - score
        )

    return importances


# =============================================================================
# One table
# =============================================================================

def evaluate_table(
    table_id,
    num_classes,
):
    task = WeightedMixedScalarSCMTask(
        num_classes=num_classes,

        dag_seed=(
            BASE_SEED
            + table_id
        ),

        aleatoric_seed=(
            100_000
            + BASE_SEED
            + table_id
        ),

        x_seed=(
            200_000
            + BASE_SEED
            + table_id
        ),

        **TASK_KWARGS,
    )

    X_train = to_numpy(
        task.X_train
    ).astype(float)

    X_test = to_numpy(
        task.X_test
    ).astype(float)

    y_train = to_numpy(
        task.y_train
    ).astype(int)

    y_test = to_numpy(
        task.y_test
    ).astype(int)

    info = task.info

    X_full = np.concatenate(
        (
            X_train,
            X_test,
        ),
        axis=0,
    )

    y_full = np.concatenate(
        (
            y_train,
            y_test,
        ),
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

    observation_names = list(
        info[
            "feature_observation_type_names"
        ]
    )

    widths = to_numpy(
        info["layer_widths"]
    ).astype(int).tolist()

    # -------------------------------------------------------------------------
    # GT importance
    # -------------------------------------------------------------------------

    # gt_strength = to_numpy(
    #     info["feature_strength"]
    # ).astype(float)

    gt_importance = to_numpy(
        info["feature_importance"]
    ).astype(float)

    # # Safety normalization.
    # gt_strength = np.nan_to_num(
    #     gt_strength,
    #     nan=0.0,
    #     posinf=0.0,
    #     neginf=0.0,
    # )

    gt_importance = np.nan_to_num(
        gt_importance,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if gt_importance.sum() > 0:
        gt_importance = (
            gt_importance
            / gt_importance.sum()
        )

    # -------------------------------------------------------------------------
    # Feature source layers
    # -------------------------------------------------------------------------

    feature_layers = []
    feature_node_indices = []

    for global_id in feature_ids:
        layer_idx, node_idx = (
            global_node_id_to_layer(
                int(global_id),
                widths,
            )
        )

        feature_layers.append(
            layer_idx
        )

        feature_node_indices.append(
            node_idx
        )

    # -------------------------------------------------------------------------
    # Print basic information
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)

    print(
        f"TABLE {table_id:03d} "
        f"| classes={num_classes} "
        f"| n={len(y_full)} "
        f"| d={X_full.shape[1]}"
    )

    print("=" * 110)

    print(
        "Layer widths:",
        widths,
    )

    print(
        "Class counts:",
        class_count_dict(
            y_full
        ),
    )

    print(
        "Feature types:",
        {
            "continuous":
            int(
                (
                    feature_type == 0
                ).sum()
            ),

            "categorical":
            int(
                (
                    feature_type == 1
                ).sum()
            ),
        },
    )

    print(
        "Observation mechanisms:",
        dict(
            Counter(
                observation_names
            )
        ),
    )

    print(
        "Feature source layers:",
        dict(
            Counter(
                feature_layers
            )
        ),
    )

    print(
        "Missing rate:",
        f"{np.isnan(X_full).mean():.2%}",
    )

    print(
        "GT importance:",
        np.round(
            gt_importance,
            4,
        ).tolist(),
    )

    # -------------------------------------------------------------------------
    # Predictive models
    # -------------------------------------------------------------------------

    model_rows = []

    models = make_models(
        feature_type=feature_type,
        seed=(
            300_000
            + table_id
        ),
    )

    fitted_rf = None
    rf_test_balanced_accuracy = None

    print()
    print("Predictive models:")

    for model_name, model in models.items():
        model.fit(
            X_train,
            y_train,
        )

        train_metrics = evaluate_model(
            model,
            X_train,
            y_train,
        )

        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print(
            f"{model_name:18s} | "
            f"train_bal="
            f"{train_metrics['balanced_accuracy']:.4f} | "
            f"test_bal="
            f"{test_metrics['balanced_accuracy']:.4f} | "
            f"auc="
            f"{test_metrics['auc']:.4f} | "
            f"logloss="
            f"{test_metrics['logloss']:.4f}"
        )

        model_rows.append(
            {
                "table_id": table_id,
                "num_classes": num_classes,
                "model": model_name,

                "train_accuracy":
                train_metrics[
                    "accuracy"
                ],

                "train_balanced_accuracy":
                train_metrics[
                    "balanced_accuracy"
                ],

                "test_accuracy":
                test_metrics[
                    "accuracy"
                ],

                "test_balanced_accuracy":
                test_metrics[
                    "balanced_accuracy"
                ],

                "test_macro_f1":
                test_metrics[
                    "macro_f1"
                ],

                "test_auc":
                test_metrics[
                    "auc"
                ],

                "test_logloss":
                test_metrics[
                    "logloss"
                ],
            }
        )

        if (
            model_name
            == "random_forest"
        ):
            fitted_rf = model

            rf_test_balanced_accuracy = (
                test_metrics[
                    "balanced_accuracy"
                ]
            )

    # -------------------------------------------------------------------------
    # Importance estimation
    # -------------------------------------------------------------------------

    single_scores = (
        single_feature_relevance(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_type=feature_type,
            seed=(
                400_000
                + table_id * 100
            ),
        )
    )

    permutation_imp, rf_baseline = (
        permutation_importance_raw(
            fitted_model=fitted_rf,
            X_test=X_test,
            y_test=y_test,
            seed=(
                500_000
                + table_id
            ),
            repeats=5,
        )
    )

    drop_imp = (
        drop_column_importance(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_type=feature_type,
            baseline_score=rf_baseline,
            seed=(
                600_000
                + table_id * 100
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Compare GT vs empirical measures
    # -------------------------------------------------------------------------

    corr_single = safe_spearman(
        gt_importance,
        single_scores,
    )

    corr_perm = safe_spearman(
        gt_importance,
        permutation_imp,
    )

    corr_drop = safe_spearman(
        gt_importance,
        drop_imp,
    )

    overlap_single = top_k_overlap(
        gt_importance,
        single_scores,
        TOP_K,
    )

    overlap_perm = top_k_overlap(
        gt_importance,
        permutation_imp,
        TOP_K,
    )

    overlap_drop = top_k_overlap(
        gt_importance,
        drop_imp,
        TOP_K,
    )

    # -------------------------------------------------------------------------
    # Per-feature rows
    # -------------------------------------------------------------------------

    feature_rows = []

    for column in range(
        X_train.shape[1]
    ):
        feature_rows.append(
            {
                "table_id":
                table_id,

                "num_classes":
                num_classes,

                "feature_column":
                column,

                "global_node_id":
                int(
                    feature_ids[
                        column
                    ]
                ),

                "layer":
                int(
                    feature_layers[
                        column
                    ]
                ),

                "node_in_layer":
                int(
                    feature_node_indices[
                        column
                    ]
                ),

                "feature_type":
                (
                    "categorical"
                    if feature_type[
                        column
                    ] == 1
                    else "continuous"
                ),

                "cardinality":
                int(
                    cardinality[
                        column
                    ]
                ),

                "observation_type":
                observation_names[
                    column
                ],

                # "gt_strength":
                # float(
                #     gt_strength[
                #         column
                #     ]
                # ),

                "gt_importance":
                float(
                    gt_importance[
                        column
                    ]
                ),

                "single_feature_balanced_accuracy":
                float(
                    single_scores[
                        column
                    ]
                ),

                "permutation_importance":
                float(
                    permutation_imp[
                        column
                    ]
                ),

                "drop_column_importance":
                float(
                    drop_imp[
                        column
                    ]
                ),
            }
        )

    feature_df = pd.DataFrame(
        feature_rows
    )

    display_df = (
        feature_df[
            [
                "feature_column",
                "global_node_id",
                "layer",
                "feature_type",
                "observation_type",
                "gt_importance",
                "single_feature_balanced_accuracy",
                "permutation_importance",
                "drop_column_importance",
            ]
        ]
        .sort_values(
            "gt_importance",
            ascending=False,
        )
    )

    print()
    print("Feature importance comparison:")

    with pd.option_context(
        "display.max_rows",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(
            display_df.to_string(
                index=False
            )
        )

    print()
    print(
        "GT comparison:"
    )

    print(
        f"Spearman GT vs single-feature : "
        f"{corr_single:.4f}"
    )

    print(
        f"Spearman GT vs permutation    : "
        f"{corr_perm:.4f}"
    )

    print(
        f"Spearman GT vs drop-column    : "
        f"{corr_drop:.4f}"
    )

    print(
        f"Top-{TOP_K} GT vs single overlap: "
        f"{overlap_single:.4f}"
    )

    print(
        f"Top-{TOP_K} GT vs perm overlap  : "
        f"{overlap_perm:.4f}"
    )

    print(
        f"Top-{TOP_K} GT vs drop overlap  : "
        f"{overlap_drop:.4f}"
    )

    table_row = {
        "table_id":
        table_id,

        "num_classes":
        num_classes,

        "n_total":
        int(
            len(
                y_full
            )
        ),

        "n_train":
        int(
            len(
                y_train
            )
        ),

        "n_test":
        int(
            len(
                y_test
            )
        ),

        "num_features":
        int(
            X_train.shape[1]
        ),

        "num_continuous":
        int(
            (
                feature_type == 0
            ).sum()
        ),

        "num_categorical":
        int(
            (
                feature_type == 1
            ).sum()
        ),

        "missing_rate":
        float(
            np.isnan(
                X_full
            ).mean()
        ),

        "rf_balanced_accuracy":
        float(
            rf_test_balanced_accuracy
        ),

        "spearman_gt_single":
        corr_single,

        "spearman_gt_permutation":
        corr_perm,

        "spearman_gt_drop_column":
        corr_drop,

        "topk_gt_single":
        overlap_single,

        "topk_gt_permutation":
        overlap_perm,

        "topk_gt_drop_column":
        overlap_drop,
    }

    return (
        table_row,
        model_rows,
        feature_rows,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    total_start = time.perf_counter()

    all_table_rows = []
    all_model_rows = []
    all_feature_rows = []

    class_generator = (
        torch.Generator(
            device="cpu"
        )
    )

    class_generator.manual_seed(
        BASE_SEED
        + 999_999
    )

    for table_id in range(
        NUM_TABLES
    ):
        num_classes = int(
            torch.randint(
                MIN_CLASSES,
                MAX_CLASSES + 1,
                (),
                generator=class_generator,
            ).item()
        )

        (
            table_row,
            model_rows,
            feature_rows,
        ) = evaluate_table(
            table_id=table_id,
            num_classes=num_classes,
        )

        all_table_rows.append(
            table_row
        )

        all_model_rows.extend(
            model_rows
        )

        all_feature_rows.extend(
            feature_rows
        )

    table_df = pd.DataFrame(
        all_table_rows
    )

    model_df = pd.DataFrame(
        all_model_rows
    )

    feature_df = pd.DataFrame(
        all_feature_rows
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = {
        "num_tables":
        len(
            table_df
        ),

        "mean_rf_balanced_accuracy":
        table_df[
            "rf_balanced_accuracy"
        ].mean(),

        "median_rf_balanced_accuracy":
        table_df[
            "rf_balanced_accuracy"
        ].median(),

        "mean_spearman_gt_single":
        table_df[
            "spearman_gt_single"
        ].mean(),

        "median_spearman_gt_single":
        table_df[
            "spearman_gt_single"
        ].median(),

        "mean_spearman_gt_permutation":
        table_df[
            "spearman_gt_permutation"
        ].mean(),

        "median_spearman_gt_permutation":
        table_df[
            "spearman_gt_permutation"
        ].median(),

        "mean_spearman_gt_drop_column":
        table_df[
            "spearman_gt_drop_column"
        ].mean(),

        "median_spearman_gt_drop_column":
        table_df[
            "spearman_gt_drop_column"
        ].median(),

        "mean_topk_gt_single":
        table_df[
            "topk_gt_single"
        ].mean(),

        "mean_topk_gt_permutation":
        table_df[
            "topk_gt_permutation"
        ].mean(),

        "mean_topk_gt_drop_column":
        table_df[
            "topk_gt_drop_column"
        ].mean(),
    }

    summary_df = pd.DataFrame(
        [
            summary
        ]
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    table_df.to_csv(
        TABLE_CSV,
        index=False,
    )

    model_df.to_csv(
        MODEL_CSV,
        index=False,
    )

    feature_df.to_csv(
        FEATURE_CSV,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(
            summary_df.to_string(
                index=False
            )
        )

    print()
    print(
        "Correlation by table:"
    )

    with pd.option_context(
        "display.max_rows",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(
            table_df[
                [
                    "table_id",
                    "num_classes",
                    "num_features",
                    "rf_balanced_accuracy",
                    "spearman_gt_single",
                    "spearman_gt_permutation",
                    "spearman_gt_drop_column",
                    "topk_gt_single",
                    "topk_gt_permutation",
                    "topk_gt_drop_column",
                ]
            ].to_string(
                index=False
            )
        )

    elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 110)
    print(
        f"Finished in "
        f"{elapsed:.2f} seconds"
    )

    print()
    print(
        "Saved:"
    )

    print(
        TABLE_CSV.resolve()
    )

    print(
        MODEL_CSV.resolve()
    )

    print(
        FEATURE_CSV.resolve()
    )

    print(
        SUMMARY_CSV.resolve()
    )


if __name__ == "__main__":
    main()
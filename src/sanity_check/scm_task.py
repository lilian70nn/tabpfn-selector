import torch
import numpy as np
import pandas as pd

from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v10 import WeightedMixedScalarSCMTask


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance


def ancestor_descendant_ratio(info):
    feature_ids = info["feature_ids"].cpu().numpy().astype(int)
    widths = info["layer_widths"].cpu().numpy().astype(int)
    adjacency = [a.cpu().numpy() for a in info["adjacency_matrices"]]
    offsets = np.cumsum(np.r_[0, widths])

    selected_nodes = []
    for feature_id in feature_ids:
        layer = np.searchsorted(offsets[1:], feature_id, side="right")
        node = feature_id - offsets[layer]
        selected_nodes.append((layer, node))

    def is_ancestor(a, b):
        layer_a, node_a = a
        layer_b, node_b = b
        if layer_a >= layer_b:
            return False

        reachable = {node_a}
        for layer in range(layer_a, layer_b):
            reachable = {child for node in reachable for child in np.where(adjacency[layer][node])[0]}
            if not reachable:
                return False

        return node_b in reachable

    related = 0
    total = 0
    for i in range(len(selected_nodes)):
        for j in range(i + 1, len(selected_nodes)):
            total += 1
            if is_ancestor(selected_nodes[i], selected_nodes[j]) or is_ancestor(selected_nodes[j], selected_nodes[i]):
                related += 1

    return related / total if total > 0 else 0.0



def run_scm_sanity_check(dataset, dataset_no_penalty):

    rows = []

    for i in range(len(dataset)):
        task = dataset[i]
        task_no_penalty = dataset_no_penalty[i] 
        X_train = task.X_train.cpu().numpy()
        y_train = task.y_train.cpu().numpy()
        X_test = task.X_test.cpu().numpy()
        y_test = task.y_test.cpu().numpy()
        info = task.info
        feature_type = info["feature_type"].cpu().numpy()
        cardinality = info["cardinality"].cpu().numpy()
        obs_type = info["feature_observation_type_ids"].cpu().numpy()
        feature_ids = info["feature_ids"].cpu().numpy()
        widths = info["layer_widths"].cpu().numpy()
        gt = info["feature_importance"].cpu().numpy()

        n = len(X_train) + len(X_test)
        d = X_train.shape[1]
        is_classification = task.n_classes is not None
        task_kind = "classification" if is_classification else "regression"

        X_all = np.concatenate([X_train, X_test], axis=0)
        row = {
            "dataset_id": i,
            "task_kind": task_kind,
            "n": n,
            "d": d,
            "missing_rate": float(np.isnan(X_all).mean()),
            "continuous_ratio": float((feature_type == 0).mean()),
            "categorical_ratio": float((feature_type == 1).mean()),
        }

        if is_classification:
            row["n_classes"] = int(task.n_classes)
            labels, counts = np.unique(np.concatenate([y_train, y_test]), return_counts=True)
            class_props = counts / counts.sum()
            row["class_ratio"] = class_props.tolist()
        else:
            row["n_classes"] = np.nan
            row["y_mean"] = float(np.mean(np.concatenate([y_train, y_test])))
            row["y_std"] = float(np.std(np.concatenate([y_train, y_test])))

        row["obs_continuous_ratio"] = float((obs_type == 0).mean())
        row["obs_prototype_ratio"] = float((obs_type == 1).mean())
        row["obs_binning_ratio"] = float((obs_type == 2).mean())
        row["ancestor_descendant_ratio_penalty_025"] = ancestor_descendant_ratio(task.info)
        row["ancestor_descendant_ratio_penalty_100"] = ancestor_descendant_ratio(task_no_penalty.info)

        offsets = np.cumsum(np.r_[0, widths])
        selected_layers = np.array([
            np.searchsorted(offsets[1:], feature_id, side="right")
            for feature_id in feature_ids
        ])

        for layer in range(len(widths) - 1):
            row[f"layer_{layer}_ratio"] = float((selected_layers == layer).mean())

        continuous_idx = np.where(feature_type == 0)[0]
        categorical_idx = np.where(feature_type == 1)[0]

        transformers = []
        if len(continuous_idx) > 0:
            transformers.append(("continuous", SimpleImputer(strategy="median"), continuous_idx))
        if len(categorical_idx) > 0:
            categorical_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
            transformers.append(("categorical", categorical_pipeline, categorical_idx))
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        if is_classification:
            models = {
                "dummy": DummyClassifier(strategy="prior"),
                "logistic": LogisticRegression(max_iter=3000, class_weight="balanced"),
                "random_forest": RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=0,
                    n_jobs=-1,
                ),
                "extra_trees": ExtraTreesClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=0,
                    n_jobs=-1,
                ),
            }
            for model_name, estimator in models.items():
                model = make_pipeline(preprocessor, estimator)
                model.fit(X_train, y_train)
                if model_name == "random_forest":
                    fitted_rf = model
                y_pred = model.predict(X_test)
                row[f"{model_name}_balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
                row[f"{model_name}_macro_f1"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

                y_prob = model.predict_proba(X_test)
                row[f"{model_name}_log_loss"] = float(log_loss(y_test, y_prob, labels=model.classes_))
                if task.n_classes == 2:
                    row[f"{model_name}_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    row[f"{model_name}_auc"] = float(
                        roc_auc_score(
                            y_test,
                            y_prob,
                            multi_class="ovr",
                            average="macro",
                            labels=model.classes_,
                        )
                    )
        else:
            models = {
                "dummy": DummyRegressor(strategy="mean"),
                "ridge": Ridge(alpha=1.0),
                "random_forest": RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=0,
                    n_jobs=-1,
                ),
                "extra_trees": ExtraTreesRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=0,
                    n_jobs=-1,
                ),
            }
            for model_name, estimator in models.items():
                model = make_pipeline(preprocessor, estimator)
                model.fit(X_train, y_train)
                if model_name == "random_forest":
                    fitted_rf = model
                y_pred = model.predict(X_test)
                row[f"{model_name}_r2"] = float(r2_score(y_test, y_pred))
                row[f"{model_name}_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                row[f"{model_name}_mae"] = float(mean_absolute_error(y_test, y_pred))


        single_feature_score = np.zeros(d)
        for j in range(d):
            X_train_j = X_train[:, [j]]
            X_test_j = X_test[:, [j]]

            if feature_type[j] == 0:
                preprocessor_j = SimpleImputer(strategy="median")
            else:
                preprocessor_j = make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore"),
                )

            if is_classification:
                estimator_j = RandomForestClassifier(
                    n_estimators=150,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=0,
                    n_jobs=-1,
                )
            else:
                estimator_j = RandomForestRegressor(
                    n_estimators=150,
                    min_samples_leaf=2,
                    random_state=0,
                    n_jobs=-1,
                )

            model_j = make_pipeline(preprocessor_j, estimator_j)
            model_j.fit(X_train_j, y_train)
            y_pred_j = model_j.predict(X_test_j)

            if is_classification:
                single_feature_score[j] = balanced_accuracy_score(y_test, y_pred_j)
            else:
                single_feature_score[j] = r2_score(y_test, y_pred_j)


        scoring = "balanced_accuracy" if is_classification else "r2"
        perm_result = permutation_importance(
            fitted_rf,
            X_test,
            y_test,
            scoring=scoring,
            n_repeats=5,
            random_state=0,
            n_jobs=-1,
        )

        permutation_imp = perm_result.importances_mean

        row["spearman_gt_single"] = float(spearmanr(gt, single_feature_score).statistic)
        row["spearman_gt_permutation"] = float(spearmanr(gt, permutation_imp).statistic)

        top_k = min(5, d)
        gt_top = set(np.argsort(gt)[-top_k:])
        single_top = set(np.argsort(single_feature_score)[-top_k:])
        permutation_top = set(np.argsort(permutation_imp)[-top_k:])

        row["topk_gt_single"] = len(gt_top & single_top) / top_k
        row["topk_gt_permutation"] = len(gt_top & permutation_top) / top_k

        row["gt_importance"] = gt.tolist()
        row["single_feature_importance"] = single_feature_score.tolist()
        row["permutation_importance"] = permutation_imp.tolist()


        rows.append(row)

    df = pd.DataFrame(rows)
    return df



if __name__ == "__main__":

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
        device=torch.device("cpu"),
    )

    base_seeds = [0, 10_000, 20_000, 30_000, 40_000]
    num_tasks_per_seed = 100

    all_dfs = []

    for base_seed in base_seeds:
        print(f"\n===== BASE SEED {base_seed} =====")

        task_kwargs_025 = TASK_KWARGS.copy()
        task_kwargs_025["sampling_penalty"] = 0.25

        task_kwargs_100 = TASK_KWARGS.copy()
        task_kwargs_100["sampling_penalty"] = 1.0

        dataset_025 = SyntheticTaskDataset(
            num_tasks=num_tasks_per_seed,
            task_factory=WeightedMixedScalarSCMTask,
            task_kind="classification",
            min_classes=2,
            max_classes=4,
            base_seed=base_seed,
            task_kwargs=task_kwargs_025,
        )

        dataset_100 = SyntheticTaskDataset(
            num_tasks=num_tasks_per_seed,
            task_factory=WeightedMixedScalarSCMTask,
            task_kind="classification",
            min_classes=2,
            max_classes=4,
            base_seed=base_seed,
            task_kwargs=task_kwargs_100,
        )

        df_seed = run_scm_sanity_check(dataset_025, dataset_100)
        df_seed["base_seed"] = base_seed
        all_dfs.append(df_seed)

    df = pd.concat(all_dfs, ignore_index=True)

    summary = df.groupby("n_classes").mean(numeric_only=True)
    summary.to_csv("scm_v10_sanity_summary.csv")

    seed_summary = df.groupby(["base_seed", "n_classes"]).mean(numeric_only=True)
    seed_summary.to_csv("scm_v10_sanity_summary_by_seed.csv")

    df.to_csv("scm_v10_sanity_all_tables.csv", index=False)

    print("\nOVERALL SUMMARY")
    print(summary)

    print("\nSUMMARY BY SEED")
    print(seed_summary)
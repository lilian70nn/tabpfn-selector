import numpy as np
import pandas as pd
import torch

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from src.data.scm_task_v2.task import SCMTask


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_preprocessor(feature_type):
    feature_type = np.asarray(feature_type).astype(bool)
    categorical_idx = np.where(feature_type)[0].tolist()
    continuous_idx = np.where(~feature_type)[0].tolist()

    transformers = []

    if continuous_idx:
        transformers.append((
            "continuous",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            continuous_idx,
        ))

    if categorical_idx:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_idx,
        ))

    return ColumnTransformer(transformers)


def evaluate_task(task):
    X_train = to_numpy(task.X_train)
    X_test = to_numpy(task.X_test)

    y_train = to_numpy(task.info["target_latent_train"]).reshape(-1)
    y_test = to_numpy(task.info["target_latent_test"]).reshape(-1)

    feature_type = to_numpy(task.info["feature_type"]).astype(bool)

    model = Pipeline([
        ("preprocessor", build_preprocessor(feature_type)),
        ("model", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return r2_score(y_train, pred_train), r2_score(y_test, pred_test)


def get_task_info(task):
    info = task.info

    feature_type = to_numpy(info["feature_type"]).astype(bool)
    importance = to_numpy(info["feature_importance"]).reshape(-1)

    # categorical feature ratio
    cat_ratio = float(feature_type.mean())

    # importance concentration
    importance_max = float(importance.max())
    importance_top3 = float(np.sort(importance)[-3:].sum()) if len(importance) >= 3 else float(importance.sum())

    # target parent count
    target_connection = task.scm.connections[-1]
    target_parents = torch.where(target_connection.adj[:, 0])[0]
    target_parent_count = int(target_parents.numel())

    # target symbolic program
    target_function = target_connection.child_functions[0]
    target_program = None if target_function is None else target_function.program

    # actual sampled latent noise
    latent_noise = info.get("sampled_latent_noise_scale", np.nan)
    if isinstance(latent_noise, torch.Tensor):
        if latent_noise.numel() == 1:
            latent_noise = float(latent_noise.item())
        else:
            latent_noise = str(latent_noise.detach().cpu().tolist())

    return {
        "cat_ratio": cat_ratio,
        "importance_max": importance_max,
        "importance_top3": importance_top3,
        "target_parent_count": target_parent_count,
        "latent_noise": latent_noise,
        "target_program": str(target_program),
    }


def main():
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

        "connection_probs": (
            (0.20, 0.35),
            (0.45, 0.65),
        ),

        "source_prior_probs": (
            0.55,
            0.20,
            0.15,
            0.10,
        ),

        "arity_probs": (
            2.5,
            3.0,
            3.0,
        ),

        "unary_op_probs": (
            1.0,
            1.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.5,
        ),

        "binary_op_probs":(3.0, 2.5, 1.5, 2.0),

        "ternary_op_probs": (4.0, 0.75, 0.75, 2.0),

        "observation_type_probs": (
            7.0,
            1.5,
            1.5,
        ),

        "latent_noise_scale": (
            0.0,
            0.03,
        ),

        "scale_min": 0.25,
        "scale_max": 4.0,

        "categorical_cardinalities": (
            2,
            3,
            4,
            5,
            6,
        ),

        "categorical_cardinality_probs": (
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),

        "min_samples_per_category": 8,
        "min_component_weight": 0.05,
        "observation_noise_scale": 0.03,
    }

    num_tasks = 100
    num_classes = 4

    rows = []

    for idx in range(num_tasks):
        seed = 100000 * num_classes + idx

        task = SCMTask(
            **prior,
            num_classes=num_classes,
            dag_seed=seed,
            x_seed=seed + 1,
            aleatoric_seed=seed + 2,
        )

        is_valid = task.info.get("is_valid", True)

        if hasattr(is_valid, "item"):
            is_valid = is_valid.item()

        if not bool(is_valid):
            continue

        train_r2, test_r2 = evaluate_task(task)
        task_info = get_task_info(task)

        row = {
            "task": idx,
            "train_r2": train_r2,
            "test_r2": test_r2,
            **task_info,
        }

        rows.append(row)

        print(
            f"\r{idx + 1}/{num_tasks}, "
            f"valid={len(rows)}, "
            f"test R2={test_r2:.3f}",
            end="",
            flush=True,
        )

    print()

    df = pd.DataFrame(rows)
    df = df.sort_values("test_r2", ascending=True)

    df.to_csv("target_latent_diagnostics.csv", index=False)

    print("\n===== Overall =====")
    print(f"tasks:          {len(df)}")
    print(f"mean test R2:   {df['test_r2'].mean():.4f}")
    print(f"median test R2: {df['test_r2'].median():.4f}")
    print(f"min test R2:    {df['test_r2'].min():.4f}")
    print(f"max test R2:    {df['test_r2'].max():.4f}")

    print("\n===== Worst 20 tasks =====")

    columns = [
        "task",
        "train_r2",
        "test_r2",
        "cat_ratio",
        "importance_max",
        "importance_top3",
        "target_parent_count",
        "latent_noise",
    ]

    print(df[columns].head(20).to_string(index=False))

    print("\nSaved to target_latent_diagnostics.csv")


if __name__ == "__main__":
    main()
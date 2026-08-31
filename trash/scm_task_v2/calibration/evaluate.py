import numpy as np
import torch

from ..task import SCMTask
from .structural import compute_structural_metrics
from .difficulty import compute_difficulty_metrics
from .matching import compute_observable_profile, profile_distance_to_reference


def _evaluate_task(task, X_train, y_train, X_test, y_test, info, probe_epochs=100, probe_seed=0, real_reference=None, real_scales=None):
    structural = compute_structural_metrics(task, info)

    difficulty = compute_difficulty_metrics(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_classes=task.num_classes,
        epochs=probe_epochs,
        probe_seed=probe_seed,
        feature_type=info["feature_type"],
        cardinality=info["cardinality"],
    )

    if task.num_classes is None and difficulty["mlp_score"] < -1.0:
        target_connection = task.scm.connections[-1]
        target_function = target_connection.child_functions[0]
        target_program = None if target_function is None else target_function.program

        print(
            f"[BAD TARGET] "
            f"mlp_r2={difficulty['mlp_score']:.4f}, "
            f"linear_r2={difficulty['linear_score']:.4f}, "
            f"program={target_program}"
        )


    X_all = torch.cat((X_train, X_test), dim=0)
    y_all = torch.cat((y_train, y_test), dim=0)

    observable = compute_observable_profile(
        X=X_all,
        y=y_all,
        feature_type=info["feature_type"],
        cardinality=info["cardinality"],
        num_classes=task.num_classes,
    )

    validity = {
        "is_valid": bool(info["is_valid"]),
        "categorical_features_ok": bool(info["categorical_features_ok"]),
        "target_ok": bool(info["target_ok"]),
        "importance_ok": bool(info["importance_ok"]),
    }

    result = {
        "structural": structural,
        "difficulty": difficulty,
        "observable": observable,
        "validity": validity,
    }

    if real_reference is not None:
        result["real_data_distance"] = profile_distance_to_reference(observable, real_reference, scales=real_scales)

    return result


def _summarize_values(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("Cannot summarize an empty collection.")

    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _aggregate_section(task_results, section):
    metric_names = task_results[0][section].keys()
    return {metric_name: _summarize_values([result[section][metric_name] for result in task_results]) for metric_name in metric_names}


def _compute_rates(task_results):
    valid = np.asarray([result["validity"]["is_valid"] for result in task_results], dtype=float)
    categorical_valid = np.asarray([result["validity"]["categorical_features_ok"] for result in task_results], dtype=float)
    target_valid = np.asarray([result["validity"]["target_ok"] for result in task_results], dtype=float)
    importance_valid = np.asarray([result["validity"]["importance_ok"] for result in task_results], dtype=float)
    categorical_fraction = np.asarray([result["observable"].get("categorical_fraction", 0.0) for result in task_results], dtype=float)
    dummy_score = np.asarray([result["difficulty"]["dummy_score"] for result in task_results], dtype=float)
    mlp_score = np.asarray([result["difficulty"]["mlp_score"] for result in task_results], dtype=float)
    nonlinear_gain = np.asarray([result["difficulty"]["nonlinear_gain"] for result in task_results], dtype=float)

    return {
        "valid_rate": float(valid.mean()),
        "categorical_valid_rate": float(categorical_valid.mean()),
        "target_valid_rate": float(target_valid.mean()),
        "importance_valid_rate": float(importance_valid.mean()),
        "all_continuous_rate": float((categorical_fraction == 0.0).mean()),
        "trivial_rate": float((mlp_score >= 0.95).mean()),
        "weak_signal_rate": float(((mlp_score - dummy_score) < 0.05).mean()),
        "nonlinear_task_rate": float((nonlinear_gain >= 0.05).mean()),
    }


def _extract_sampled_parameters(task_results):
    parameter_names = (
        "sampled_connection_probs",
        "sampled_latent_noise_scale",
        "sampled_arity_probs",
        "sampled_unary_op_probs",
        "sampled_binary_op_probs",
        "sampled_ternary_op_probs",
        "sampled_observation_type_probs",
    )

    sampled = {}

    for parameter_name in parameter_names:
        values = [result["sampled_parameters"].get(parameter_name) for result in task_results]
        values = [value for value in values if value is not None]

        if not values:
            continue

        array = np.asarray(values, dtype=float)

        if array.ndim == 1:
            sampled[parameter_name] = _summarize_values(array)
        else:
            sampled[parameter_name] = [_summarize_values(array[:, index]) for index in range(array.shape[1])]

    return sampled


def _to_python_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return tuple(_to_python_value(item) for item in value)
    if isinstance(value, list):
        return [_to_python_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_python_value(item) for key, item in value.items()}
    return value


def evaluate_prior(prior, n_tasks=300, num_classes=3, probe_epochs=100, real_reference=None, real_scales=None, seed_offset=0, keep_task_results=False):
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive.")

    task_results = []

    for task_index in range(n_tasks):
        seed = int(seed_offset + task_index)

        task = SCMTask(
            num_classes=num_classes,
            **prior,
            dag_seed=seed,
            x_seed=100000 + seed,
            aleatoric_seed=200000 + seed,
        )

        X_train, y_train, X_test, y_test, info = task._generate()

        result = _evaluate_task(
            task=task,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            info=info,
            probe_epochs=probe_epochs,
            probe_seed=300000 + seed,
            real_reference=real_reference,
            real_scales=real_scales,
        )

        result["sampled_parameters"] = {
            "sampled_connection_probs": _to_python_value(info.get("sampled_connection_probs")),
            "sampled_latent_noise_scale": _to_python_value(info.get("sampled_latent_noise_scale")),
            "sampled_arity_probs": _to_python_value(info.get("sampled_arity_probs")),
            "sampled_unary_op_probs": _to_python_value(info.get("sampled_unary_op_probs")),
            "sampled_binary_op_probs": _to_python_value(info.get("sampled_binary_op_probs")),
            "sampled_ternary_op_probs": _to_python_value(info.get("sampled_ternary_op_probs")),
            "sampled_observation_type_probs": _to_python_value(info.get("sampled_observation_type_probs")),
        }

        task_results.append(result)

    output = {
        "prior": _to_python_value(prior),
        "n_tasks": int(n_tasks),
        "structural": _aggregate_section(task_results, "structural"),
        "difficulty": _aggregate_section(task_results, "difficulty"),
        "observable": _aggregate_section(task_results, "observable"),
        "rates": _compute_rates(task_results),
        "sampled_parameters": _extract_sampled_parameters(task_results),
    }

    if real_reference is not None:
        distances = [result["real_data_distance"] for result in task_results]
        output["real_data_distance"] = _summarize_values(distances)

    if keep_task_results:
        output["task_results"] = task_results

    return output
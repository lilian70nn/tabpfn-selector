import torch

from src.data.scm_task_v2.task import SCMTask
from src.data.scm_task_v2.calibration.difficulty import compute_difficulty_metrics


def summarize(values):
    x = torch.tensor(values, dtype=torch.float32)

    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "p25": float(torch.quantile(x, 0.25)),
        "p75": float(torch.quantile(x, 0.75)),
        "p90": float(torch.quantile(x, 0.90)),
    }


def run(prior, n_tasks=100, num_classes=2):
    latent_to_latent = []
    latent_to_observed = []
    observed_to_observed = []

    for i in range(n_tasks):
        task = SCMTask(
            num_classes=num_classes,
            dag_seed=i,
            aleatoric_seed=100_000 + i,
            x_seed=200_000 + i,
            **prior,
        )

        info = task.info

        # --------------------------------------------------
        # A. selected latent X -> continuous latent target
        # --------------------------------------------------

        a = compute_difficulty_metrics(
            X_train=info["selected_latent_X_train"],
            y_train=info["target_latent_train"],
            X_test=info["selected_latent_X_test"],
            y_test=info["target_latent_test"],
            num_classes=None,
            epochs=1000,
            mlp_hidden_dim=64,
            probe_seed=i,
            feature_type=None,
            cardinality=None,
        )

        # --------------------------------------------------
        # B. same selected latent X -> actual observed target
        # --------------------------------------------------

        b = compute_difficulty_metrics(
            X_train=info["selected_latent_X_train"],
            y_train=task.y_train,
            X_test=info["selected_latent_X_test"],
            y_test=task.y_test,
            num_classes=num_classes,
            epochs=1000,
            mlp_hidden_dim=64,
            probe_seed=i,
            feature_type=None,
            cardinality=None,
        )

        # --------------------------------------------------
        # C. actual observed X -> actual observed target
        # --------------------------------------------------

        c = compute_difficulty_metrics(
            X_train=task.X_train,
            y_train=task.y_train,
            X_test=task.X_test,
            y_test=task.y_test,
            num_classes=num_classes,
            epochs=1000,
            mlp_hidden_dim=64,
            probe_seed=i,
            feature_type=info["feature_type"],
            cardinality=info["cardinality"],
        )

        latent_to_latent.append(a["nonlinear_gain"])
        latent_to_observed.append(b["nonlinear_gain"])
        observed_to_observed.append(c["nonlinear_gain"])

    print("\nA: selected latent X -> latent y")
    print(summarize(latent_to_latent))

    print("\nB: selected latent X -> observed y")
    print(summarize(latent_to_observed))

    print("\nC: observed X -> observed y")
    print(summarize(observed_to_observed))


if __name__ == "__main__":
    prior = {
        "n_min": 400,
        "n_max": 512,
        "d_min": 8,
        "d_max": 16,
        "test_frac": 0.15,
        "p_missing": 0.05,
        "sampling_penalty": 1,

        "num_roots": 8,
        "num_layers": 5,
        "hidden_width_min": 6,
        "hidden_width_max": 10,
        "final_width": 1,

        "connection_probs": (
            (0.20, 0.35),
            (0.20, 0.35),
            (0.25, 0.40),
            (0.50, 0.80),
        ),

        "latent_noise_scale": (0.0, 0.03),
        "source_prior_probs": (0.45, 0.20, 0.15, 0.05),

        "arity_probs": (3.0, 5.0, 2.0),
        "unary_op_probs": (0.5, 1.5, 2.0, 2.0, 1.5, 1.0, 1.5),
        "binary_op_probs": (2.5, 2.0, 3.5, 2.0),
        "ternary_op_probs": (2.0, 3.0, 2.0, 3.0),

        "scale_min": 0.25,
        "scale_max": 4.0,

        "observation_type_probs": (6.0, 2.0, 2.0),
        "categorical_cardinalities": (2, 3, 4, 5, 6),
        "categorical_cardinality_probs": (0.40, 0.30, 0.18, 0.08, 0.04),

        "min_samples_per_category": 8,
        "min_component_weight": 0.05,
        "observation_noise_scale": 0.03,
    }

    run(
        prior=prior,
        n_tasks=100,
        num_classes=2,
    )
from pprint import pprint
from .evaluate import evaluate_prior


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
        (0.30, 0.40),
        (0.55, 0.75),
    ),

    "latent_noise_scale": (0.0, 0.03),
    "source_prior_probs": (0.45, 0.20, 0.15, 0.05),

    "arity_probs": (2.5, 3.0, 3.0),

    "unary_op_probs": (0.5, 1.5, 2.0, 2.0, 1.5, 1.0, 1.5),
    "binary_op_probs": (2.0, 2.0, 2.0, 2.0),
    "ternary_op_probs": (3.0, 1.0, 1.0, 3.0),

    "scale_min": 0.25,
    "scale_max": 4.0,

    "observation_type_probs": (6.0, 2.0, 2.0),
    "categorical_cardinalities": (2, 3, 4, 5, 6),
    "categorical_cardinality_probs": (0.40, 0.30, 0.18, 0.08, 0.04),
    "min_samples_per_category": 8,
    "min_component_weight": 0.05,
    "observation_noise_scale": 0.03,
}


result = evaluate_prior(
    prior=prior,
    n_tasks=300,
    num_classes=4,
    probe_epochs=500,
    seed_offset=0,
)

print("\n=== Structural ===")
pprint(result["structural"])

print("\n=== Difficulty ===")
pprint(result["difficulty"])

print("\n=== Observable ===")
pprint(result["observable"])

print("\n=== Rates ===")
pprint(result["rates"])

print("\n=== Sampled Parameters ===")
pprint(result["sampled_parameters"])
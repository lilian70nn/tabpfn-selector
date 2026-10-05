from pprint import pprint
from .evaluate import evaluate_prior


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
    "edge_weight_concentration": (0.30, 2.00),
    "latent_noise_scale": (0.0, 0.03),

    "child_method_probs": (3, 2, 3, 2),
    "source_prior_probs": (0.45, 0.20, 0.15, 0.05),
    "joint_mlp_hidden_dim": 8,

    "edge_family_probs": (2.0, 4.0, 3.0),
    "small_mlp_hidden_dim": 8,
    "soft_tree_depth": 2,
    "soft_tree_temperature": 0.5,

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
    num_classes=3,
    probe_epochs=500,
    seed_offset=0,
)

pprint(result["structural"])
pprint(result["difficulty"])
pprint(result["observable"])
pprint(result["rates"])
pprint(result["sampled_parameters"])
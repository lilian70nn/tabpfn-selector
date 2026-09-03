import torch

SCM_PRIOR = {
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
        (0.25, 0.40),
        (0.55, 0.75),
    ),

    "source_prior_probs": (0.55, 0.20, 0.15, 0.10),
    "arity_probs": (2.5, 3.0, 3.0,),
    "unary_op_probs": (1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.75),
    "binary_op_probs":(2.0, 2.0, 2.0, 1.5, 1.5),
    "ternary_op_probs": (3.0, 1.0, 1.0, 3.0, 1.5),
    "observation_type_probs": (6.5, 1.75, 1.75),
    "latent_noise_scale": (0.0, 0.0,),
    "scale_min": 0.25,
    "scale_max": 4.0,
    "categorical_cardinalities": (2, 3, 4, 5, 6),
    "categorical_cardinality_probs": (0.40, 0.30, 0.18, 0.08, 0.04,),
    "min_samples_per_category": 8,
    "min_component_weight": 0.05,
    "observation_noise_scale": 0.03,
    "device":torch.device("cpu")
}

LINEAR_PRIOR = {
    "n_min": 400,
    "n_max": 512,
    "d_min": 8,
    "d_max": 16,
    "test_frac": 0.15,
    "p_categorical": 0.3,
    "max_cardinality": 10,
    "p_active": 0.65,
    "p_missing": 0.05,
    "noise_level": 0.1,
    "device":torch.device("cpu")
}
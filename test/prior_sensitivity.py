import pandas as pd
import copy
from src.data.config import SCM_PRIOR
from src.data.scm_task_v2.analysis import evaluate_prior

SEARCH_PARAMS = {
    "num_roots": [3, 5, 8],

    "connection_probs": [
        ((0.15, 0.30), (0.40, 0.60)),
        ((0.25, 0.40), (0.55, 0.75)),
        ((0.35, 0.50), (0.70, 0.90)),
    ],

    "arity_probs": [
        (3.5, 3.0, 2.0),
        (2.5, 3.0, 3.0),
        (1.5, 3.0, 4.0),
    ],

    "observation_type_probs": [
        (8.0, 1.0, 1.0),
        (6.5, 1.75, 1.75),
        (5.0, 2.5, 2.5),
    ],

    "observation_noise_scale": [0.0, 0.03, 0.06],
}


if __name__ == "__main__":
    
    seed = 10
    all_results = []

    for param_name, values in SEARCH_PARAMS.items():
        for value_idx, value in enumerate(values):
            prior = copy.deepcopy(SCM_PRIOR)
            prior[param_name] = value

            setting_name = f"{param_name}_{value_idx}"
            print(f"\n===== {setting_name} =====")
            print(f"{param_name} = {value}")

            regression_result = evaluate_prior(
                prior=prior,
                n_tasks=200,
                task_kind="regression",
                mlp_epochs=500,
                topk=3,
                base_seed=seed,
                prior_name=setting_name,
            )

            regression_result["task_kind"] = "regression"
            regression_result["search_param"] = param_name
            regression_result["search_value"] = str(value)
            all_results.append(regression_result)

            classification_result = evaluate_prior(
                prior=prior,
                n_tasks=200,
                task_kind="classification",
                mlp_epochs=500,
                topk=3,
                base_seed=seed,
                prior_name=setting_name,
            )

            classification_result["task_kind"] = "classification"
            classification_result["search_param"] = param_name
            classification_result["search_value"] = str(value)
            all_results.append(classification_result)


    results = pd.concat(all_results, ignore_index=True)
    results.to_csv("prior_sensitivity.csv", index=False)
    print("\nSaved to prior_sensitivity.csv")





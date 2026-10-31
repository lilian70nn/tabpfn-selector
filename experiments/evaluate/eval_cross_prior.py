import copy
import csv
from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader
from functools import partial
from src.training.eval import evaluate_synthetic_extended
from src.model.tabpfn import TabularPFNModel
from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2.task import SCMTask
from src.data.collate import collate_tasks

from experiments.config import (
    SCM_PRIOR,
    SCM_OOD_DEPTH,
    SCM_OOD_SPARSITY,
    SCM_OOD_MECHANISM,
    SCM_OOD_NOISE,
    SCM_OOD_MISSING,
    SCM_OOD_FEATURES
)


def run_extended_eval(
    model,
    val_loader,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metrics = evaluate_synthetic_extended(
        model=model,
        loader=val_loader,
        device=device,
        topk_fracs=(0.25, 0.50, 0.75),
        random_repeats=20,
    )

    return metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument(
        "--prior",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "depth",
            "sparsity",
            "mechanism",
            "noise",
            "missing",
            "features", 
        ],
    )
    args = parser.parse_args()
    prior_configs = {
        "baseline": SCM_PRIOR,
        "depth": SCM_OOD_DEPTH,
        "sparsity": SCM_OOD_SPARSITY,
        "mechanism": SCM_OOD_MECHANISM,
        "noise": SCM_OOD_NOISE,
        "missing": SCM_OOD_MISSING,
        "features": SCM_OOD_FEATURES
    }

    selected_prior = prior_configs[args.prior]

    device = torch.device("cuda")

    task_kinds = ("classification", "regression")
    methods = ("gradient", "eigen_90", "mi", "permutation")
    batch_to_start_frac = {1: 0.0, 2: 0.15, 3: 0.3}
    start_fracs = (batch_to_start_frac[args.batch],)

    output_path = Path(f"results/evaluation/extended_eval_{args.prior}_batch{args.batch}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for task_kind in task_kinds:
        for method in methods:

            prior = copy.deepcopy(selected_prior)
            prior["generate_importance"] = True
            prior["importance_method"] = method

            dataset_kwargs = dict(
                num_tasks=10000,
                task_factory=SCMTask,
                task_kind=task_kind,
                task_kwargs=prior,
            )

            if task_kind == "classification":
                dataset_kwargs.update(
                    min_classes=2,
                    max_classes=4,
                    base_seed=105500,
                )
            else:
                dataset_kwargs["base_seed"] = 102500

            val_dataset = SyntheticTaskDataset(**dataset_kwargs)

            val_loader = DataLoader(
                val_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                collate_fn=partial(collate_tasks, use_selector=True),
            )

            for start_frac in start_fracs:

                run_name = f"scm_{task_kind}_{method}_w_imp_start{start_frac:g}_epoch4"
                ckpt_path = Path("results/training") / run_name / "best_ckpt.pt"

                if not ckpt_path.exists():
                    raise FileNotFoundError(ckpt_path)

                model_kwargs = dict(
                    k=64,
                    m=120,
                    n_heads=4,
                    depth=16,
                    max_cardinality=10,
                    task_kind=task_kind,
                )

                if task_kind == "classification":
                    model_kwargs["max_classes"] = 4
                else:
                    model_kwargs["num_y_buckets"] = 100

                model = TabularPFNModel(**model_kwargs)

                print(f"\nEvaluating {run_name}", flush=True)

                metrics = run_extended_eval(
                    model=model,
                    val_loader=val_loader,
                    checkpoint_path=ckpt_path,
                    device=device,
                )

                results.append({
                    "prior": args.prior,
                    "task_kind": task_kind,
                    "importance_method": method,
                    "importance_start_frac": start_frac,
                    "epochs": 4,
                    **metrics,
                })

                with open(output_path, "w", newline="") as f:
                    fieldnames = list(dict.fromkeys(key for row in results for key in row))
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)

                print(f"Saved results for {run_name}", flush=True)

                del model

    print(f"\nFinished evaluating {len(results)} checkpoints.")

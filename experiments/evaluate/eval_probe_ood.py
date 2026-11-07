
import argparse
import copy
import csv
import re
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

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
    SCM_OOD_FEATURES,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    return parser.parse_args()


def parse_probe_name(probe_dir):
    name = probe_dir.name

    pattern = (
        r"_(classification|regression)"
        r"_(gradient|eigen_top1|eigen_90|mi|marginal|loco|permutation)"
        r"_(linear|mlp)_epoch\d+$"
    )

    match = re.search(pattern, name)

    if match is None:
        raise ValueError(f"Cannot parse probe directory name: {name}")

    task_kind, method, probe_head = match.groups()

    return task_kind, method, probe_head


def load_probe_model(probe_dir, task_kind, probe_head, device):
    model_kwargs = dict(
        k=64,
        m=120,
        n_heads=4,
        depth=16,
        max_cardinality=10,
        task_kind=task_kind,
        backbone_version="v2",
    )

    if task_kind == "classification":
        model_kwargs["max_classes"] = 4
    else:
        model_kwargs["num_y_buckets"] = 100

    model = TabularPFNModel(**model_kwargs)

    if probe_head == "linear":
        model.importance_head = nn.Linear(model_kwargs["k"], 1)

    ckpt_path = probe_dir / "best_ckpt.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True,)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def make_loader(prior_config, task_kind, method):
    prior = copy.deepcopy(prior_config)
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

    dataset = SyntheticTaskDataset(**dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=partial(collate_tasks, use_selector=True),
    )


def main():
    args = parse_args()
    probe_dir = args.probe_dir

    task_kind, method, probe_head = parse_probe_name(probe_dir)
    device = torch.device("cuda")

    model = load_probe_model(
        probe_dir=probe_dir,
        task_kind=task_kind,
        probe_head=probe_head,
        device=device,
    )

    prior_configs = {
        "baseline": SCM_PRIOR,
        "depth": SCM_OOD_DEPTH,
        "sparsity": SCM_OOD_SPARSITY,
        "mechanism": SCM_OOD_MECHANISM,
        "noise": SCM_OOD_NOISE,
        "missing": SCM_OOD_MISSING,
        "features": SCM_OOD_FEATURES,
    }

    output_path = probe_dir / "ood_results.csv"
    results = []

    for prior_name, prior_config in prior_configs.items():

        print(f"\nEvaluating {probe_dir.name} | OOD: {prior_name}", flush=True)

        val_loader = make_loader(
            prior_config=prior_config,
            task_kind=task_kind,
            method=method,
        )

        metrics = evaluate_synthetic_extended(
            model=model,
            loader=val_loader,
            device=device,
            topk_fracs=(0.25, 0.50, 0.75),
            random_repeats=20,
        )

        results.append({
            "prior": prior_name,
            **metrics,
        })

        with open(output_path, "w", newline="") as f:
            fieldnames = list(dict.fromkeys(
                key for row in results for key in row
            ))
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved {prior_name} results to {output_path}", flush=True)

    print(f"\nFinished evaluating {probe_dir.name}", flush=True)


if __name__ == "__main__":
    main()
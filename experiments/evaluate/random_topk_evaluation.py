import math
import csv
import torch
import copy
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from dataclasses import replace

from experiments.config import SCM_PRIOR
from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2.task import SCMTask
from src.data.collate import collate_tasks
from src.model.tabpfn import TabularPFNModel
from src.training.helper import move_batch_to_device


def make_random_topk_cell_mask(batch, topk_frac=0.25):
    B = batch.cell_mask.shape[0]
    device = batch.cell_mask.device
    keep_feature = torch.zeros(B, batch.d_max, dtype=torch.bool, device=device)

    for b in range(B):
        d = int(batch.d_emb[b].item())
        k = max(1, math.ceil(topk_frac * d))
        selected_idx = torch.randperm(d, device=device)[:k]
        keep_feature[b, selected_idx] = True

    random_topk_cell_mask = batch.cell_mask.clone()
    random_topk_cell_mask[:, :, :batch.d_max] &= keep_feature[:, None, :]
    return random_topk_cell_mask


@torch.no_grad()
def evaluate_random_topk(model, loader, device, max_batches=50, topk_frac=0.25, n_repeats=20):
    model.eval()
    repeat_losses = []

    for _ in range(n_repeats):
        pred_loss_sum = 0.0
        n_batches = 0

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            batch = move_batch_to_device(batch, device)
            random_mask = make_random_topk_cell_mask(batch, topk_frac=topk_frac)
            random_batch = replace(batch, cell_mask=random_mask)

            out = model(random_batch)
            pred_loss = model.prediction_loss(random_batch, out)

            pred_loss_sum += float(pred_loss.detach())
            n_batches += 1

        repeat_losses.append(pred_loss_sum / max(n_batches, 1))

    losses = torch.tensor(repeat_losses)

    return {
        "random_topk_pred_loss": losses.mean().item(),
        "random_topk_pred_loss_std": losses.std(unbiased=True).item(),
    }


def build_val_loader(task_kind, importance_method, batch_size=16):
    prior = copy.deepcopy(SCM_PRIOR)
    prior["generate_importance"] = True
    prior["importance_method"] = importance_method

    if task_kind == "classification":
        val_dataset = SyntheticTaskDataset(
            num_tasks=10000,
            task_factory=SCMTask,
            task_kind="classification",
            min_classes=2,
            max_classes=4,
            base_seed=105500,
            task_kwargs=prior,
        )
    else:
        val_dataset = SyntheticTaskDataset(
            num_tasks=10000,
            task_factory=SCMTask,
            task_kind="regression",
            base_seed=102500,
            task_kwargs=prior,
        )

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(collate_tasks, use_selector=True),
    )


def build_model(task_kind):
    model_kwargs = dict(k=64, m=120, n_heads=4, depth=16, max_cardinality=10, task_kind=task_kind)

    if task_kind == "classification":
        model_kwargs["max_classes"] = 4
    else:
        model_kwargs["num_y_buckets"] = 100

    return TabularPFNModel(**model_kwargs)


if __name__ == "__main__":
    device = torch.device("cuda")

    methods = ["gradient", "eigen_top1", "eigen_90", "mi", "marginal", "permutation"]
    starts = [0.0, 0.3]
    task_kinds = ["classification", "regression"]

    project_root = Path(__file__).resolve().parents[2]
    results_root = project_root / "results" / "training"
    output_dir = project_root / "results" / "evaluation"
    output_path = output_dir / "random_topk_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_runs = len(task_kinds) * len(methods) * len(starts)
    run_idx = 0

    for task_kind in task_kinds:
        for method in methods:
            for start in starts:
                run_idx += 1
                start_name = f"start{start:g}"

                print(f"[{run_idx:02d}/{total_runs}] {task_kind} | {method} | {start_name}")

                val_loader = build_val_loader(task_kind=task_kind, importance_method=method, batch_size=16)
                model = build_model(task_kind)

                run_name = f"scm_{task_kind}_{method}_w_imp_start{start:g}"
                checkpoint_path = results_root / run_name / "best_ckpt.pt"

                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)

                result = evaluate_random_topk(
                    model=model,
                    loader=val_loader,
                    device=device,
                    max_batches=50,
                    topk_frac=0.25,
                    n_repeats=20,
                )

                rows.append({
                    "task": task_kind,
                    "method": method,
                    "start": start_name,
                    "random_topk_pred_loss": result["random_topk_pred_loss"],
                    "random_topk_pred_loss_std": result["random_topk_pred_loss_std"],
                })

                print(f"    loss = {result['random_topk_pred_loss']:.4f} +/- {result['random_topk_pred_loss_std']:.4f}")

                del model
                del val_loader
                torch.cuda.empty_cache()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "method",
                "start",
                "random_topk_pred_loss",
                "random_topk_pred_loss_std",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output_path}")
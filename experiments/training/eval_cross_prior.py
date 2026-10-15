import torch
from torch.utils.data import DataLoader
from functools import partial
from src.training.eval import evaluate_synthetic_extended
from src.model.tabpfn import TabularPFNModel
from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2.task import SCMTask
from experiments.config import SCM_PRIOR
from src.data.collate import collate_tasks


def run_extended_eval(
    model,
    val_loader,
    checkpoint_path,
    device,
    max_batches=50,
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
        max_batches=max_batches,
        topk_fracs=(0.25, 0.50, 0.75),
        random_repeats=20,
    )

    return metrics


if __name__ == "__main__":

    ckpt_path = "results/training/scm_classification_gradient_w_imp_start0/best_ckpt.pt"

    model = TabularPFNModel(
        k=64,
        m=120,
        n_heads=4,
        depth=16,
        max_cardinality=10,
        task_kind="classification",
        max_classes=4,
    )

    val_dataset = SyntheticTaskDataset(
        num_tasks=10000,
        task_factory=SCMTask,
        task_kind="classification",
        min_classes=2,
        max_classes=4,
        base_seed=105500,
        task_kwargs=SCM_PRIOR,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(collate_tasks, use_selector=True),
    )


    results = run_extended_eval(
        model=model,
        val_loader=val_loader,
        checkpoint_path=ckpt_path,
        device=torch.device("cuda"),
        max_batches=50,
    )

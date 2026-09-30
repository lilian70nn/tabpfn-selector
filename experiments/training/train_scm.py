import argparse
import copy
import torch
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader

from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2.task import SCMTask
from src.data.collate import collate_tasks
from src.model.tabpfn import TabularPFNModel
from src.training.train import train_synthetic
from experiments.config import SCM_PRIOR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--importance-method",
        type=str,
        default="eigen_90",
        choices=[
            "gradient",
            "eigen_top1",
            "eigen_90",
            "mi",
            "marginal",
            "loco",
            "permutation"
        ],
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Training steps as a multiple of len(train_loader).",
    )

    parser.add_argument(
        "--task-kind",
        type=str,
        default="classification",
        choices=["classification", "regression"],
    )

    parser.add_argument(
        "--use-importance",
        action="store_true",
    )

    parser.add_argument(
        "--importance-weight",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--importance-start-frac",
        type=float,
        default=0.0,
        help="Fraction of training steps before importance supervision starts.",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Don't modify the global SCM_PRIOR object.
    prior = copy.deepcopy(SCM_PRIOR)
    prior["generate_importance"] = args.use_importance
    prior["importance_method"] = args.importance_method

    if args.task_kind == "classification":
        train_dataset = SyntheticTaskDataset(
            num_tasks=105500,
            task_factory=SCMTask,
            task_kind="classification",
            min_classes=2,
            max_classes=4,
            base_seed=0,
            task_kwargs=prior,
        )

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
        train_dataset = SyntheticTaskDataset(
            num_tasks=102500,
            task_factory=SCMTask,
            task_kind="regression",
            base_seed=0,
            task_kwargs=prior,
        )

        val_dataset = SyntheticTaskDataset(
            num_tasks=10000,
            task_factory=SCMTask,
            task_kind="regression",
            base_seed=102500,
            task_kwargs=prior,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=partial(collate_tasks, use_selector=args.use_importance),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(collate_tasks, use_selector=args.use_importance),
    )
    

    model_kwargs = dict(
        k=64,
        m=120,
        n_heads=4,
        depth=16,
        max_cardinality=10,
        task_kind=args.task_kind,
    )

    if args.task_kind == "classification":
        model_kwargs["max_classes"] = 4
    else:
        model_kwargs["num_y_buckets"] = 100

    model = TabularPFNModel(**model_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-2,
    )

    if args.save_path is None:
        if args.use_importance:
            run_name = f"scm_{args.task_kind}_{args.importance_method}_w_imp"
        else:
            run_name = f"scm_{args.task_kind}_wo_imp"

        save_path = Path(__file__).resolve().parents[2] / "results" / "training" / run_name
    else:
        save_path = Path(args.save_path)

    steps = int(args.epochs * len(train_loader))

    train_synthetic(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        steps=steps,
        importance_weight=args.importance_weight if args.use_importance else None,
        importance_start_frac=args.importance_start_frac,
        grad_clip=1.0,
        log_every=50,
        val_loader=val_loader,
        val_every=300,
        val_batches=50,
        imp_trace=args.use_importance,
        trace_num_tables=10,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
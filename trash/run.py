import argparse
from functools import partial
import torch
from torch.utils.data import DataLoader

from src.data.linear_task import MixedLinearTask
from src.data.datasets import SyntheticTaskDataset
from src.data.collate import collate_tasks
from src.model.tabpfn_v2 import TabularPFNModel
from src.training.train import train_synthetic


CANONICAL = dict(
    k=72,
    m=256,
    n_head=6,
    depth=16,
    task_kind="classification",
    batch_size=16,
    selector=True,
    train_length=100000,
    val_length=10000,
    steps=10000,
    lr=2e-4,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k", type=int, default=CANONICAL["k"])
    parser.add_argument("--m", type=int, default=CANONICAL["m"])
    parser.add_argument("--n-head", type=int, default=CANONICAL["n_head"])
    parser.add_argument("--depth", type=int, default=CANONICAL["depth"])
    parser.add_argument(
        "--task-kind",
        choices=["classification", "regression"],
        default=CANONICAL["task_kind"],
    )

    parser.add_argument("--batch-size", type=int, default=CANONICAL["batch_size"])
    parser.add_argument("--selector", dest="selector", action="store_true")
    parser.add_argument("--no-selector", dest="selector", action="store_false")
    parser.set_defaults(selector=CANONICAL["selector"])

    parser.add_argument("--train-length", type=int, default=CANONICAL["train_length"])
    parser.add_argument("--val-length", type=int, default=CANONICAL["val_length"])
    parser.add_argument("--steps", type=int, default=CANONICAL["steps"])
    parser.add_argument("--lr", type=float, default=CANONICAL["lr"])

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument(
        "--save-path",
        type=str,
        default="logs/train_logs.txt",
    )
    parser.add_argument(
        "--best-ckpt-path",
        type=str,
        default="logs/train_best_ckpt.pt",
    )

    return parser.parse_args()


def get_device(args):
    if args.device is not None:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_config(args):
    print("=" * 80)
    print("Selector-PFN training configuration")
    print("=" * 80)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 80)


def main():
    args = parse_args()
    print_config(args)

    device = get_device(args)

    task_kwargs = dict(
        n_min=400,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_categorical=0.3,
        max_cardinality=5,
        p_active=0.85,
        p_missing=0.05,
        noise_level=0.1,
        device=torch.device("cpu"),
    )

    train_dataset = SyntheticTaskDataset(
        num_tasks=args.train_length,
        task_factory=MixedLinearTask,
        task_kind=args.task_kind,
        min_classes=2,
        max_classes=4,
        base_seed=0,
        task_kwargs=task_kwargs,
    )

    val_dataset = SyntheticTaskDataset(
        num_tasks=args.val_length,
        task_factory=MixedLinearTask,
        task_kind=args.task_kind,
        min_classes=2,
        max_classes=4,
        base_seed=args.train_length,
        task_kwargs=task_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_tasks, use_selector=args.selector),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_tasks, use_selector=args.selector),
    )

    model = TabularPFNModel(
        k=args.k,
        m=args.m,
        n_heads=args.n_head,
        depth=args.depth,
        max_cardinality=5,
        task_kind=args.task_kind,
        max_classes=4 if args.task_kind == "classification" else None,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
    )

    train_synthetic(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        steps=args.steps,
        importance_weight=100 if args.selector else 0,
        grad_clip=1.0,
        log_every=50,
        val_loader=val_loader,
        val_every=500,
        val_batches=50,
        save_path=args.save_path,
        best_ckpt_path=args.best_ckpt_path,
    )


if __name__ == "__main__":
    main()
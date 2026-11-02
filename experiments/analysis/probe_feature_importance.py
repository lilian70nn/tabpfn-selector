import argparse
import copy
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.config import SCM_PRIOR
from src.data.collate import collate_tasks
from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2.task import SCMTask
from src.model.tabpfn import TabularPFNModel
from src.training.train import train_synthetic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--probe-head', choices=['linear', 'mlp'], required=True)
    parser.add_argument('--task-kind', choices=['classification', 'regression'], required=True)
    parser.add_argument('--importance-method', default='gradient',
                        choices=['gradient', 'eigen_top1', 'eigen_90', 'mi', 'marginal', 'permutation'])
    parser.add_argument("--importance-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def load_checkpoint(model, path):
    # Compatible with a raw state_dict or common checkpoint wrappers.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise TypeError('Checkpoint must contain a model state_dict. Inspect its keys and update load_checkpoint().')
    model.load_state_dict(state, strict=True)


def to_device(batch, device):
    for name, value in vars(batch).items():
        if isinstance(value, torch.Tensor):
            setattr(batch, name, value.to(device, non_blocking=True))
    return batch


def make_dataset(task_kind, num_tasks, base_seed, importance_method):
    prior = copy.deepcopy(SCM_PRIOR)
    prior['generate_importance'] = True
    prior['importance_method'] = importance_method
    kwargs = dict(
        num_tasks=num_tasks,
        task_factory=SCMTask,
        task_kind=task_kind,
        base_seed=base_seed,
        task_kwargs=prior,
    )
    if task_kind == 'classification':
        kwargs.update(min_classes=2, max_classes=4)
    return SyntheticTaskDataset(**kwargs)


def make_loader(dataset, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=16,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=2 if shuffle else 0,
        pin_memory=True,
        persistent_workers=bool(shuffle),
        collate_fn=partial(collate_tasks, use_selector=True),
    )


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            loss = model.importance_loss(batch, out)
            batch_size = batch.X_train.shape[0]
            total_loss += loss.item() * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


def main():
    args = parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_kwargs = dict(
        k=64,
        m=120,
        n_heads=4,
        depth=16,
        max_cardinality=10,
        task_kind=args.task_kind,
        backbone_version='v2',
    )
    if args.task_kind == 'classification':
        model_kwargs['max_classes'] = 4
    else:
        model_kwargs['num_y_buckets'] = 100

    model = TabularPFNModel(**model_kwargs)
    load_checkpoint(model, args.checkpoint)

    for param in model.parameters():
        param.requires_grad_(False)

    k = model_kwargs['k']
    if args.probe_head == 'linear':
        model.importance_head = nn.Linear(k, 1)
    else:
        model.importance_head = nn.Sequential(
            nn.LayerNorm(k),
            nn.Linear(k, k),
            nn.GELU(),
            nn.Linear(k, 1),
        )
    model.to(device)
    model.eval()  # Frozen backbone; gradients still flow through the new head.

    optimizer = torch.optim.AdamW(model.importance_head.parameters(), lr=2e-4, weight_decay=1e-2)
    train_loader = make_loader(make_dataset(args.task_kind, 105500 if args.task_kind == 'classification' else 102500, 0, args.importance_method,), True, 0)
    val_loader = make_loader(make_dataset(args.task_kind, 10000, 105500 if args.task_kind == 'classification' else 102500, args.importance_method,), False, 0)

    save_dir = (
        Path("results/probing")
        / f"{args.checkpoint.parent.name}_{args.task_kind}_{args.importance_method}_{args.probe_head}_epoch{args.epochs}"
    )

    steps = len(train_loader) * args.epochs

    train_synthetic(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        steps=steps,
        importance_weight=args.importance_weight,
        importance_start_frac=0.0,
        grad_clip=1.0,
        log_every=100,
        val_loader=val_loader,
        val_every=500,
        best_metric="importance_loss",
        imp_trace=True,
        trace_num_tables=10,
        save_path=save_dir,
    )


if __name__ == '__main__':
    main()


from src.data.linear_task import MixedLinearTask
from src.data.datasets import SyntheticTaskDataset
import torch
from torch.utils.data import DataLoader
from functools import partial
from src.data.collate import collate_tasks
from src.model.tabpfn_v2 import TabularPFNModel
from src.training.train import train_synthetic

device = torch.device("cuda")

train_dataset = SyntheticTaskDataset(
    length=100000,
    task_factory=MixedLinearTask,
    task_kind="classification",
    min_classes=2,
    max_classes=4,
    base_seed=0,
    task_kwargs=dict(
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
    ),
)

val_dataset = SyntheticTaskDataset(
    length=10000,
    task_factory=MixedLinearTask,
    task_kind="classification",
    min_classes=2,
    max_classes=4,
    base_seed=100000,
    task_kwargs=dict(
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
    ),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=partial(collate_tasks, use_selector=True),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=partial(collate_tasks, use_selector=True),
)


model = TabularPFNModel(
    k=72,
    m=256,
    n_heads=6,
    depth=16,
    max_cardinality=5,
    task_kind="classification",
    max_classes=4,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)


train_synthetic(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    steps=10000,
    importance_weight=100,
    grad_clip=1.0,
    log_every=50,
    val_loader=val_loader,
    val_every=500,
    val_batches=50,
    save_path="logs/large_synth_cls_2to4_512maxn_batch16_missing5_pred_imp_log.txt",
    best_ckpt_path="logs/large_synth_cls_2to4_512maxn_batch16_missing5_pred_imp_best_ckpt.pt",
)

import torch

from src.data.datasets import SyntheticTaskDataset
from src.data.linear_task import LinearTask
from src.data.collate import collate_tasks
from src.model.tabpfn_v2 import TabularPFNModel
from src.training.train import train_synthetic
from torch.utils.data import DataLoader


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from functools import partial


PRIOR = {
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


train_dataset = SyntheticTaskDataset(
    num_tasks=5000,
    task_factory=LinearTask,
    task_kind="classification",
    min_classes=2,
    max_classes=4,
    base_seed=0,
    task_kwargs=PRIOR
)

val_dataset = SyntheticTaskDataset(
    num_tasks=500,
    task_factory=LinearTask,
    task_kind="classification",
    min_classes=2,
    max_classes=4,
    base_seed=100000,
    task_kwargs=PRIOR,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=24,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=partial(collate_tasks, use_selector=True),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=24,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=partial(collate_tasks, use_selector=True),
)


model = TabularPFNModel(
    k=64,
    m=120,
    n_heads=4,
    depth=12,
    max_cardinality=10,
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
    steps=1500,
    importance_weight=50,
    grad_clip=1.0,
    log_every=50,
    val_loader=val_loader,
    val_every=300,
    val_batches=50,
    imp_trace=True,
    trace_num_tables=10,
    save_path="/content",

)
 

 
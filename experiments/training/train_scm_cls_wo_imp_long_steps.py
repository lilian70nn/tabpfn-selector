import torch
from pathlib import Path

from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2.task import SCMTask
from src.data.collate import collate_tasks
from src.model.tabpfn import TabularPFNModel
from src.training.train import train_synthetic
from experiments.config import SCM_PRIOR
from torch.utils.data import DataLoader


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from functools import partial



train_dataset = SyntheticTaskDataset(
    num_tasks=100000,
    task_factory=SCMTask,
    task_kind="classification",
    min_classes=2,
    max_classes=4,
    base_seed=0,
    task_kwargs=SCM_PRIOR
)

val_dataset = SyntheticTaskDataset(
    num_tasks=10000,
    task_factory=SCMTask,
    task_kind="classification",
    min_classes=2,
    max_classes=4,
    base_seed=100000,
    task_kwargs=SCM_PRIOR,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=partial(collate_tasks, use_selector=False),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=partial(collate_tasks, use_selector=False),
)


model = TabularPFNModel(
    k=64,
    m=120,
    n_heads=4,
    depth=16,
    max_cardinality=10,
    task_kind="classification",
    max_classes=4,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)

save_path = Path(__file__).resolve().parents[2] / "results" / "training"

train_synthetic(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    steps=15000,
    importance_weight=None,
    grad_clip=1.0,
    log_every=50,
    val_loader=val_loader,
    val_every=300,
    val_batches=50,
    imp_trace=False,
    trace_num_tables=10,
    save_path=save_path / "scm_cls_wo_imp_training_long_results",

)
 

 
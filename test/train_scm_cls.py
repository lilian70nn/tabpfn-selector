import torch

from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task import MixedSCMTask
from src.data.collate import collate_tasks
from src.model.tabpfn_v2 import TabularPFNModel
from src.training.train import train_synthetic
from torch.utils.data import DataLoader


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from functools import partial



train_dataset = SyntheticTaskDataset(
    num_tasks=100000,
    task_factory=MixedSCMTask,
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
        p_cat=0.3,
        max_cardinality=10,
        p_missing=0.05,
        node_noise_scale=0.05,
        num_roots=5,
        num_layers=4,
        max_nodes_per_layer=12,
        edge_prob=0.3,
        min_parents_per_node=1,
        num_bins=5,
        device=torch.device("cpu"),
    ),
)

val_dataset = SyntheticTaskDataset(
    num_tasks=10000,
    task_factory=MixedSCMTask,
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
        p_cat=0.3,
        max_cardinality=10,
        p_missing=0.05,
        node_noise_scale=0.05,
        num_roots=5,
        num_layers=4,
        max_nodes_per_layer=12,
        edge_prob=0.3,
        min_parents_per_node=1,
        num_bins=5,
        device=torch.device("cpu"),
    ),
)


train_loader = DataLoader(
    train_dataset,
    batch_size=12,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=partial(collate_tasks, use_selector=True),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=12,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=partial(collate_tasks, use_selector=True),
)


model = TabularPFNModel(
    k=120,
    m=384,
    n_heads=8,
    depth=24,
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
    steps=30000,
    importance_weight=20,
    grad_clip=1.0,
    log_every=50,
    val_loader=val_loader,
    val_every=500,
    val_batches=50,
    save_path="/dss/dsshome1/07/ra58bim2/pfn_exp/outputs/scm_cls.txt",
    best_ckpt_path = "/dss/dsshome1/07/ra58bim2/pfn_exp/outputs/scm_best_ckpt.pt"
)
 

 
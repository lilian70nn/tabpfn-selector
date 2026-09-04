import torch
from pathlib import Path

from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2.task import SCMTask
from src.data.collate import collate_tasks
from src.model.tabpfn import TabularPFNModel
from src.training.train import train_synthetic
from torch.utils.data import DataLoader


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from functools import partial



SCM_PRIOR = {
    "n_min": 400,
    "n_max": 512,
    "d_min": 8,
    "d_max": 16,
    "test_frac": 0.15,
    "p_missing": 0.05,
    "num_roots": 5,
    "num_layers": 3,
    "final_width": 1,

    "connection_probs": (
        (0.25, 0.40),
        (0.55, 0.75),
    ),

    "source_prior_probs": (0.55, 0.20, 0.15, 0.10),
    "arity_probs": (2.5, 3.0, 3.0,),
    "unary_op_probs": (1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.75),
    "binary_op_probs":(2.0, 2.0, 2.0, 1.5, 1.5),
    "ternary_op_probs": (3.0, 1.0, 1.0, 3.0, 1.5),
    "observation_type_probs": (6.5, 1.75, 1.75),
    "latent_noise_scale": (0.0, 0.0,),
    "scale_min": 0.25,
    "scale_max": 4.0,
    "categorical_cardinalities": (2, 3, 4, 5, 6),
    "categorical_cardinality_probs": (0.40, 0.30, 0.18, 0.08, 0.04,),
    "min_samples_per_category": 8,
    "min_component_weight": 0.05,
    "observation_noise_scale": 0.03,
    "device":torch.device("cpu")
}

LINEAR_PRIOR = {
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

save_path = Path(__file__).resolve().parent

train_synthetic(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    steps=15000,
    importance_weight=50,
    grad_clip=1.0,
    log_every=50,
    val_loader=val_loader,
    val_every=300,
    val_batches=50,
    imp_trace=True,
    trace_num_tables=10,
    save_path=save_path / "cross_cls_1_w_imp_training_results",

)


 

 
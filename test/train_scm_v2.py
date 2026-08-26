import torch

from src.data.datasets import SyntheticTaskDataset
from src.data.scm_task_v2 import WeightedMixedScalarSCMTask
from src.data.collate import collate_tasks
from src.model.tabpfn_v2 import TabularPFNModel
from src.training.train import train_synthetic
from torch.utils.data import DataLoader


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from functools import partial



TASK_KWARGS = dict(
    n_min=400,
    n_max=512,
    d_min=8,
    d_max=16,
    test_frac=0.15,
    p_missing=0.05,
    num_roots=8,
    num_layers=5,
    hidden_width_min=6,
    hidden_width_max=10,
    final_width=1,
    connection_probs=(0.20, 0.20, 0.30, 0.85),
    edge_weight_concentration=0.30,
    latent_noise_scale=0.0,
    sampling_penalty=0.25,
    observation_noise_scale=0.03,
    observation_type_probs=(0.60, 0.20, 0.20),
    categorical_cardinalities=(2, 3, 4, 5, 6),
    categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    min_samples_per_category=8,
    min_component_weight=0.05,
    source_prior_probs=(0.45, 0.20, 0.15, 0.05),
    linear_activation_prob=0.60,
    small_mlp_prob=0.25,
    soft_tree_prob=0.15,
    small_mlp_hidden_dim=None,
    soft_tree_depth=2,
    soft_tree_temperature=0.5,
    device=torch.device("cpu"),
)


train_dataset = SyntheticTaskDataset(
    num_tasks=50000,
    task_factory=WeightedMixedScalarSCMTask,
    task_kind="classification",
    min_classes=2,
    max_classes=4,
    base_seed=0,
    task_kwargs=TASK_KWARGS
)

val_dataset = SyntheticTaskDataset(
    num_tasks=5000,
    task_factory=WeightedMixedScalarSCMTask,
    task_kind="classification",
    min_classes=2,
    max_classes=4,
    base_seed=100000,
    task_kwargs=TASK_KWARGS,
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


train_synthetic(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    steps=10000,
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
 

 
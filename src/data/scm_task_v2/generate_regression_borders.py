import torch
from pathlib import Path
from ..datasets import SyntheticTaskDataset
from .task import SCMTask
from .priors import PRIOR

NUM_TASKS = 5000
NUM_BUCKETS = 100
BASE_SEED = 17
OUT_PATH = Path(__file__).resolve().parent / "borders_100.pt"

dataset = SyntheticTaskDataset(
    num_tasks=NUM_TASKS,
    task_factory=SCMTask,
    task_kind="regression",
    base_seed=BASE_SEED,
    task_kwargs=PRIOR
)

all_z = []

for i in range(len(dataset)):
    task = dataset[i]

    if task is None:
        continue

    y = task.y_train.float()

    y_mean = y.mean()
    y_std = y.std(unbiased=False).clamp_min(1e-6)

    z = (y - y_mean) / y_std

    all_z.append(z.cpu())

all_z = torch.cat(all_z)

quantiles = torch.linspace(0.001, 0.999, NUM_BUCKETS + 1)
borders = torch.quantile(all_z, quantiles)

borders[0] = -float("inf")
borders[-1] = float("inf")

torch.save(
    {
        "borders": borders,
        "num_buckets": NUM_BUCKETS,
        "num_tasks": NUM_TASKS,
        "num_samples": int(all_z.numel()),
    },
    OUT_PATH,
)

print("saved:", OUT_PATH)
print("borders shape:", borders.shape)
print("num samples:", all_z.numel())
print(borders)
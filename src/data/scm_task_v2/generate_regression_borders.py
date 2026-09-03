import torch
from pathlib import Path

from ..datasets import SyntheticTaskDataset
from .task import SCMTask
from ..config import SCM_PRIOR, LINEAR_PRIOR

from ..linear_task import LinearTask



NUM_TASKS = 5000
NUM_BUCKETS = 100
BASE_SEED = 17

SCM_RATIO = 0.5
LINEAR_RATIO = 0.5

OUT_PATH = Path(__file__).resolve().parent.parent / "borders_100.pt"


num_scm_tasks = int(NUM_TASKS * SCM_RATIO)
num_linear_tasks = NUM_TASKS - num_scm_tasks


scm_dataset = SyntheticTaskDataset(
    num_tasks=num_scm_tasks,
    task_factory=SCMTask,
    task_kind="regression",
    base_seed=BASE_SEED,
    task_kwargs=SCM_PRIOR,
)

linear_dataset = SyntheticTaskDataset(
    num_tasks=num_linear_tasks,
    task_factory=LinearTask,
    task_kind="regression",
    base_seed=BASE_SEED + num_scm_tasks,
    task_kwargs=LINEAR_PRIOR,
)


all_z = []


def collect_standardized_targets(dataset):
    for i in range(len(dataset)):
        task = dataset[i]

        if task is None:
            continue

        y = task.y_train.float()

        y_mean = y.mean()
        y_std = y.std(unbiased=False).clamp_min(1e-6)

        z = (y - y_mean) / y_std

        all_z.append(z.cpu())


collect_standardized_targets(scm_dataset)
collect_standardized_targets(linear_dataset)


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
        "num_scm_tasks": num_scm_tasks,
        "num_linear_tasks": num_linear_tasks,
        "num_samples": int(all_z.numel()),
    },
    OUT_PATH,
)

print("saved:", OUT_PATH)
print("borders shape:", borders.shape)
print("num samples:", all_z.numel())
print("num scm tasks:", num_scm_tasks)
print("num linear tasks:", num_linear_tasks)
print(borders)
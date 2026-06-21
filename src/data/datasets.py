from torch.utils.data import Dataset
import random

class SyntheticTaskDataset(Dataset):
    def __init__(
        self,
        length,
        task_factory,
        task_kwargs=None,
        task_kind="classification",   # "classification" or "regression"
        min_classes=2,
        max_classes=10,
        base_seed=0,
    ):
        self.length = int(length)
        self.task_factory = task_factory
        self.task_kwargs = dict(task_kwargs or {})
        self.task_kind = task_kind
        self.min_classes = int(min_classes)
        self.max_classes = int(max_classes)
        self.base_seed = int(base_seed)

        assert self.task_kind in ["classification", "regression"]
        assert "num_classes" not in self.task_kwargs

        if self.task_kind == "classification":
            assert self.min_classes >= 2
            assert self.max_classes >= self.min_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = random.Random(self.base_seed + int(idx))

        dag_seed = rng.randrange(2**31)
        x_seed = rng.randrange(2**31)
        aleatoric_seed = rng.randrange(2**31)

        if self.task_kind == "classification":
            num_classes = rng.randint(self.min_classes, self.max_classes)
        else:
            num_classes = None

        return self.task_factory(
            **self.task_kwargs,
            num_classes=num_classes,
            dag_seed=dag_seed,
            x_seed=x_seed,
            aleatoric_seed=aleatoric_seed,
        )

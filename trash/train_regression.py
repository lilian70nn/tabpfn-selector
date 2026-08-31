import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn.functional as F

class GenerateTask(ABC):
    def __init__(self) -> None:
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._info = None
        self.n_features = -1
        self.n_classes = None

        with torch.inference_mode():
            Xtr, ytr, Xte, yte, info = self._generate()

        self._X_train = Xtr
        self._y_train = ytr
        self._X_test = Xte
        self._y_test = yte
        self._info = info

    @abstractmethod
    def _generate(self):
        pass

    @abstractmethod
    def visualize(self) -> None:
        raise NotImplementedError

    @property
    def X_train(self):
        return self._X_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def X_test(self):
        return self._X_test

    @property
    def y_test(self):
        return self._y_test

    @property
    def info(self):
        return self._info
    
def make_gen(device, seed):
    g = torch.Generator(device=device)
    if seed is None:
        seed = int(g.seed())
    else:
        seed = int(seed)
        g.manual_seed(seed)
    return g, seed



def stratified_classification_split(y, test_frac, generator, device):
    y = y.long()
    classes = torch.unique(y, sorted=True)

    train_parts = []
    test_parts = []

    for c in classes:
        idx = torch.nonzero(y == c, as_tuple=False).flatten()
        idx = idx[torch.randperm(idx.numel(), device=device, generator=generator)]

        n_c = idx.numel()
        n_test_c = int(round(float(n_c) * float(test_frac)))

        if n_c >= 2:
            n_test_c = max(1, min(n_test_c, n_c - 1))
        else:
            n_test_c = 0

        test_parts.append(idx[:n_test_c])
        train_parts.append(idx[n_test_c:])

    train_idx = torch.cat(train_parts)
    test_idx = torch.cat(test_parts)

    train_idx = train_idx[
        torch.randperm(train_idx.numel(), device=device, generator=generator)
    ]
    test_idx = test_idx[
        torch.randperm(test_idx.numel(), device=device, generator=generator)
    ]

    return train_idx, test_idx


def discretize_latent_random_bins(
    latent_y,
    C,
    generator,
    min_per_class=2,
    alpha=5.0,
):
    n = latent_y.shape[0]
    device = latent_y.device

    assert C >= 2
    assert n >= C * min_per_class, (
        f"Need n >= C * min_per_class, got n={n}, C={C}, "
        f"min_per_class={min_per_class}"
    )

    order = torch.argsort(latent_y)

    weights = torch.rand(C, device=device, generator=generator)
    weights = weights.pow(1.0 / float(alpha))
    props = weights / weights.sum().clamp_min(1e-12)

    remaining_n = n - C * min_per_class
    counts = torch.floor(props * remaining_n).long()
    counts = counts + min_per_class

    diff = int(n - counts.sum().item())

    if diff > 0:
        extra_idx = torch.randperm(C, device=device, generator=generator)
        for k in range(diff):
            counts[extra_idx[k % C]] += 1

    elif diff < 0:
        need = -diff
        for c in torch.randperm(C, device=device, generator=generator).tolist():
            removable = int((counts[c] - min_per_class).item())
            take = min(removable, need)
            counts[c] -= take
            need -= take
            if need == 0:
                break
        assert need == 0

    assert int(counts.sum().item()) == n
    assert int(counts.min().item()) >= min_per_class

    y = torch.empty(n, device=device, dtype=torch.long)

    start = 0
    for c in range(C):
        end = start + int(counts[c].item())
        y[order[start:end]] = c
        start = end

    return y


class MixedLinearTask(GenerateTask):
    """
    Mixed tabular synthetic prior.

    Supports:
    - continuous features
    - categorical features
    - missing values in observed X
    - regression if num_classes is None
    - classification if num_classes >= 2

    Key convention:
    - X_clean is used to generate y.
    - X_obs is shown to the model and may contain NaN.
    - categorical values are stored as category ids: 0, 1, ..., K_j - 1.
    - categorical features affect y through lookup effects, not id * weight.
    """

    CONTINUOUS = 0
    CATEGORICAL = 1

    def __init__(
        self,
        num_classes=None,          # None = regression, int >= 2 = classification
        n_max=500,
        d_max=20,
        n_min=128,
        d_min=2,
        test_frac=0.15,
        p_categorical=0.3,
        max_cardinality=10,
        p_active=0.5,
        p_missing=0.05,
        noise_level=0.1,
        device=None,
        dag_seed=None,
        aleatoric_seed=None,
        x_seed=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device

        self.num_classes = num_classes
        self.n_max = int(n_max)
        self.d_max = int(d_max)
        self.n_min = int(n_min)
        self.d_min = int(d_min)
        self.test_frac = float(test_frac)
        self.p_categorical = float(p_categorical)
        self.max_cardinality = int(max_cardinality)
        self.p_active = float(p_active)
        self.p_missing = float(p_missing)
        self.noise_level = float(noise_level)

        if self.num_classes is not None:
            self.num_classes = int(self.num_classes)
            assert self.num_classes >= 2, "num_classes must be None or >= 2"

        assert self.d_max >= 2
        assert self.n_max >= 32
        assert self.n_min >= 3
        assert self.n_max >= self.n_min
        assert self.d_min >= 1
        assert self.d_max >= self.d_min
        assert 0.0 < self.test_frac < 1.0
        assert 0.0 <= self.p_categorical <= 1.0
        assert self.max_cardinality >= 2
        assert 0.0 <= self.p_active <= 1.0
        assert 0.0 <= self.p_missing <= 1.0
        assert self.noise_level >= 0.0

        self.g_dag, self.dag_seed = make_gen(device, dag_seed)
        self.g_aleatoric, self.aleatoric_seed = make_gen(device, aleatoric_seed)
        self.g_x, self.x_seed = make_gen(device, x_seed)

        self.d = torch.randint(d_min, d_max + 1, (1,), device=device, generator=self.g_dag).item()
        self.n = torch.randint(n_min, n_max + 1, (1,), device=device, generator=self.g_dag).item()

        super().__init__()


    def _generate(self):

        n, d, device = self.n, self.d, self.device

        # 1. Decide feature types
        is_cat = torch.rand(d, device=device, generator=self.g_dag) < self.p_categorical
        feature_type = is_cat.long()
        cardinality = torch.zeros(d, device=device, dtype=torch.long)
        for j in range(d):
            if bool(is_cat[j]):
                cardinality[j] = torch.randint(2, self.max_cardinality + 1, (1,), device=device, generator=self.g_dag,).item()

        # 2. Generate X_clean
        X_clean = torch.empty(n, d, device=device, dtype=torch.float32)
        for j in range(d):
            if bool(is_cat[j]):
                K_j = int(cardinality[j].item())
                probs = torch.rand(K_j, device=device, generator=self.g_dag)
                probs = probs / probs.sum().clamp_min(1e-12)
                X_clean[:, j] = torch.multinomial(probs, num_samples=n, replacement=True, generator=self.g_x,).float()

            else:
                scale = torch.exp(0.5 * torch.randn((), device=device, generator=self.g_dag))
                shift = torch.randn((), device=device, generator=self.g_dag)
                X_clean[:, j] = (
                    scale * torch.randn(n, device=device, generator=self.g_x)
                    + shift
                )

        # 3. Decide active features
        active = torch.rand(d, device=device, generator=self.g_dag) < self.p_active
        if not bool(active.any()):
            idx = torch.randint(0, d, (1,), device=device, generator=self.g_dag).item()
            active[idx] = True
        feature_strength = torch.zeros(d, device=device, dtype=torch.float32)

        # 4. Generate scalar latent_y for both regression and classification
        latent_y = torch.zeros(n, device=device, dtype=torch.float32)
        for j in range(d):
            if not bool(active[j]):
                continue
            if bool(is_cat[j]):
                K_j = int(cardinality[j].item())
                # categorical feature: lookup scalar effect
                effects = torch.randn(K_j, device=device, generator=self.g_dag)
                xj = X_clean[:, j].long().clamp(0, K_j - 1)
                contrib_j = effects[xj]  # [n]
            else:
                # continuous feature: scalar linear effect
                w_j = torch.randn((), device=device, generator=self.g_dag)
                contrib_j = w_j * X_clean[:, j]  # [n]
            latent_y = latent_y + contrib_j
            feature_strength[j] = contrib_j.std(unbiased=False)

        # 5. Add noise
        noise_scale = self.noise_level * latent_y.std(unbiased=False).clamp_min(1e-6)
        noise = noise_scale * torch.randn(n, device=device, generator=self.g_aleatoric,)
        latent_y = latent_y + noise

        # 6. Convert latent_y to y
        if self.num_classes is None:
            # Regression: y is continuous
            y = latent_y
            self.n_classes = None
        else:
            # Classification: discretize latent_y into C quantile bins
            C = int(self.num_classes)
            y = discretize_latent_random_bins(
                latent_y=latent_y,
                C=C,
                generator=self.g_aleatoric,
                min_per_class=2,
                alpha=5.0,
            )
            self.n_classes = C

        # 7. Add missing values to observed X
        X_obs = X_clean.clone()
        if self.p_missing > 0:
            missing_mask = (
                torch.rand(X_obs.shape, device=device, generator=self.g_x)
                < self.p_missing
            )
            X_obs[missing_mask] = torch.nan
        else:
            missing_mask = torch.zeros_like(X_obs, dtype=torch.bool)

        # 8. Split train/test
        if self.num_classes is not None:
            train_idx, test_idx = stratified_classification_split(
                y=y.long(),
                test_frac=self.test_frac,
                generator=self.g_x,
                device=device,
            )
        else:
            n_test = max(1, int(round(self.n * self.test_frac)))
            n_test = min(n_test, self.n - 2)
            n_train = self.n - n_test
            perm = torch.randperm(self.n, device=device, generator=self.g_x)
            train_idx = perm[:n_train]
            test_idx = perm[n_train:]

        X_train = X_obs[train_idx]
        y_train = y[train_idx]
        X_test = X_obs[test_idx]
        y_test = y[test_idx]

        # 9. Metadata
        self.n_features = d
        self.feature_type = feature_type
        self.cardinality = cardinality

        eps = 1e-8
        is_active = (feature_strength > eps).float()
        importance_ratio = feature_strength / feature_strength.sum().clamp_min(1e-12)

        info = {
            "feature_type": feature_type,
            "cardinality": cardinality,
            "is_active": is_active,
            "importance_ratio": importance_ratio,
            "feature_strength": feature_strength,
            "sampled_active": active.float(),
            "missing_mask_train": missing_mask[train_idx],
            "missing_mask_test": missing_mask[test_idx],
        }

        return X_train, y_train, X_test, y_test, info

    def visualize(self):
        return None

    def forward(self, X: torch.Tensor):
        return None
    

def build_cell_mask(
    B,
    Ntr_max,
    Nte_max,
    d_max,
    n_train,
    n_test,
    d_emb,
    device,
    use_selector=False,
):
    if not torch.is_tensor(n_train):
        n_train = torch.tensor(n_train, device=device, dtype=torch.long)
    else:
        n_train = n_train.to(device=device, dtype=torch.long)

    if not torch.is_tensor(n_test):
        n_test = torch.tensor(n_test, device=device, dtype=torch.long)
    else:
        n_test = n_test.to(device=device, dtype=torch.long)

    if not torch.is_tensor(d_emb):
        d_emb = torch.tensor(d_emb, device=device, dtype=torch.long)
    else:
        d_emb = d_emb.to(device=device, dtype=torch.long)

    N = Ntr_max + 1 + Nte_max
    F = d_max + 1
    selector_idx = Ntr_max
    test_start = Ntr_max + 1
    y_slot = d_max

    idx_N = torch.arange(N, device=device).view(1, N)
    train_ok = idx_N < n_train.view(B, 1)
    test_ok = (idx_N >= test_start) & (idx_N < (test_start + n_test).view(B, 1))
    normal_row_ok = train_ok | test_ok
    idx_F = torch.arange(F, device=device).view(1, F)
    feat_ok = idx_F < d_emb.view(B, 1)
    y_ok = idx_F == y_slot
    normal_slot_ok = feat_ok | y_ok
    cell_mask = normal_row_ok[:, :, None] & normal_slot_ok[:, None, :]

    if use_selector:
        selector_ok = idx_N == selector_idx
        # selector row: only real feature slots, no y slot
        selector_cell_mask = selector_ok[:, :, None] & feat_ok[:, None, :]
        cell_mask = cell_mask | selector_cell_mask

    return cell_mask


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


from dataclasses import dataclass

@dataclass
class TaskBatch:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor

    Ntr_max: int
    Nte_max: int
    d_max: int

    n_train: torch.Tensor
    n_test: torch.Tensor
    d_emb: torch.Tensor

    feature_type: torch.Tensor
    cardinality: torch.Tensor

    is_active: torch.Tensor
    importance_ratio: torch.Tensor
    feature_strength: torch.Tensor

    cell_mask: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor | None
    y_std: torch.Tensor | None

    n_classes: torch.Tensor | None
    use_selector: bool = True


def collate_tasks(tasks, use_selector=True):
    B = len(tasks)
    device = tasks[0].X_train.device

    n_train = torch.tensor(
        [t.X_train.shape[0] for t in tasks],
        dtype=torch.long,
        device=device,
    )
    n_test = torch.tensor(
        [t.X_test.shape[0] for t in tasks],
        dtype=torch.long,
        device=device,
    )
    d_emb = torch.tensor(
        [t.X_train.shape[1] for t in tasks],
        dtype=torch.long,
        device=device,
    )

    Ntr_max = int(n_train.max().item())
    Nte_max = int(n_test.max().item())
    d_max = int(d_emb.max().item())

    y_dtype = tasks[0].y_train.dtype

    X_train = torch.full(
        (B, Ntr_max, d_max),
        torch.nan,
        dtype=torch.float32,
        device=device,
    )
    X_test = torch.full(
        (B, Nte_max, d_max),
        torch.nan,
        dtype=torch.float32,
        device=device,
    )

    y_train = torch.zeros(
        (B, Ntr_max),
        dtype=y_dtype,
        device=device,
    )
    y_test = torch.zeros(
        (B, Nte_max),
        dtype=y_dtype,
        device=device,
    )

    feature_type = torch.zeros(
        (B, d_max),
        dtype=torch.long,
        device=device,
    )
    cardinality = torch.zeros(
        (B, d_max),
        dtype=torch.long,
        device=device,
    )

    is_active = torch.zeros(
        (B, d_max),
        dtype=torch.float32,
        device=device,
    )
    importance_ratio = torch.zeros(
        (B, d_max),
        dtype=torch.float32,
        device=device,
    )
    feature_strength = torch.zeros(
        (B, d_max),
        dtype=torch.float32,
        device=device,
    )

    x_mean = torch.zeros((B, d_max), dtype=torch.float32, device=device)
    x_std = torch.ones((B, d_max), dtype=torch.float32, device=device)

    y_mean = torch.zeros((B,), dtype=torch.float32, device=device)
    y_std = torch.ones((B,), dtype=torch.float32, device=device)

    n_classes_list = []

    for b, task in enumerate(tasks):

        nt = task.X_train.shape[0]
        ne = task.X_test.shape[0]
        d = task.X_train.shape[1]

        ft = task.info["feature_type"].to(device=device)
        is_cont = ft == 0

        Xtr_i = task.X_train.float()

        mean_i = torch.zeros(d, dtype=torch.float32, device=device)
        std_i = torch.ones(d, dtype=torch.float32, device=device)

        if bool(is_cont.any()):
            X_cont = Xtr_i[:, is_cont]

            cont_mean = torch.nanmean(X_cont, dim=0)

            centered = X_cont - cont_mean[None, :]
            cont_var = torch.nanmean(centered ** 2, dim=0)
            cont_std = torch.sqrt(cont_var).clamp_min(1e-6)

            cont_mean = torch.nan_to_num(cont_mean, nan=0.0)
            cont_std = torch.nan_to_num(cont_std, nan=1.0).clamp_min(1e-6)

            mean_i[is_cont] = cont_mean
            std_i[is_cont] = cont_std

        x_mean[b, :d] = mean_i
        x_std[b, :d] = std_i

        if task.n_classes is None:
            ytr_i = task.y_train.float()
            y_mean[b] = ytr_i.mean()
            y_std[b] = ytr_i.std(unbiased=False).clamp_min(1e-6)

        X_train[b, :nt, :d] = task.X_train
        y_train[b, :nt] = task.y_train

        X_test[b, :ne, :d] = task.X_test
        y_test[b, :ne] = task.y_test

        feature_type[b, :d] = task.info["feature_type"]
        cardinality[b, :d] = task.info["cardinality"]

        is_active[b, :d] = task.info["is_active"]
        importance_ratio[b, :d] = task.info["importance_ratio"]
        feature_strength[b, :d] = task.info["feature_strength"]

        # n_classes_list.append(task.n_classes)
        if task.n_classes is None:
            n_classes_list.append(None)
        else:
            #n_classes_list.append(int(task.n_classes))
            ytr = task.y_train.long()
            n_classes_list.append(int(ytr.max().item()) + 1)

    all_regression = all(c is None for c in n_classes_list)
    all_classification = all(c is not None for c in n_classes_list)

    assert all_regression or all_classification, (
        "Do not mix regression and classification tasks in one batch."
    )

    if all_regression:
        n_classes = None
    else:
        n_classes = torch.tensor(
            [int(c) for c in n_classes_list],
            dtype=torch.long,
            device=device,
        )

        y_mean = None
        y_std = None

    cell_mask = build_cell_mask(
            B=B,
            Ntr_max=Ntr_max,
            Nte_max=Nte_max,
            d_max=d_max,
            n_train=n_train,
            n_test=n_test,
            d_emb=d_emb,
            device=device,
            use_selector=use_selector,
        )

    return TaskBatch(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        Ntr_max=Ntr_max,
        Nte_max=Nte_max,
        d_max=d_max,
        n_train=n_train,
        n_test=n_test,
        d_emb=d_emb,
        feature_type=feature_type,
        cardinality=cardinality,
        is_active=is_active,
        importance_ratio=importance_ratio,
        feature_strength=feature_strength,
        cell_mask=cell_mask,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        n_classes=n_classes,
        use_selector=use_selector,

    )

from torch import nn

def make_regression_borders(
    num_bins: int = 100,
    low: float = -3.0,
    high: float = 3.0,
    device=None,
    dtype=torch.float32,
):
    assert num_bins >= 2
    assert low < high

    return torch.linspace(
        low,
        high,
        num_bins + 1,
        device=device,
        dtype=dtype,
    )


def bucketize_y(
    y_z: torch.Tensor,
    borders: torch.Tensor,
):
    num_bins = borders.numel() - 1
    y_bucket = torch.bucketize(
        y_z,
        borders.to(device=y_z.device, dtype=y_z.dtype),
        right=False,
    ) - 1
    y_bucket = y_bucket.clamp(0, num_bins - 1)

    return y_bucket.long()

class TabularInputEncoder(nn.Module):
    CONTINUOUS = 0
    CATEGORICAL = 1

    def __init__(
        self,
        k: int,
        max_cardinality: int,
        task_kind: str,
        max_classes: int | None = None,
        num_y_buckets: int | None = None,
    ):
        super().__init__()

        assert task_kind in ["classification", "regression"]

        self.k = int(k)
        self.max_cardinality = int(max_cardinality)
        self.task_kind = task_kind

        self.feature_id_dim = max(1, k // 4)
        self.feature_id_proj = nn.Linear(self.feature_id_dim, k)

        self.cont_encoder = nn.Linear(1, self.k)
        self.cat_encoder = nn.Embedding(self.max_cardinality + 1, self.k)
        self.feature_type_embed = nn.Embedding(2, self.k)

        self.missing_token = nn.Parameter(torch.randn(self.k) * 0.02)
        self.y_unknown_token = nn.Parameter(torch.randn(self.k) * 0.02)
        self.selector_token = nn.Parameter(torch.randn(self.k) * 0.02)

        self.max_classes = None
        self.num_y_buckets = None
        self.y_class_encoder = None
        self.y_reg_encoder = None
        #self.regression_borders = None

        if self.task_kind == "classification":
            assert max_classes is not None
            assert num_y_buckets is None
            assert max_classes >= 2
            self.max_classes = int(max_classes)
            self.y_class_encoder = nn.Embedding(max_classes, self.k)
            self.regression_borders = None

        else:

            assert num_y_buckets is not None
            assert max_classes is None
            assert num_y_buckets >= 2

            self.num_y_buckets = int(num_y_buckets)
            self.y_reg_encoder = nn.Embedding(self.num_y_buckets, self.k)

            self.register_buffer(
                "regression_borders",
                make_regression_borders(
                    num_bins=self.num_y_buckets,
                    low=-3.0,
                    high=3.0,
                ),
            )


    def forward(self, batch):
        device = batch.X_train.device

        B = batch.X_train.shape[0]
        Ntr_max = batch.Ntr_max
        Nte_max = batch.Nte_max
        d_max = batch.d_max

        selector_idx = Ntr_max
        test_start = Ntr_max + 1

        N = Ntr_max + 1 + Nte_max
        F = d_max + 1
        y_slot = d_max

        cell_mask = batch.cell_mask.to(device=device)
        assert cell_mask.shape == (B, N, F)
        assert batch.x_mean.shape == (B, d_max)
        assert batch.x_std.shape == (B, d_max)

        tokens = torch.zeros(B, N, F, self.k, device=device)

        feature_type = batch.feature_type.clamp(0, 1)  # [B, d_max]
        is_cont = feature_type == self.CONTINUOUS      # [B, d_max]
        is_cat = feature_type == self.CATEGORICAL      # [B, d_max]

        row_idx = torch.arange(N, device=device)[None, :, None]
        not_selector_row = row_idx != selector_idx

        type_tokens = self.feature_type_embed(feature_type)  # [B, d_max, K]

        # X feature slots
        feature_tokens = torch.zeros(B, N, d_max, self.k, device=device)

        X_all = torch.full(
            (B, N, d_max),
            torch.nan,
            dtype=torch.float32,
            device=device,
        )

        X_all[:, :Ntr_max, :] = batch.X_train
        X_all[:, test_start:, :] = batch.X_test
        # selector row stays NaN; later overwritten by selector token

        feature_mask = cell_mask[:, :, :d_max]  # [B, N, d_max]
        X_nan = torch.isnan(X_all)

        X_all_norm = (
            torch.nan_to_num(X_all, nan=0.0)
            - batch.x_mean[:, None, :]
        ) / batch.x_std[:, None, :]

        type_all = type_tokens[:, None, :, :].expand(B, N, d_max, self.k)
        cont_cell = (feature_mask & is_cont[:, None, :] & ~X_nan)
        cat_cell = (feature_mask & is_cat[:, None, :] & ~X_nan)
        missing_cell = (feature_mask & X_nan & not_selector_row)

        # continuous cells
        if bool(cont_cell.any()):
            vals = X_all_norm[cont_cell]          # [num_cont_cells]
            enc = self.cont_encoder(vals[:, None])
            feature_tokens[cont_cell] = enc + type_all[cont_cell]

        # categorical cells
        cat_ids = torch.nan_to_num(X_all, nan=0.0).long()
        cat_ids = cat_ids.clamp(0, self.max_cardinality)

        if bool(cat_cell.any()):
            ids = cat_ids[cat_cell]
            enc = self.cat_encoder(ids)
            feature_tokens[cat_cell] = enc + type_all[cat_cell]

        # missing cells
        if bool(missing_cell.any()):
            feature_tokens[missing_cell] = (
                self.missing_token[None, :]
                + type_all[missing_cell]
            )

        # selector row: overwrite selector feature cells
        feature_tokens[:, selector_idx, :, :] = (
            self.selector_token.view(1, 1, self.k) + type_tokens
        )

        tokens[:, :, :d_max, :] = feature_tokens

        # y slot
        train_y_mask = cell_mask[:, :Ntr_max, y_slot]       # [B, Ntr]
        test_y_mask = cell_mask[:, test_start:, y_slot]     # [B, Nte]

        if self.task_kind == "classification":
            assert batch.n_classes is not None
            assert self.y_class_encoder is not None
            assert int(batch.n_classes.max().item()) <= self.max_classes

            y_train_ids = batch.y_train.long().clamp(0, self.max_classes - 1)
            y_train_tokens = self.y_class_encoder(y_train_ids)

        else:
            assert batch.n_classes is None
            assert batch.y_mean is not None
            assert batch.y_std is not None
            assert self.y_reg_encoder is not None
            assert self.regression_borders is not None

            y_z = (
                batch.y_train.float()
                - batch.y_mean[:, None]
            ) / batch.y_std[:, None]
            y_bucket = bucketize_y(y_z, self.regression_borders)
            y_train_tokens = self.y_reg_encoder(y_bucket)

        tokens[:, :Ntr_max, y_slot, :] = torch.where(train_y_mask[:, :, None], y_train_tokens, tokens[:, :Ntr_max, y_slot, :],)
        y_unknown = self.y_unknown_token.view(1, 1, self.k).expand(B, Nte_max, self.k,)
        tokens[:, test_start:, y_slot, :] = torch.where(test_y_mask[:, :, None], y_unknown, tokens[:, test_start:, y_slot, :],)
        feature_noise = torch.randn(B, d_max, self.feature_id_dim, device=tokens.device, dtype=tokens.dtype,)
        feature_id = self.feature_id_proj(feature_noise)  # [B, d_max, K]
        row_has_x = torch.ones(B, N, device=tokens.device, dtype=torch.bool)
        row_has_x[:, selector_idx] = False
        feat_id_mask = cell_mask[:, :, :d_max] & row_has_x[:, :, None]
        tokens[:, :, :d_max, :] = tokens[:, :, :d_max, :] + (feature_id[:, None, :, :] * feat_id_mask[:, :, :, None].to(tokens.dtype))
        # padding / invalid cells -> zero vector
        tokens = tokens * cell_mask[:, :, :, None].to(tokens.dtype)

        assert tokens.shape == (B, N, F, self.k)
        assert torch.all(tokens[~cell_mask] == 0)

        meta = {
            "B": B,
            "N": N,
            "F": F,
            "Ntr_max": Ntr_max,
            "Nte_max": Nte_max,
            "d_max": d_max,
            "selector_idx": selector_idx,
            "test_start": test_start,
            "y_slot": y_slot,
            "n_train_keys": Ntr_max + 1,
        }

        if self.task_kind == "regression":
            meta["y_mean"] = batch.y_mean
            meta["y_std"] = batch.y_std
            meta["regression_borders"] = self.regression_borders
            meta["num_y_buckets"] = self.num_y_buckets

        return tokens, cell_mask, meta
    



class AxisAttention(nn.Module):
    """
    Generic self-attention over the second-to-last axis.

    Expected input:
        x:         [B, G, L, K]
        cell_mask: [B, G, L]
                   True  = valid token/cell
                   False = padding token/cell

    It computes attention over L:
        scores: [B, heads, G, L_query, L_key]

    For feature attention:
        x         = data_full                         # [B, N, F, K]
        cell_mask = cell_mask                         # [B, N, F]
        restrict_to_train_keys = False

    For sample attention:
        x         = data_full.permute(0, 2, 1, 3)      # [B, F, N, K]
        cell_mask = cell_mask.permute(0, 2, 1)         # [B, F, N]
        restrict_to_train_keys = True
    """

    def __init__(self, k, n_heads):
        super().__init__()
        self.k = k
        self.n_heads = n_heads

        assert k % n_heads == 0, "k must be divisible by n_heads"
        self.d_k = k // n_heads

        self.W_q = nn.Linear(k, k, bias=False)
        self.W_k = nn.Linear(k, k, bias=False)
        self.W_v = nn.Linear(k, k, bias=False)
        self.W_c = nn.Linear(k, k, bias=False)

    def forward(
        self,
        data,
        cell_mask,
        restrict_to_train_keys=False,
        n_train_keys=None,
        attn_bias=None,
    ):
        """
        data:         [B, G, L, K]
        cell_mask: [B, G, L], bool

        restrict_to_train_keys:
            If True, keys with index >= n_train_keys are masked out.
            This is useful for sample/row attention where only train rows
            can be attended to.

        n_train_keys:
            Usually Ntr_max when L is the row/sample axis.

        attn_bias:
            Optional additive attention bias.
            Shape should be broadcastable to [B, heads, G, L, L].
            Example: [B, 1, G, L, L] or [B, 1, 1, 1, L].

        return_attn:
            If True, return (proj, attn). Otherwise return proj.
        """
        B, G, L, K = data.shape
        assert K == self.k, f"Expected hidden dim {self.k}, got {K}"

        if cell_mask.dtype != torch.bool:
            cell_mask = cell_mask.bool()

        # Q/K/V: [B, G, L, K]
        Q = self.W_q(data)
        K_ = self.W_k(data)
        V = self.W_v(data)

        # [B, G, L, K] -> [B, heads, G, L, d_k]
        Q = Q.view(B, G, L, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)
        K_ = K_.view(B, G, L, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)
        V = V.view(B, G, L, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)

        # scores: [B, heads, G, L_query, L_key]
        scores = (Q @ K_.transpose(-2, -1)) / (self.d_k ** 0.5)

        # pair_mask: [B, G, L_query, L_key]
        # valid query can attend valid key
        pair_mask = cell_mask[:, :, :, None] & cell_mask[:, :, None, :]

        # Optional: for sample attention, only allow keys before n_train_keys.
        # This reproduces your old key_ok = train_ok logic.
        if restrict_to_train_keys:
            if n_train_keys is None:
                raise ValueError("n_train_keys must be provided when restrict_to_train_keys=True")

            pair_mask = pair_mask.clone()
            pair_mask[..., n_train_keys:] = False

        # Apply hard mask.
        # Masked positions become -inf before softmax.
        scores = scores.masked_fill(~pair_mask[:, None, :, :, :], float("-inf"))

        # Optional additive bias, e.g. selector score bias.
        if attn_bias is not None:
            scores = scores + attn_bias

        attn = torch.softmax(scores, dim=-1)

        # If a padding query has no valid keys, softmax([-inf, ...]) gives NaN.
        # We intentionally convert those rows to zero attention.
        attn = torch.nan_to_num(attn, nan=0.0)

        out = attn @ V
        # out: [B, heads, G, L, d_k]

        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, G, L, self.k)

        proj = self.W_c(out)

        # Clear padding cells after projection.
        proj = proj * cell_mask[:, :, :, None].to(proj.dtype)

        return proj
    
class ResidualNorm(nn.Module):
    def __init__(self, k, sublayer):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(k)

    def forward(self, data, *args, **kwargs):
        data_temp = self.sublayer(self.norm(data), *args, **kwargs)
        return data_temp + data


class Feedforward(nn.Module):
    """
    Feedforward neural network.
    d_emb: embedding dimension
    m: hidden dimension

    Input: x of shape (n, k)
    Output: x of shape (n, k)
    """
    def __init__(self, k, m, dropout=0.1):
        super().__init__()
        self.m = m
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(k, m),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(m, k),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        data = self.net(data)
        return data



class TransformerBlock(nn.Module):
    def __init__(self, k, m, n_heads):
        super().__init__()
        self.fAtt = ResidualNorm(k, AxisAttention(k, n_heads))
        self.sAtt = ResidualNorm(k, AxisAttention(k, n_heads))
        self.forw = ResidualNorm(k, Feedforward(k, m))

    def forward(self, data, cell_mask, meta):
        data = self.fAtt(data, cell_mask)
        data = data * cell_mask[:, :, :, None].to(data.dtype)

        data = data.permute(0, 2, 1, 3).contiguous()       # [B, F, N, K]
        mask_t = cell_mask.permute(0, 2, 1).contiguous()   # [B, F, N]

        data = self.sAtt(
            data,
            mask_t,
            restrict_to_train_keys=True,
            n_train_keys=meta["n_train_keys"],
        )
        data = data * mask_t[:, :, :, None].to(data.dtype)

        data = data.permute(0, 2, 1, 3).contiguous()       # [B, N, F, K]
        data = self.forw(data)
        data = data * cell_mask[:, :, :, None].to(data.dtype)

        return data


class TabularBackbone(nn.Module):
    def __init__(self, k, m, n_heads, depth):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(k=k, m=m, n_heads=n_heads)
            for _ in range(depth)
        ])

    def forward(self, tokens, cell_mask, meta):
        x = tokens

        for block in self.blocks:
            x = block(x, cell_mask, meta)

        return x
    

class TabularPFNModel(nn.Module):
    def __init__(
        self,
        k: int,
        m: int,
        n_heads: int,
        depth: int,
        max_cardinality: int,
        task_kind: str,
        max_classes: int | None = None,
        num_y_buckets: int | None = None,
    ):
        super().__init__()

        assert task_kind in ["classification", "regression"]
        self.task_kind = task_kind

        self.encoder = TabularInputEncoder(
            k=k,
            max_cardinality=max_cardinality,
            task_kind=task_kind,
            max_classes=max_classes,
            num_y_buckets=num_y_buckets,
        )

        self.backbone = TabularBackbone(
            k=k,
            m=m,
            n_heads=n_heads,
            depth=depth,
        )

        self.importance_head = nn.Sequential(
            nn.LayerNorm(k),
            nn.Linear(k, k),
            nn.GELU(),
            nn.Linear(k, 1),
          )

        if task_kind == "classification":
            assert max_classes is not None
            self.head = nn.Linear(k, max_classes)
        else:
            assert num_y_buckets is not None
            self.head = nn.Linear(k, num_y_buckets)

    def forward(self, batch):
        tokens, cell_mask, meta = self.encoder(batch)

        h = self.backbone(tokens, cell_mask, meta)

        test_start = meta["test_start"]
        y_slot = meta["y_slot"]
        selector_idx = meta["selector_idx"]
        d_max = meta["d_max"]

        test_repr = h[:, test_start:, y_slot, :]      # [B, Nte_max, K]
        test_mask = cell_mask[:, test_start:, y_slot] # [B, Nte_max]

        importance_logits = None

        if bool(batch.use_selector):
            selector_repr = h[:, selector_idx, :d_max, :]          # [B, d_max, K]
            importance_logits = self.importance_head(selector_repr).squeeze(-1)  # [B, d_max]

        y_test_logits = self.head(test_repr)

        return {
            "logits": y_test_logits,
            "test_mask": test_mask,
            "importance_logits": importance_logits,
            "meta": meta,
        }

    def prediction_loss(self, batch, out):
        logits = out["logits"]
        test_mask = out["test_mask"]

        if self.task_kind == "classification":
            assert batch.n_classes is not None

            B, Nte_max, C = logits.shape
            class_idx = torch.arange(C, device=logits.device)[None, None, :]
            valid_class = class_idx < batch.n_classes[:, None, None]

            logits = logits.masked_fill(~valid_class, float("-inf"))
            target = batch.y_test.long()

            target_ok = target < batch.n_classes[:, None]
            assert bool(target_ok[test_mask].all())

            return F.cross_entropy(
                logits[test_mask],
                target[test_mask],
            )

        else:
            assert batch.y_mean is not None
            assert batch.y_std is not None

            y_z = (
                batch.y_test.float()
                - batch.y_mean[:, None]
            ) / batch.y_std[:, None]

            target_bucket = bucketize_y(
                y_z,
                self.encoder.regression_borders,
            )

            return F.cross_entropy(
                logits[test_mask],
                target_bucket[test_mask],
            )

    def importance_loss(self, batch, out):

        assert bool(batch.use_selector)
        assert out["importance_logits"] is not None

        logits = out["importance_logits"]  # [B, d_max]
        pred = torch.sigmoid(logits)

        feat_idx = torch.arange(batch.d_max, device=pred.device)[None, :]
        feat_mask = feat_idx < batch.d_emb[:, None]

        target = batch.importance_ratio.float()  # [B, d_max]

        return F.mse_loss(
            pred[feat_mask],
            target[feat_mask],
        )


    def total_loss(self, batch, out, importance_weight=None):
        pred_loss = self.prediction_loss(batch, out)

        loss = pred_loss

        result = {
            "loss": loss,
            "pred_loss": pred_loss,
        }

        if bool(batch.use_selector):
            assert importance_weight is not None
            assert importance_weight > 0

            importance_loss = self.importance_loss(batch, out)
            loss = pred_loss + importance_weight * importance_loss

            result["loss"] = loss
            result["importance_loss"] = importance_loss

        else:
            assert importance_weight is None

        return result


import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)


@torch.no_grad()
def classification_metrics(batch, out):
    logits = out["logits"]          # [B, Nte_max, C]
    test_mask = out["test_mask"]    # [B, Nte_max]

    assert batch.n_classes is not None

    B, Nte_max, C = logits.shape

    class_idx = torch.arange(C, device=logits.device)[None, None, :]
    valid_class = class_idx < batch.n_classes[:, None, None]
    logits = logits.masked_fill(~valid_class, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    y_pred = logits.argmax(dim=-1)  # [B, Nte_max]
    y_true = batch.y_test.long()

    accs = []
    balanced_accs = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []

    for b in range(B):
        mask_b = test_mask[b]

        yt = y_true[b, mask_b].detach().cpu().numpy()
        yp = y_pred[b, mask_b].detach().cpu().numpy()

        if len(yt) == 0:
            continue

        c_b = int(batch.n_classes[b].item())
        labels = list(range(c_b))

        p, r, f1, _ = precision_recall_fscore_support(
            yt,
            yp,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        accs.append(accuracy_score(yt, yp))
        balanced_accs.append(r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        # ROC-AUC
        prob_b = probs[b, mask_b, :c_b].detach().cpu().numpy()
        # AUC requires at least 2 classes in y_true for this task
        if len(set(yt.tolist())) >= 2:
            try:
                if c_b == 2:
                    auc = roc_auc_score(yt, prob_b[:, 1])

                else:
                    auc = roc_auc_score(
                        yt,
                        prob_b,
                        labels=labels,
                        multi_class="ovr",
                        average="macro",
                    )
                roc_aucs.append(auc)

            except ValueError:
                pass

    def avg(xs):
        return float(sum(xs) / max(len(xs), 1))

    return {
        "acc": avg(accs),
        "balanced_acc": avg(balanced_accs),
        "macro_precision": avg(precisions),
        "macro_recall": avg(recalls),
        "macro_f1": avg(f1s),
        "roc_auc": avg(roc_aucs),
    }


@torch.no_grad()
def importance_metrics(batch, out):

    assert bool(batch.use_selector)
    assert out["importance_logits"] is not None

    pred = torch.sigmoid(out["importance_logits"])  # [B, d_max]
    target = batch.importance_ratio.float()

    feat_idx = torch.arange(batch.d_max, device=pred.device)[None, :]
    feat_mask = feat_idx < batch.d_emb[:, None]

    p = pred[feat_mask]
    t = target[feat_mask]

    mse = F.mse_loss(p, t)

    p_center = p - p.mean()
    t_center = t - t.mean()

    pearson = (
        (p_center * t_center).mean()
        / (p_center.std(unbiased=False) * t_center.std(unbiased=False)).clamp_min(1e-12)
    )

    return {
        "importance_mse": float(mse.detach()),
        "importance_pearson": float(pearson.detach()),
    }


from dataclasses import fields
import torch

def move_batch_to_device(batch, device):
    kwargs = {}
    for f in fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v):
            kwargs[f.name] = v.to(device, non_blocking=True)
        else:
            kwargs[f.name] = v
    return type(batch)(**kwargs)

def infer_loader_use_selector(loader):
    batch = next(iter(loader))
    assert hasattr(batch, "use_selector"), (
        "Batch must contain use_selector. "
        "Set batch.use_selector inside collate_tasks."
    )
    return bool(batch.use_selector)



@torch.no_grad()
def evaluate_synthetic(
    model,
    loader,
    device,
    max_batches=50,
    importance_weight=None,
):
    model.eval()

    loader_use_selector = infer_loader_use_selector(loader)

    if loader_use_selector:
        assert importance_weight is not None
        assert importance_weight > 0
    else:
        assert importance_weight is None

    total_loss_sum = 0.0
    pred_loss_sum = 0.0
    imp_loss_sum = 0.0
    n_batches = 0
    n_imp_batches = 0

    metric_sums = {}
    metric_counts = {}

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        assert bool(batch.use_selector) == loader_use_selector

        batch = move_batch_to_device(batch, device)

        out = model(batch)
        loss_dict = model.total_loss(
            batch,
            out,
            importance_weight=importance_weight,
        )

        total_loss_sum += float(loss_dict["loss"].detach())
        pred_loss_sum += float(loss_dict["pred_loss"].detach())
        n_batches += 1

        if loader_use_selector:
            imp_loss_sum += float(loss_dict["importance_loss"].detach())
            n_imp_batches += 1

        if model.task_kind == "classification":
            metrics = classification_metrics(batch, out)
        else:
            metrics = {}

        if loader_use_selector:
            metrics.update(importance_metrics(batch, out))

        for k, v in metrics.items():
            if v != v:  # skip nan
                continue
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            metric_counts[k] = metric_counts.get(k, 0) + 1

    result = {
        "loss": total_loss_sum / max(n_batches, 1),
        "pred_loss": pred_loss_sum / max(n_batches, 1),
    }

    if loader_use_selector:
        result["importance_loss"] = imp_loss_sum / max(n_imp_batches, 1)

    for k, v in metric_sums.items():
        result[k] = v / max(metric_counts[k], 1)

    return result


import torch


def train_synthetic(
    model,
    train_loader,
    optimizer,
    device,
    steps=5000,
    importance_weight: float | None = None,
    grad_clip=1.0,
    log_every=50,
    val_loader=None,
    val_every=500,
    val_batches=50,
    save_path=None,
    best_ckpt_path=None,
):
    model.to(device)
    model.train()

    if save_path is not None:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write("")

    best_pred_loss = float("inf")

    if best_ckpt_path is not None:
        from pathlib import Path
        best_ckpt_path = Path(best_ckpt_path)
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def log_line(s):
        print(s, flush=True)
        if save_path is not None:
            with open(save_path, "a") as f:
                f.write(s + "\n")

    loader_use_selector = infer_loader_use_selector(train_loader)

    if loader_use_selector:
        assert importance_weight is not None
        assert importance_weight > 0
    else:
        assert importance_weight is None

    if val_loader is not None:
        val_use_selector = infer_loader_use_selector(val_loader)
        assert val_use_selector == loader_use_selector, (
            "train_loader and val_loader must use the same use_selector setting"
        )

    train_iter = iter(train_loader)

    running_loss = 0.0
    running_pred = 0.0
    running_imp = 0.0
    running_n = 0
    running_imp_n = 0

    for step in range(1, steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        assert bool(batch.use_selector) == loader_use_selector

        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        out = model(batch)
        loss_dict = model.total_loss(
            batch,
            out,
            importance_weight=importance_weight,
        )

        loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += float(loss_dict["loss"].detach())
        running_pred += float(loss_dict["pred_loss"].detach())
        running_n += 1

        if loader_use_selector:
            running_imp += float(loss_dict["importance_loss"].detach())
            running_imp_n += 1

        if step % log_every == 0:
            if loader_use_selector:
                log_line(
                    f"step {step:06d} | "
                    f"loss {running_loss / running_n:.4f} | "
                    f"pred {running_pred / running_n:.4f} | "
                    f"imp {running_imp / max(running_imp_n, 1):.6f}"
                )
            else:
                log_line(
                    f"step {step:06d} | "
                    f"loss {running_loss / running_n:.4f} | "
                    f"pred {running_pred / running_n:.4f}"
                )

            running_loss = 0.0
            running_pred = 0.0
            running_imp = 0.0
            running_n = 0
            running_imp_n = 0

        if val_loader is not None and step % val_every == 0:
            val_metrics = evaluate_synthetic(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=val_batches,
                importance_weight=importance_weight,
            )

            if val_metrics["pred_loss"] < best_pred_loss:
                best_pred_loss = val_metrics["pred_loss"]

                if best_ckpt_path is not None:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "step": step,
                            "best_pred_loss": best_pred_loss,
                            "val_metrics": val_metrics,
                        },
                        best_ckpt_path,
                    )

                log_line(
                    f"[best] step {step:06d} | "
                    f"val_pred_loss {best_pred_loss:.6f} | "
                    f"saved {best_ckpt_path}"
                )

            if loader_use_selector:
                log_line(
                    f"[val] step {step:06d} | "
                    f"loss {val_metrics['loss']:.4f} | "
                    f"pred {val_metrics['pred_loss']:.4f} | "
                    f"imp {val_metrics['importance_loss']:.6f} | "
                    f"acc {val_metrics.get('acc', float('nan')):.4f} | "
                    f"bal_acc {val_metrics.get('balanced_acc', float('nan')):.4f} | "
                    f"f1 {val_metrics.get('macro_f1', float('nan')):.4f} | "
                    f"auc {val_metrics.get('roc_auc', float('nan')):.4f} | "
                    f"imp_corr {val_metrics.get('importance_pearson', float('nan')):.4f}"
                )
            else:
                log_line(
                    f"[val] step {step:06d} | "
                    f"loss {val_metrics['loss']:.4f} | "
                    f"pred {val_metrics['pred_loss']:.4f} | "
                    f"acc {val_metrics.get('acc', float('nan')):.4f} | "
                    f"bal_acc {val_metrics.get('balanced_acc', float('nan')):.4f} | "
                    f"f1 {val_metrics.get('macro_f1', float('nan')):.4f} | "
                    f"auc {val_metrics.get('roc_auc', float('nan')):.4f}"
                )

            model.train()


device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from functools import partial

train_dataset = SyntheticTaskDataset(
    length=100000,
    task_factory=MixedLinearTask,
    task_kind="regression",
    base_seed=0,
    task_kwargs=dict(
        n_min=256,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_categorical=0.3,
        max_cardinality=5,
        p_active=0.85,
        p_missing=0.1,
        noise_level=0.1,
        device=torch.device("cpu"),
    ),
)

val_dataset = SyntheticTaskDataset(
    length=10000,
    task_factory=MixedLinearTask,
    task_kind="regression",
    base_seed=100000,
    task_kwargs=dict(
        n_min=256,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_categorical=0.3,
        max_cardinality=5,
        p_active=0.85,
        p_missing=0.1,
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
    collate_fn=partial(collate_tasks, use_selector=False),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=partial(collate_tasks, use_selector=False),
)

model = TabularPFNModel(
    k=72,
    m=128,
    n_heads=6,
    depth=16,
    max_cardinality=5,
    task_kind="regression",
    num_y_buckets=100
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2,
)

train_synthetic(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    steps=50000,
    importance_weight=None,
    grad_clip=1.0,
    log_every=50,
    val_loader=val_loader,
    val_every=500,
    val_batches=50,
    save_path="/dss/dsshome1/07/ra58bim2/pfn_exp/outputs/large_synth_reg_pred_only_log.txt",
    best_ckpt_path = "/dss/dsshome1/07/ra58bim2/pfn_exp/outputs/large_synth_reg_pred_only_best_ckpt.pt"
)

print("=============================================")

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
    m=128,
    n_heads=6,
    depth=16,
    max_cardinality=5,
    task_kind="regression",
    num_y_buckets=100
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2,
)


train_synthetic(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    steps=50000,
    importance_weight=100,
    grad_clip=1.0,
    log_every=50,
    val_loader=val_loader,
    val_every=500,
    val_batches=50,
    save_path="/dss/dsshome1/07/ra58bim2/pfn_exp/outputs/large_synth_reg_pred_imp_log.txt",
    best_ckpt_path = "/dss/dsshome1/07/ra58bim2/pfn_exp/outputs/large_synth_reg_pred_imp_best_ckpt.pt"
)
 
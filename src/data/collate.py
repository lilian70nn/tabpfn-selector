from dataclasses import dataclass
import torch
from .helper import build_cell_mask

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
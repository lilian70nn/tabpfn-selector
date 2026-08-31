# src/data/openml_collate.py

import torch
import pandas as pd
import openml
from sklearn.model_selection import train_test_split

from src.data.collate import TaskBatch


CONTINUOUS = 0
CATEGORICAL = 1
MAX_CARDINALITY = 20
TEST_FRAC = 0.2
RANDOM_STATE = 0


def _encode_cat_train_test(s_train, s_test, max_cardinality=MAX_CARDINALITY):
    s_train = s_train.astype("object")
    s_test = s_test.astype("object")

    valid = s_train.dropna().astype(str)

    if len(valid) == 0:
        return (
            torch.full((len(s_train),), torch.nan),
            torch.full((len(s_test),), torch.nan),
            2,
        )

    counts = valid.value_counts()

    # 防止 categorical id 超过 encoder embedding 的 max_cardinality
    if len(counts) > max_cardinality:
        kept = list(counts.index[: max_cardinality - 1])
        other_id = len(kept)
        mapping = {c: i for i, c in enumerate(kept)}
        K = max_cardinality

        def enc(v):
            if pd.isna(v):
                return float("nan")
            return float(mapping.get(str(v), other_id))

    else:
        kept = list(counts.index)
        mapping = {c: i for i, c in enumerate(kept)}
        K = max(2, len(mapping))

        def enc(v):
            if pd.isna(v):
                return float("nan")
            return float(mapping.get(str(v), float("nan")))

    xtr = torch.tensor([enc(v) for v in s_train], dtype=torch.float32)
    xte = torch.tensor([enc(v) for v in s_test], dtype=torch.float32)

    return xtr, xte, K


def _encode_cont_train_test(s_train, s_test):
    xtr = pd.to_numeric(s_train, errors="coerce").astype("float32")
    xte = pd.to_numeric(s_test, errors="coerce").astype("float32")

    return (
        torch.tensor(xtr.to_numpy(), dtype=torch.float32),
        torch.tensor(xte.to_numpy(), dtype=torch.float32),
    )


def _x_mean_std(X_train, feature_type):
    d = X_train.shape[1]

    x_mean = torch.zeros(d, dtype=torch.float32)
    x_std = torch.ones(d, dtype=torch.float32)

    for j in range(d):
        if int(feature_type[j].item()) == CATEGORICAL:
            continue

        col = X_train[:, j]
        mask = torch.isfinite(col)

        if bool(mask.any()):
            vals = col[mask]
            x_mean[j] = vals.mean()
            x_std[j] = vals.std(unbiased=False).clamp_min(1e-6)

    return x_mean, x_std


def collate_openml_task(items):
    """
    DataLoader input:
        list(OPENML_DATASETS.items())

    DataLoader must use:
        batch_size=1

    items example:
        [("adult", 1590)]

    Returns:
        TaskBatch with B=1
    """

    assert len(items) == 1, "Use DataLoader(..., batch_size=1) for OpenML eval."

    name, openml_id = items[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = openml.datasets.get_dataset(int(openml_id))

    X_df, y_raw, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )

    task_type = y_raw.dtype
    

    X_df = X_df.reset_index(drop=True)
    y_raw = pd.Series(y_raw).reset_index(drop=True)

    assert len(categorical_indicator) == X_df.shape[1], (
        len(categorical_indicator),
        X_df.shape[1],
    )

    # drop missing target rows
    keep = ~y_raw.isna()
    X_df = X_df.loc[keep].reset_index(drop=True)
    y_raw = y_raw.loc[keep].reset_index(drop=True)

    # classification only for now
    class_counts = y_raw.astype(str).value_counts()
    stratify = y_raw.astype(str) if bool((class_counts >= 2).all()) else None

    X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
        X_df,
        y_raw,
        test_size=TEST_FRAC,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    X_train_df = X_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)
    y_train_raw = y_train_raw.reset_index(drop=True)
    y_test_raw = y_test_raw.reset_index(drop=True)

    Xtr_cols = []
    Xte_cols = []
    feature_type = []
    cardinality = []

    for j, col in enumerate(X_df.columns):
        s_train = X_train_df[col]
        s_test = X_test_df[col]

        is_cat = bool(categorical_indicator[j])

        if is_cat:
            xtr, xte, K = _encode_cat_train_test(s_train, s_test)
            Xtr_cols.append(xtr)
            Xte_cols.append(xte)
            feature_type.append(CATEGORICAL)
            cardinality.append(K)
        else:
            xtr, xte = _encode_cont_train_test(s_train, s_test)
            Xtr_cols.append(xtr)
            Xte_cols.append(xte)
            feature_type.append(CONTINUOUS)
            cardinality.append(0)

    X_train = torch.stack(Xtr_cols, dim=1).to(device)
    X_test = torch.stack(Xte_cols, dim=1).to(device)

    feature_type = torch.tensor(feature_type, dtype=torch.long, device=device)
    cardinality = torch.tensor(cardinality, dtype=torch.long, device=device)

    # y classification encoding
    classes = y_train_raw.dropna().astype(str).unique().tolist()
    y_mapping = {c: i for i, c in enumerate(classes)}

    y_train = torch.tensor(
        [y_mapping[str(v)] for v in y_train_raw],
        dtype=torch.long,
        device=device,
    )

    y_test = torch.tensor(
        [y_mapping.get(str(v), -1) for v in y_test_raw],
        dtype=torch.long,
        device=device,
    )

    if bool((y_test < 0).any()):
        raise ValueError(f"{name}: test set contains unseen class labels.")

    n_classes = torch.tensor(
        [len(y_mapping)],
        dtype=torch.long,
        device=device,
    )

    n_train, d = X_train.shape
    n_test = X_test.shape[0]

    x_mean, x_std = _x_mean_std(X_train.cpu(), feature_type.cpu())
    x_mean = x_mean.to(device)[None, :]
    x_std = x_std.to(device)[None, :]

    # add batch dim
    X_train = X_train[None, :, :]
    X_test = X_test[None, :, :]
    y_train = y_train[None, :]
    y_test = y_test[None, :]
    feature_type = feature_type[None, :]
    cardinality = cardinality[None, :]

    Ntr_max = n_train
    Nte_max = n_test
    d_max = d

    N = Ntr_max + 1 + Nte_max
    F = d_max + 1

    selector_idx = Ntr_max
    test_start = Ntr_max + 1
    y_slot = d_max

    cell_mask = torch.zeros(1, N, F, dtype=torch.bool, device=device)

    cell_mask[:, :Ntr_max, :d_max] = True
    cell_mask[:, selector_idx, :d_max] = True
    cell_mask[:, test_start:, :d_max] = True
    cell_mask[:, :Ntr_max, y_slot] = True
    cell_mask[:, test_start:, y_slot] = True

    # dummy real-data selector labels
    is_active = torch.zeros(1, d, device=device)
    feature_strength = torch.zeros(1, d, device=device)
    importance_ratio = torch.ones(1, d, device=device) / d

    return TaskBatch(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        Ntr_max=Ntr_max,
        Nte_max=Nte_max,
        d_max=d_max,
        n_train=torch.tensor([n_train], device=device),
        n_test=torch.tensor([n_test], device=device),
        d_emb=torch.tensor([d], device=device),
        feature_type=feature_type,
        cardinality=cardinality,
        is_active=is_active,
        importance_ratio=importance_ratio,
        feature_strength=feature_strength,
        cell_mask=cell_mask,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=None,
        y_std=None,
        n_classes=n_classes,
        use_selector=True,
    )
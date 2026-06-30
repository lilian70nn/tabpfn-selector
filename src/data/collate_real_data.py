import torch
import pandas as pd
import openml
from sklearn.model_selection import train_test_split

from src.data.collate import TaskBatch
from src.data.collate import build_cell_mask

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif


TEST_FRAC = 0.1
RANDOM_STATE = 0

def _encode_cat_from_train(s_train, s_test):
    s_train = s_train.astype("object")
    s_test = s_test.astype("object")

    cats = s_train.dropna().astype(str).unique().tolist()
    mapping = {c: i for i, c in enumerate(cats)}

    K = max(2, len(mapping))

    def enc(v):
        if pd.isna(v):
            return float("nan")
        return float(mapping.get(str(v), float("nan")))

    x_train = torch.tensor(
        [enc(v) for v in s_train],
        dtype=torch.float32,
    )

    x_test = torch.tensor(
        [enc(v) for v in s_test],
        dtype=torch.float32,
    )

    return x_train, x_test, K


def _encode_cont_train_test(s_train, s_test):
    xtr = pd.to_numeric(s_train, errors="coerce").astype("float32")
    xte = pd.to_numeric(s_test, errors="coerce").astype("float32")

    return (
        torch.tensor(xtr.to_numpy(), dtype=torch.float32),
        torch.tensor(xte.to_numpy(), dtype=torch.float32),
    )


def collate_openml_task(items,use_selector=True,
                        classification=True,
                        shuffle_features=True,
                        feature_seed=0,
                        compute_reference_importance=True,
                        reference_seed=0,):
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

    X_df, y_raw, x_categorical_indicator, feature_names = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )
    X_df = X_df.reset_index(drop=True)
    y_raw = pd.Series(y_raw).reset_index(drop=True)

    assert len(x_categorical_indicator) == X_df.shape[1], (
        len(x_categorical_indicator),
        X_df.shape[1],
    )

    keep = ~y_raw.isna()
    X_df = X_df.loc[keep].reset_index(drop=True)
    y_raw = y_raw.loc[keep].reset_index(drop=True)

    perm = torch.randperm(len(X_df), generator=torch.Generator().manual_seed(RANDOM_STATE))
    idx = perm[:2000].numpy()

    X_df = X_df.iloc[idx].reset_index(drop=True)
    y_raw = y_raw.iloc[idx].reset_index(drop=True)   

    if classification or isinstance(y_raw.dtype, pd.CategoricalDtype) or y_raw.dtype == "object" or y_raw.dtype.name == "category":
        y_cat = y_raw.astype("category")
        n_classes = torch.tensor(
            [len(y_cat.cat.categories)],
            dtype=torch.long,
            device=device,

        )
        y_ids = torch.tensor(
            y_cat.cat.codes.to_numpy(),
            dtype=torch.long,
            device=device,
        )
        stratify = y_ids.cpu().numpy()

        y_mean = None
        y_std = None

    else:
        n_classes = None
        stratify = None

        y_ids = torch.tensor(
            pd.to_numeric(y_raw).astype("float32").to_numpy(),
            dtype=torch.float32,
            device=device,
        )

        y_mean = torch.mean(y_ids).view(1)
        y_std = torch.std(y_ids, unbiased=False).clamp_min(1e-6).view(1)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y_ids.cpu(),
        test_size=TEST_FRAC,
        random_state=RANDOM_STATE,
        stratify = stratify,
    )
    x_mean = torch.zeros((X_train_df.shape[1],), dtype=torch.float32, device=device)
    x_std = torch.ones((X_train_df.shape[1],), dtype=torch.float32, device=device)

    if n_classes is not None:
        y_train = torch.as_tensor(y_train, device=device, dtype=torch.long)
        y_test = torch.as_tensor(y_test, device=device, dtype=torch.long)
    else:
        y_train = torch.as_tensor(y_train, device=device, dtype=torch.float32)
        y_test = torch.as_tensor(y_test, device=device, dtype=torch.float32)

    Xtr_cols = []
    Xte_cols = []
    feature_type = []
    cardinality = []

    for j, col in enumerate(X_df.columns):
        s_train = X_train_df[col]
        s_test = X_test_df[col]

        if x_categorical_indicator[j]:
            x_train, x_test, K = _encode_cat_from_train(s_train, s_test)
            Xtr_cols.append(x_train)
            Xte_cols.append(x_test)
            feature_type.append(1)
            cardinality.append(K)
        else:
            xtr, xte = _encode_cont_train_test(s_train, s_test)
            Xtr_cols.append(xtr)
            Xte_cols.append(xte)
            feature_type.append(0)
            cardinality.append(0)

            mask = torch.isfinite(xtr)
            if bool(mask.any()):
                vals = xtr[mask]
                x_mean[j] = vals.mean()
                x_std[j] = vals.std(unbiased=False).clamp_min(1e-6)

        
    X_train = torch.stack(Xtr_cols, dim=1).to(device)
    X_test = torch.stack(Xte_cols, dim=1).to(device)

    feature_type = torch.tensor(feature_type, dtype=torch.long, device=device)
    cardinality = torch.tensor(cardinality, dtype=torch.long, device=device)

    n_train, d = X_train.shape
    n_test = X_test.shape[0]

    if shuffle_features:
        feat_gen = torch.Generator().manual_seed(int(feature_seed))
        feature_perm = torch.randperm(d, generator=feat_gen).to(device)
    else:
        feature_perm = torch.arange(d, device=device)

    X_train = X_train[:, feature_perm]
    X_test = X_test[:, feature_perm]
    feature_type = feature_type[feature_perm]
    cardinality = cardinality[feature_perm]
    x_mean = x_mean[feature_perm]
    x_std = x_std[feature_perm]

    reference_importance_mi = torch.zeros(d, dtype=torch.float32, device=device)
    reference_importance_rf = torch.zeros(d, dtype=torch.float32, device=device)

    if compute_reference_importance:
        try:
            X_ref = X_train.detach().cpu().numpy().copy()
            y_ref = y_train.detach().cpu().numpy()

            for j in range(d):
                col = X_ref[:, j]
                ok = np.isfinite(col)

                if not ok.any():
                    X_ref[:, j] = 0.0
                    continue

                if int(feature_type[j].item()) == 1:
                    vals = col[ok].astype(np.int64)
                    mode = np.bincount(vals).argmax()
                    col[~ok] = float(mode)
                else:
                    col[~ok] = float(col[ok].mean())

                X_ref[:, j] = col

            discrete_features = feature_type.detach().cpu().numpy().astype(bool)

            ref_imp_np = mutual_info_classif(
                X_ref,
                y_ref,
                discrete_features=discrete_features,
                random_state=int(reference_seed),
            ).astype("float32")

            ref_imp_np = np.maximum(ref_imp_np, 0.0)
            ref_imp_np = ref_imp_np / (ref_imp_np.sum() + 1e-12)

            reference_importance_mi = torch.tensor(
                ref_imp_np,
                dtype=torch.float32,
                device=device,
            )

        except Exception as e:
            print(f"[MI reference failed] {name}: {repr(e)}")
            reference_importance_mi = torch.zeros(d, dtype=torch.float32, device=device)

    if compute_reference_importance:
        try:
            X_ref = X_train.detach().cpu().numpy().copy()
            y_ref = y_train.detach().cpu().numpy()

            col_mean = np.nanmean(X_ref, axis=0)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)

            inds = np.where(~np.isfinite(X_ref))
            X_ref[inds] = np.take(col_mean, inds[1])

            ref_model = RandomForestClassifier(
                n_estimators=200,
                random_state=int(reference_seed),
                n_jobs=-1,
                class_weight="balanced_subsample",
            )

            ref_model.fit(X_ref, y_ref)

            X_ref_test = X_test.detach().cpu().numpy().copy()
            y_ref_test = y_test.detach().cpu().numpy()

            inds = np.where(~np.isfinite(X_ref_test))
            X_ref_test[inds] = np.take(col_mean, inds[1])

            perm_result = permutation_importance(
                ref_model,
                X_ref_test,
                y_ref_test,
                scoring="balanced_accuracy",
                n_repeats=20,
                random_state=int(reference_seed),
                n_jobs=-1,
            )

            ref_imp_np = perm_result.importances_mean.astype("float32")
            ref_imp_np = np.maximum(ref_imp_np, 0.0)
            ref_imp_np = ref_imp_np / (ref_imp_np.sum() + 1e-12)

            reference_importance_rf = torch.tensor(
                ref_imp_np,
                dtype=torch.float32,
                device=device,
            )

        except Exception as e:
            print(f"[RF permutation reference failed] {name}: {repr(e)}")
            reference_importance_rf = torch.zeros(d, dtype=torch.float32, device=device)

    reference_importance_mi_original = torch.empty_like(reference_importance_mi)
    reference_importance_mi_original[feature_perm] = reference_importance_mi

    reference_importance_rf_original = torch.empty_like(reference_importance_rf)
    reference_importance_rf_original[feature_perm] = reference_importance_rf

    X_train = X_train[None, :, :]
    X_test = X_test[None, :, :]
    y_train = y_train[None, :]
    y_test = y_test[None, :]
    feature_type = feature_type[None, :]
    cardinality = cardinality[None, :]
    feature_perm = feature_perm[None, :]
    reference_importance_mi_original = reference_importance_mi_original[None, :]
    reference_importance_rf_original = reference_importance_rf_original[None, :]



    cell_mask = build_cell_mask(
            B=1,
            Ntr_max=n_train,
            Nte_max=n_test,
            d_max=d,
            n_train=n_train,
            n_test=n_test,
            d_emb=d,
            device=device,
            use_selector=use_selector,
        )

    return TaskBatch(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Ntr_max=n_train,
        Nte_max=n_test,
        d_max=d,
        n_train=torch.tensor([n_train], device=device),
        n_test=torch.tensor([n_test], device=device),
        d_emb=torch.tensor([d], device=device),
        feature_type=feature_type,
        cardinality=cardinality,
        is_active=torch.zeros((1, d), dtype=torch.float32, device=device),
        importance_ratio=torch.ones((1, d), dtype=torch.float32, device=device) / d,
        feature_strength=torch.zeros((1, d), dtype=torch.float32, device=device),
        cell_mask=cell_mask,
        x_mean=x_mean[None, :],
        x_std=x_std[None, :],
        y_mean=y_mean,
        y_std=y_std,
        n_classes=n_classes,
        use_selector=use_selector,
        feature_perm=feature_perm,
        reference_importance_mi=reference_importance_mi_original,
        reference_importance_rf=reference_importance_rf_original,
    )


import torch
import numpy as np
import pandas as pd
import openml


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .collate import TaskBatch
from .collate import build_cell_mask


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

    x_train = torch.tensor([enc(v) for v in s_train], dtype=torch.float32)
    x_test = torch.tensor([enc(v) for v in s_test], dtype=torch.float32,)

    return x_train, x_test, K


def _encode_high_card_cat_from_train(s_train, s_test):
    s_train = s_train.astype("object")
    s_test = s_test.astype("object")

    cats = s_train.dropna().astype(str).unique().tolist()
    mapping = {c: i for i, c in enumerate(cats)}

    def enc(v):
        if pd.isna(v):
            return float("nan")
        return float(mapping.get(str(v), float("nan")))

    x_train = torch.tensor([enc(v) for v in s_train], dtype=torch.float32)
    x_test = torch.tensor([enc(v) for v in s_test], dtype=torch.float32)
    return x_train, x_test


def _encode_cont_train_test(s_train, s_test):
    xtr = pd.to_numeric(s_train, errors="coerce").astype("float32")
    xte = pd.to_numeric(s_test, errors="coerce").astype("float32")

    return (
        torch.tensor(xtr.to_numpy(), dtype=torch.float32),
        torch.tensor(xte.to_numpy(), dtype=torch.float32),
    )


def collate_openml_task(
        items,
        n_repeats=30,
        use_selector=True,
        classification=True,
        shuffle_features=True,
        feature_seed=0,
        split_seed=0,
        compute_reference_importance=True,
        reference_seed=0,
        selected_features=None
):
    

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
    assert int(n_repeats) >= 1

    name, openml_id = items[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = openml.datasets.get_dataset(int(openml_id))

    X_df, y_raw, x_categorical_indicator, _ = dataset.get_data(
        target=dataset.default_target_attribute,
        dataset_format="dataframe",
    )
    X_df = X_df.reset_index(drop=True)
    y_raw = pd.Series(y_raw).reset_index(drop=True)
    x_categorical_indicator = np.asarray(x_categorical_indicator, dtype=bool)

    assert len(x_categorical_indicator) == X_df.shape[1], (len(x_categorical_indicator), X_df.shape[1])

    keep = ~y_raw.isna()
    X_df = X_df.loc[keep].reset_index(drop=True)
    y_raw = y_raw.loc[keep].reset_index(drop=True)

    perm = torch.randperm(len(X_df), generator=torch.Generator().manual_seed(RANDOM_STATE))
    idx = perm[:2000].numpy()

    X_df = X_df.iloc[idx].reset_index(drop=True)
    y_raw = y_raw.iloc[idx].reset_index(drop=True)   

    if selected_features is not None:
        assert len(selected_features) == n_repeats
        selected_features = [np.asarray(x, dtype=int) for x in selected_features]

    if classification:
        y_cat = y_raw.astype("category")
        n_classes_value = len(y_cat.cat.categories)
        n_classes = torch.full((n_repeats,), n_classes_value, dtype=torch.long, device=device)
        y_ids = torch.tensor(y_cat.cat.codes.to_numpy(), dtype=torch.long, device=device)
        stratify = y_ids.cpu().numpy()

        y_mean = None
        y_std = None

    else:
        n_classes_value = None
        n_classes = None
        stratify = None

        y_ids = torch.tensor(pd.to_numeric(y_raw).astype("float32").to_numpy(), dtype=torch.float32, device=device)
        y_mean = torch.zeros(n_repeats, dtype=torch.float32, device=device)
        y_std = torch.ones(n_repeats, dtype=torch.float32, device=device)

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []
    feature_type_list, cardinality_list = [], []
    x_mean_list, x_std_list = [], []
    feature_perm_list = []
    reference_mi_list, reference_rf_list, reference_linear_list = [], [], []

    for rep in range(int(n_repeats)):

        cur_split_seed = int(split_seed) + rep
        cur_feature_seed = int(feature_seed) + rep

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X_df,
            y_ids.cpu(),
            test_size=TEST_FRAC,
            random_state=cur_split_seed,
            stratify = stratify,
        )

        if selected_features is not None:
            selected_rep = selected_features[rep]
            X_train_df = X_train_df.iloc[:, selected_rep]
            X_test_df = X_test_df.iloc[:, selected_rep]
            cat_indicator_rep = x_categorical_indicator[selected_rep]

        else:
            cat_indicator_rep = x_categorical_indicator

        
        x_mean = torch.zeros((X_train_df.shape[1],), dtype=torch.float32, device=device)
        x_std = torch.ones((X_train_df.shape[1],), dtype=torch.float32, device=device)


        if n_classes is not None:
            y_train = torch.as_tensor(y_train, device=device, dtype=torch.long)
            y_test = torch.as_tensor(y_test, device=device, dtype=torch.long)

        else:
            y_train = torch.as_tensor(y_train, device=device, dtype=torch.float32)
            y_test = torch.as_tensor(y_test, device=device, dtype=torch.float32)

            y_mean[rep] = y_train.mean()
            y_std[rep] = y_train.std(unbiased=False).clamp_min(1e-6)

        Xtr_cols = []
        Xte_cols = []
        feature_type = []
        cardinality = []

        for j, col in enumerate(X_train_df.columns):
            s_train = X_train_df[col]
            s_test = X_test_df[col]

            distinct = int(s_train.nunique(dropna=True))
            is_object = s_train.dtype == "object" or isinstance(s_train.dtype, pd.CategoricalDtype)
            is_cat = bool(cat_indicator_rep[j])

            if (not is_cat) and is_object:
                is_cat = True

            if is_cat and distinct <= 6:
                xtr, xte, K = _encode_cat_from_train(s_train, s_test)
                Xtr_cols.append(xtr)
                Xte_cols.append(xte)
                feature_type.append(1)
                cardinality.append(K)

            elif is_cat and distinct > 6:
                xtr, xte = _encode_high_card_cat_from_train(s_train, s_test)
                Xtr_cols.append(xtr)
                Xte_cols.append(xte)
                feature_type.append(0)
                cardinality.append(0)

                mask = torch.isfinite(xtr)
                if bool(mask.any()):
                    vals = xtr[mask]
                    x_mean[j] = vals.mean()
                    x_std[j] = vals.std(unbiased=False).clamp_min(1e-6)


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


        d = X_train.shape[1]

        if shuffle_features:
            feat_gen = torch.Generator().manual_seed(cur_feature_seed)
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
        reference_importance_linear_perm = torch.zeros(d, dtype=torch.float32, device=device)

        if compute_reference_importance:
            try:
                X_ref = X_train.detach().cpu().numpy().copy()
                y_ref = y_train.detach().cpu().numpy().reshape(-1)

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

                if n_classes is not None:
                    ref_imp_np = mutual_info_classif(
                        X_ref,
                        y_ref,
                        discrete_features=discrete_features,
                        random_state=int(reference_seed),
                    ).astype("float32")
                else:
                    ref_imp_np = mutual_info_regression(
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
                y_ref = y_train.detach().cpu().numpy().reshape(-1)

                col_mean = np.nanmean(X_ref, axis=0)
                col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)

                inds = np.where(~np.isfinite(X_ref))
                X_ref[inds] = np.take(col_mean, inds[1])

                if n_classes is not None:
                    ref_model = RandomForestClassifier(
                        n_estimators=200,
                        random_state=int(reference_seed),
                        n_jobs=-1,
                        class_weight="balanced_subsample",
                    )
                    scoring = "balanced_accuracy"
                else:
                    ref_model = RandomForestRegressor(
                        n_estimators=200,
                        random_state=int(reference_seed),
                        n_jobs=-1,
                    )
                    scoring = "r2"

                ref_model.fit(X_ref, y_ref)

                X_ref_test = X_test.detach().cpu().numpy().copy()
                y_ref_test = y_test.detach().cpu().numpy().reshape(-1)

                inds = np.where(~np.isfinite(X_ref_test))
                X_ref_test[inds] = np.take(col_mean, inds[1])

                perm_result = permutation_importance(
                    ref_model, X_ref_test, y_ref_test, scoring=scoring,
                    n_repeats=20, random_state=int(reference_seed), n_jobs=-1,
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

        if compute_reference_importance:
            try:
                X_ref = X_train.detach().cpu().numpy().copy()
                y_ref = y_train.detach().cpu().numpy().reshape(-1)

                col_mean = np.nanmean(X_ref, axis=0)
                col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)

                inds = np.where(~np.isfinite(X_ref))
                X_ref[inds] = np.take(col_mean, inds[1])

                X_ref_test = X_test.detach().cpu().numpy().copy()
                y_ref_test = y_test.detach().cpu().numpy().reshape(-1)

                inds = np.where(~np.isfinite(X_ref_test))
                X_ref_test[inds] = np.take(col_mean, inds[1])

                if n_classes is not None:
                    ref_model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=int(reference_seed),
                            solver="lbfgs",
                        ),
                    )
                    scoring = "roc_auc" if n_classes_value == 2 else "balanced_accuracy"
                else:
                    ref_model = make_pipeline(
                        StandardScaler(),
                        Ridge(),
                    )
                    scoring = "r2"

                ref_model.fit(X_ref, y_ref)

                perm_result = permutation_importance(
                    ref_model, X_ref_test, y_ref_test, scoring=scoring,
                    n_repeats=30, random_state=int(reference_seed), n_jobs=-1,
                )

                ref_imp_np = perm_result.importances_mean.astype("float32")
                ref_imp_np = np.maximum(ref_imp_np, 0.0)
                ref_imp_np = ref_imp_np / (ref_imp_np.sum() + 1e-12)

                reference_importance_linear_perm = torch.tensor(
                    ref_imp_np, dtype=torch.float32, device=device
                )

            except Exception as e:
                print(f"[Linear permutation reference failed] {name}: {repr(e)}")
                reference_importance_linear_perm = torch.zeros(d, dtype=torch.float32, device=device)

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        feature_type_list.append(feature_type)
        cardinality_list.append(cardinality)
        x_mean_list.append(x_mean)
        x_std_list.append(x_std)
        feature_perm_list.append(feature_perm)
        reference_mi_list.append(reference_importance_mi)
        reference_rf_list.append(reference_importance_rf)
        reference_linear_list.append(reference_importance_linear_perm)

    X_train = torch.stack(X_train_list, dim=0)
    X_test = torch.stack(X_test_list, dim=0)
    y_train = torch.stack(y_train_list, dim=0)
    y_test = torch.stack(y_test_list, dim=0)
    feature_type = torch.stack(feature_type_list, dim=0)
    cardinality = torch.stack(cardinality_list, dim=0)
    x_mean = torch.stack(x_mean_list, dim=0)
    x_std = torch.stack(x_std_list, dim=0)
    feature_perm = torch.stack(feature_perm_list, dim=0)
    reference_importance_mi = torch.stack(reference_mi_list, dim=0)
    reference_importance_rf = torch.stack(reference_rf_list, dim=0)
    reference_importance_linear_perm = torch.stack(reference_linear_list, dim=0)

    n_train = torch.full((n_repeats,), X_train.shape[1], dtype=torch.long, device=device)
    n_test = torch.full((n_repeats,), X_test.shape[1], dtype=torch.long, device=device)
    d_emb = torch.full((n_repeats,), X_train.shape[2], dtype=torch.long, device=device)

    Ntr_max = X_train.shape[1]
    Nte_max = X_test.shape[1]
    d_max = X_train.shape[2]


    cell_mask = build_cell_mask(
            B=n_repeats,
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
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Ntr_max=Ntr_max,
        Nte_max=Nte_max,
        d_max=d_max,
        n_train=n_train,
        n_test=n_test,
        d_emb=d_emb,
        feature_type=feature_type,
        cardinality=cardinality,
        feature_importance=torch.zeros((n_repeats, d), dtype=torch.float32, device=device),
        cell_mask=cell_mask,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        n_classes=n_classes,
        use_selector=use_selector,
        feature_perm=feature_perm,
        reference_importance_mi=reference_importance_mi,
        reference_importance_rf=reference_importance_rf,
        reference_importance_linear_perm=reference_importance_linear_perm,
    )


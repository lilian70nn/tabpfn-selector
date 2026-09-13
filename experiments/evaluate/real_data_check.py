import torch
import numpy as np
import pandas as pd
import openml
from ..config import CLS_DATASETS

MAX_ROWS = 2000
RANDOM_STATE = 0


def load_like_collate(openml_id, max_rows=2000, random_state=0):
    dataset = openml.datasets.get_dataset(openml_id)

    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute
    )

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    categorical_indicator = np.asarray(categorical_indicator, dtype=bool)
    feature_names = list(feature_names)

    assert len(categorical_indicator) == X.shape[1], (len(categorical_indicator), X.shape[1])

    valid_target = ~y.isna()
    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    if len(X) > max_rows:
        g = torch.Generator().manual_seed(random_state)
        idx = torch.randperm(len(X), generator=g)[:max_rows].numpy()
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    return X, y, categorical_indicator, feature_names


def check_dataset(name, openml_id):
    X, y, categorical_indicator, feature_names = load_like_collate(openml_id, MAX_ROWS, RANDOM_STATE)

    print("\n" + "=" * 140)
    print(f"{name} | OpenML ID={openml_id} | rows={len(X)} | features={X.shape[1]}")
    print("=" * 140)
    print(f"{'idx':>3}  {'feature':<35} {'OpenML type':<12} {'dtype':<15} {'distinct':>10}  sample values")
    print("-" * 140)

    for j, feature_name in enumerate(feature_names):
        s = X.iloc[:, j]
        feature_type = "categorical" if categorical_indicator[j] else "continuous"
        dtype = str(s.dtype)
        distinct = int(s.nunique(dropna=True))
        samples = s.dropna().unique()[:5].tolist()
        print(f"{j:>3}  {str(feature_name):<35} {feature_type:<12} {dtype:<15} {distinct:>10}  {samples}")

    n_cat = int(categorical_indicator.sum())
    n_con = int((~categorical_indicator).sum())
    high_card = [j for j in range(X.shape[1]) if categorical_indicator[j] and X.iloc[:, j].nunique(dropna=True) > 10]

    print("-" * 140)
    print(f"categorical={n_cat} | continuous={n_con} | cat_ratio={n_cat / len(categorical_indicator):.3f}")
    print(f"categorical with distinct > 10: {len(high_card)}")

    if high_card:
        print("high-cardinality categorical features:")
        for j in high_card:
            s = X.iloc[:, j]
            print(f"  [{j}] {feature_names[j]} | dtype={s.dtype} | distinct={s.nunique(dropna=True)}")


def main():
    for name, openml_id in CLS_DATASETS.items():
        try:
            check_dataset(name, openml_id)
        except Exception as e:
            print("\n" + "=" * 140)
            print(f"{name} ({openml_id}) FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
import openml
import pandas as pd

dataset = openml.datasets.get_dataset(1559)

X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute,
    dataset_format="dataframe",
)

print("dataset:", dataset.name)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\n=== X ===")
print(X.head(20))

print("\n=== y ===")
print(y.head(20))

print("\n=== dtypes ===")
print(X.dtypes)

print("\n=== categorical ===")
for name, is_cat in zip(attribute_names, categorical_indicator):
    print(name, is_cat)

print("\n=== unique values ===")
for col in X.columns:
    print(f"\n{col}:")
    print(X[col].value_counts(dropna=False))

print("\n=== target ===")
print(y.value_counts(dropna=False))

print("\n=== missing ===")
print(X.isna().sum())
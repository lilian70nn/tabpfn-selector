import pandas as pd

df = pd.read_csv("target_latent_diagnostics.csv")
df = df.sort_values("test_r2")

for _, row in df.head(10).iterrows():
    print("=" * 100)
    print("task:", row["task"])
    print("test R2:", row["test_r2"])
    print("parents:", row["target_parent_count"])
    print("program:")
    print(row["target_program"])
import csv
from pathlib import Path

RESULTS_DIR = Path("results/evaluation")

PRIORS = [
    "baseline",
    "depth",
    "sparsity",
    "mechanism",
    "noise",
    "missing",
    "features",
]

for prior in PRIORS:
    results = []
    fieldnames = []

    for batch in (1, 2, 3):
        path = RESULTS_DIR / f"extended_eval_{prior}_batch{batch}.csv"

        with open(path, newline="") as f:
            reader = csv.DictReader(f)

            for field in reader.fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)

            rows = list(reader)

            if len(rows) != 8:
                raise ValueError(
                    f"{path}: expected 8 rows, found {len(rows)}"
                )

            results.extend(rows)

    output_path = RESULTS_DIR / f"extended_eval_{prior}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {output_path} ({len(results)} rows)")
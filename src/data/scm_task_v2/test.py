import random
from collections import Counter

from src.data.scm_task_v2.task import SCMTask


base_seed = 42
N = 5000

stats = Counter()

for idx in range(N):
    rng = random.Random(base_seed + idx)

    dag_seed = rng.randrange(2**31)
    x_seed = rng.randrange(2**31)
    aleatoric_seed = rng.randrange(2**31)

    task = SCMTask(
        num_classes=2,
        dag_seed=dag_seed,
        x_seed=x_seed,
        aleatoric_seed=aleatoric_seed,
    )

    _, _, _, _, info = task._generate()

    cat_ok = bool(info["categorical_features_ok"])
    target_ok = bool(info["target_ok"])
    importance_ok = bool(info["importance_ok"])

    stats["total"] += 1

    if info["is_valid"]:
        stats["valid"] += 1
    else:
        stats["invalid"] += 1

    if not cat_ok:
        stats["fail_categorical"] += 1
    if not target_ok:
        stats["fail_target"] += 1
    if not importance_ok:
        stats["fail_importance"] += 1

    failures = []
    if not cat_ok:
        failures.append("categorical")
    if not target_ok:
        failures.append("target")
    if not importance_ok:
        failures.append("importance")

    if failures:
        stats["combo_" + "+".join(failures)] += 1


print(stats)

print(f"\nvalid rate: {stats['valid'] / stats['total']:.2%}")
print(f"categorical fail: {stats['fail_categorical'] / stats['total']:.2%}")
print(f"target fail: {stats['fail_target'] / stats['total']:.2%}")
print(f"importance fail: {stats['fail_importance'] / stats['total']:.2%}")

print("\nFailure combinations:")
for key, value in sorted(stats.items()):
    if key.startswith("combo_"):
        print(key, value, f"{value / stats['total']:.2%}")
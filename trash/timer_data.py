import time
import torch
from src.data.scm_task_v10 import WeightedMixedScalarSCMTask

times = []

for i in range(20):
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    task = WeightedMixedScalarSCMTask(
        num_classes=2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    elapsed = t1 - t0
    times.append(elapsed)

    print(f"task {i:02d} | total={elapsed:.4f}s")

print()
print(f"mean   = {sum(times) / len(times):.4f}s")
print(f"min    = {min(times):.4f}s")
print(f"max    = {max(times):.4f}s")
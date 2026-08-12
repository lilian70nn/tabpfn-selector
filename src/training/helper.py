from dataclasses import fields
import torch
from functools import partial

def move_batch_to_device(batch, device):
    kwargs = {}
    for f in fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v):
            kwargs[f.name] = v.to(device, non_blocking=True)
        else:
            kwargs[f.name] = v
    return type(batch)(**kwargs)

# def infer_loader_use_selector(loader):
#     batch = next(iter(loader))
#     assert hasattr(batch, "use_selector"), (
#         "Batch must contain use_selector. "
#         "Set batch.use_selector inside collate_tasks."
#     )
#     return bool(batch.use_selector)
def infer_loader_use_selector(loader):
    collate_fn = loader.collate_fn

    if isinstance(collate_fn, partial):
        if "use_selector" in collate_fn.keywords:
            return bool(collate_fn.keywords["use_selector"])

    raise ValueError(
        "Cannot infer use_selector from loader.collate_fn."
    )
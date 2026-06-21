from dataclasses import fields
import torch

def move_batch_to_device(batch, device):
    kwargs = {}
    for f in fields(batch):
        v = getattr(batch, f.name)
        if torch.is_tensor(v):
            kwargs[f.name] = v.to(device, non_blocking=True)
        else:
            kwargs[f.name] = v
    return type(batch)(**kwargs)

def infer_loader_use_selector(loader):
    batch = next(iter(loader))
    assert hasattr(batch, "use_selector"), (
        "Batch must contain use_selector. "
        "Set batch.use_selector inside collate_tasks."
    )
    return bool(batch.use_selector)
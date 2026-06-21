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
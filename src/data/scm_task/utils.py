import torch

def randn(*shape, generator, device):
    return torch.randn(
        *shape,
        generator=generator,
        device=device,
    )


def rand(*shape, generator, device):
    return torch.rand(
        *shape,
        generator=generator,
        device=device,
    )


def randint(low, high, shape, generator, device):
    return torch.randint(
        low,
        high,
        shape,
        generator=generator,
        device=device,
    )


def standardize(x, dim=0, eps = 1e-6):
    mean = x.mean(dim=dim, keepdim=True).detach()
    std = x.std(dim=dim, unbiased=False, keepdim=True,).clamp_min(eps).detach()
    return (x - mean) / std


def normalize_probs(values, device, expected_len=None, name="probabilities"):
    probs = torch.as_tensor(values, device=device, dtype=torch.float32)
    if expected_len is not None and probs.numel() != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values.")

    if (probs.numel() == 0 or bool((probs < 0).any()) or probs.sum() <= 0):
        raise ValueError(f"Invalid {name}.")
    return probs / probs.sum()

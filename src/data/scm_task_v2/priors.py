import math
import torch

from .utils import rand


def sample_uniform(bounds, generator, device):
    low, high = map(float, bounds)

    if high < low:
        raise ValueError("Uniform bounds require low <= high.")

    u = rand((), generator=generator, device=device)
    value = low + (high - low) * u
    return float(value.item())


def sample_loguniform(bounds, generator, device):
    low, high = map(float, bounds)

    if low <= 0 or high <= 0 or high < low:
        raise ValueError("Log-uniform bounds require 0 < low <= high.")

    u = rand((), generator=generator, device=device)
    log_value = math.log(low)  + (math.log(high) - math.log(low)) * u

    return float(torch.exp(log_value).item())


def sample_dirichlet(concentration, generator, device, expected_len=None):

    concentration = torch.as_tensor(
        concentration,
        device=device,
        dtype=torch.float32,
    )

    if concentration.ndim != 1:
        raise ValueError("Dirichlet concentration must be one-dimensional.")
    if expected_len is not None and concentration.numel() != expected_len:
        raise ValueError(f"Dirichlet concentration must have {expected_len} entries.")
    if bool((concentration <= 0).any()):
        raise ValueError("Dirichlet concentrations must be positive.")

    raw = torch._standard_gamma(concentration, generator=generator)
    probs = raw / raw.sum().clamp_min(1e-12)

    return tuple(float(x) for x in probs.tolist())


def sample_connection_probs(bounds, generator, device, expected_len):
    bounds = tuple(bounds)

    if len(bounds) != expected_len:
        raise ValueError(f"connection_prob_bounds must contain {expected_len} layer bounds.")

    values = []

    for low, high in bounds:
        low = float(low)
        high = float(high)
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError("Connection probability bounds must satisfy 0 <= low <= high <= 1.")

        u = rand((), generator=generator, device=device)
        value = low + (high - low) * u
        values.append(float(value.item()))

    return tuple(values)
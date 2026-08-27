from dataclasses import dataclass
import torch
from .utils import normalize_probs

@dataclass(frozen=True)
class FeatureObservation:
    values: torch.Tensor
    is_categorical: bool
    cardinality: int
    observation_type_id: int
    observation_type_name: str
    quality_score: float
    retention: float
    prototypes: torch.Tensor
    thresholds: torch.Tensor


class ScalarObservationHead:
    """
    The underlying SCM node is always continuous and scalar.
    Observation type determines how that scalar becomes a table column:
    - CONTINUOUS: use the scalar directly;
    - PROTOTYPE: choose K scalar prototypes and assign the nearest category;
    - BINNING: split the scalar using K - 1 thresholds.
    """

    CONTINUOUS = 0
    PROTOTYPE = 1
    BINNING = 2

    NAMES = ("continuous_scalar", "prototype_discretization", "dirichlet_binning")

    def __init__(
        self,
        generator,
        device,
        observation_type_probs=(0.60, 0.20, 0.20),
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        min_samples_per_category=8,
        min_component_weight=0.05,
        observation_noise_scale=0.05,
    ):
        self.device = device
        self.min_samples_per_category = int(min_samples_per_category)
        self.min_component_weight = float(min_component_weight)
        self.observation_noise_scale = float(observation_noise_scale)

        self.observation_type_probs = (
            normalize_probs(
                observation_type_probs,
                device,
                expected_len=3,
                name="observation_type_probs",
            )
        )
        self.cardinalities = tuple(int(k) for k in categorical_cardinalities)
        self.cardinality_probs = (
            normalize_probs(
                categorical_cardinality_probs,
                device,
                expected_len=len(self.cardinalities),
                name="categorical_cardinality_probs"
            )
        )
        self.sampled_type = int(torch.multinomial(self.observation_type_probs, 1, generator=generator).item())

    def _sample_cardinality(self, n, generator):
        feasible = [
            i for i, k in enumerate(self.cardinalities)
            if (k * self.min_samples_per_category <= n) and (k * self.min_component_weight <= 1.0)
        ]

        if not feasible:
            return 0

        feasible_tensor = torch.tensor(feasible, device=self.device, dtype=torch.long)
        probabilities = self.cardinality_probs[feasible_tensor]
        probabilities = probabilities / probabilities.sum()
        position = int(torch.multinomial(probabilities, 1, generator=generator).item())
        return self.cardinalities[feasible[position]]


    def _categorical_retention(self, scalar, labels):
        """
        Fraction of latent variance retained by the categorical observation.
        eta^2 = Var(E[z | category]) / Var(z)
        1.0 means categories preserve essentially all latent variation.
        0.0 means categories contain essentially no information about the latent.
        """
        scalar = scalar.float()
        labels = labels.long()

        total_var = scalar.var(unbiased=False)

        if total_var <= 1e-12:
            return 0.0

        global_mean = scalar.mean()
        between_var = torch.zeros((), device=scalar.device, dtype=scalar.dtype)

        for category in torch.unique(labels):
            mask = labels == category
            if not mask.any():
                continue
            weight = mask.float().mean()
            category_mean = scalar[mask].mean()
            between_var = between_var + weight * (category_mean - global_mean).square()

        retention = between_var / total_var.clamp_min(1e-12)
        return float(retention.clamp(0.0, 1.0).detach().item())

    def _continuous(self, z, generator, name = "continuous_scalar"):
        score = z[:, 0].clone()

        if self.observation_noise_scale > 0:
            noise = torch.randn(
                score.shape,
                generator=generator,
                device=z.device,
                dtype=z.dtype,
            )
            score = score + self.observation_noise_scale * noise

        return FeatureObservation(
            values=score,
            is_categorical=False,
            cardinality=0,
            observation_type_id=self.CONTINUOUS,
            observation_type_name=name,
            quality_score=0.0,
            retention=1.0,
            prototypes=torch.empty(0, 1, device=z.device, dtype=z.dtype),
            thresholds=torch.empty(0, device=z.device, dtype=z.dtype),
        )

    def _select_prototypes(self, scalar, k, generator):
        indices = torch.randperm(
            scalar.shape[0],
            generator=generator,
            device=scalar.device,
        )[:k]
        return scalar[indices]

    def _prototype(self, z, generator, k=None):
        scalar = z[:, 0].clone()
        if k is None:
            k = self._sample_cardinality(scalar.shape[0], generator)
        if k == 0:
            return self._continuous(z, generator, name="continuous_fallback_from_prototype")

        prototypes = self._select_prototypes(scalar, k, generator)
        distances = torch.abs(scalar[:, None] - prototypes[None, :])
        labels = distances.argmin(dim=1).long()
        counts = torch.bincount(labels, minlength=k)
        smallest_fraction = float((counts.float() / counts.sum().clamp_min(1)).min().item())
        retention = self._categorical_retention(scalar, labels)

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=k,
            observation_type_id=self.PROTOTYPE,
            observation_type_name="prototype_discretization",
            quality_score=smallest_fraction,
            retention=retention,
            prototypes=prototypes[:, None],
            thresholds=torch.empty(0, device=z.device, dtype=z.dtype),
        )

    def _dirichlet_binning(self, z, generator, concentration=3.0, k=None):
        scalar = z[:, 0].clone()
        if k is None:
            k = self._sample_cardinality(scalar.shape[0], generator)
        n = scalar.numel()
        k = int(k)

        if k < 2:
            raise ValueError("k must be at least 2.")

        minimum = max(self.min_samples_per_category, int(torch.ceil(torch.tensor(self.min_component_weight * n, device=z.device)).item()))
        if k * minimum > n:
            raise ValueError(f"Cannot create {k} categories with minimum category size {minimum} and n={n}.")

        remaining = n - k * minimum
        alpha = torch.full((k,), float(concentration), device=z.device, dtype=torch.float32)
        raw = torch._standard_gamma(alpha, generator=generator)
        proportions = raw / raw.sum().clamp_min(1e-12)

        extras_float = proportions * remaining
        extras = torch.floor(extras_float).long()
        leftover = remaining - int(extras.sum().item())
        if leftover > 0:
            residual_order = torch.argsort(extras_float - extras.float(), descending=True)
            extras[residual_order[:leftover]] += 1

        counts = extras + minimum
        cumulative = torch.cumsum(counts, dim=0)[:-1]
        sorted_values = torch.sort(scalar).values
        thresholds = 0.5 * (sorted_values[cumulative - 1] + sorted_values[cumulative])
        labels = torch.bucketize(scalar, thresholds).long()

        observed_counts = torch.bincount(labels, minlength=k)
        smallest_fraction = float((observed_counts.float() / observed_counts.sum()).min().item())
        retention = self._categorical_retention(scalar, labels)

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=k,
            observation_type_id=self.BINNING,
            observation_type_name="dirichlet_binning",
            quality_score=smallest_fraction,
            retention=retention,
            prototypes=torch.empty(0, 1, device=z.device, dtype=z.dtype),
            thresholds=thresholds,
        )

    def observe(self, latent, generator):
        z = latent.float()
        if self.sampled_type == self.CONTINUOUS:
            return self._continuous(z, generator)
        if self.sampled_type == self.PROTOTYPE:
            return self._prototype(z, generator)
        return self._dirichlet_binning(z, generator)


    def observe_categorical(self, latent, generator, k):
        z = latent.float()
        categorical_probs = self.observation_type_probs[1:]
        categorical_probs = categorical_probs / categorical_probs.sum()
        method = int(torch.multinomial(categorical_probs, 1, generator=generator).item())
        if method == 0:
            return self._prototype(z, generator, k=k)
        return self._dirichlet_binning(z, generator, k=k)
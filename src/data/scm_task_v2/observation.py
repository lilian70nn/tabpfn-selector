from dataclasses import dataclass
import torch
from .utils import normalize_probs, randint

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

    # def _select_prototypes(self, scalar, k, generator):
    #     indices = torch.randperm(
    #         scalar.shape[0],
    #         generator=generator,
    #         device=scalar.device,
    #     )[:k]
    #     return scalar[indices]

    # def _prototype(self, z, generator, k=None):
    #     scalar = z[:, 0].clone()
    #     if k is None:
    #         k = self._sample_cardinality(scalar.shape[0], generator)
    #     if k == 0:
    #         return self._continuous(z, generator, name="continuous_fallback_from_prototype")

    #     prototypes = self._select_prototypes(scalar, k, generator)
    #     distances = torch.abs(scalar[:, None] - prototypes[None, :])
    #     labels = distances.argmin(dim=1).long()
    #     counts = torch.bincount(labels, minlength=k)
    #     smallest_fraction = float((counts.float() / counts.sum().clamp_min(1)).min().item())
    #     retention = self._categorical_retention(scalar, labels)

    #     return FeatureObservation(
    #         values=labels,
    #         is_categorical=True,
    #         cardinality=k,
    #         observation_type_id=self.PROTOTYPE,
    #         observation_type_name="prototype_discretization",
    #         quality_score=smallest_fraction,
    #         retention=retention,
    #         prototypes=prototypes[:, None],
    #         thresholds=torch.empty(0, device=z.device, dtype=z.dtype),
    #     )

    def _select_prototypes(self, scalar, k, generator):
        n = scalar.shape[0]
        first_idx = int(randint(0, n, (1,), generator=generator, device=scalar.device).item())
        selected_indices = [first_idx]
        prototypes = [scalar[first_idx]]

        for _ in range(1, k):
            current = torch.stack(prototypes)
            distances = torch.abs(scalar[:, None] - current[None, :])
            min_distances = distances.min(dim=1).values
            weights = min_distances.square()
            weights[selected_indices] = 0.0
            if bool(weights.sum() <= 1e-12):
                return None
            next_idx = int(torch.multinomial(weights, 1, generator=generator).item())
            selected_indices.append(next_idx)
            prototypes.append(scalar[next_idx])

        return torch.stack(prototypes)


    def _prototype(self, z, generator, k=None, max_attempts=5):
        scalar = z[:, 0].clone()
        n = scalar.shape[0]

        if k is None:
            k = self._sample_cardinality(scalar.shape[0], generator)
        if k == 0:
            return self._continuous(z, generator, name="continuous_fallback_from_prototype")

        minimum = max(self.min_samples_per_category, int(torch.ceil(torch.tensor(self.min_component_weight * n, device=z.device)).item()))
        for _ in range(max_attempts):
            prototypes = self._select_prototypes(scalar, k, generator)
            if prototypes is None:
                continue

            distances = torch.abs(scalar[:, None] - prototypes[None, :])
            labels = distances.argmin(dim=1).long()
            counts = torch.bincount(labels, minlength=k)
            if bool((counts >= minimum).all()):
                smallest_fraction = float((counts.float()/counts.sum().clamp_min(1)).min().item())
                retention = self._categorical_retention(scalar, labels)

                permutation = torch.randperm(k, generator=generator, device=z.device)
                labels = permutation[labels]
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

        return self._continuous(z, generator, name="continuous_fallback_from_prototype")            


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

        permutation = torch.randperm(k, generator=generator, device=z.device)
        labels = permutation[labels]

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


    # def observe_categorical(self, latent, generator, k):
    #     z = latent.float()
    #     categorical_probs = self.observation_type_probs[1:]
    #     categorical_probs = categorical_probs / categorical_probs.sum()
    #     method = int(torch.multinomial(categorical_probs, 1, generator=generator).item())
    #     if method == 0:
    #         observed = self._prototype(z, generator, k=k)
    #         if not observed.is_categorical:
    #             return self._dirichlet_binning(z, generator, k=k)
    #         return observed

    #     return self._dirichlet_binning(z, generator, k=k)

    def _target_discretization(self, z, observed_X, feature_type, feature_importance, k, n_neighbors=10, x_weight=0.7, generator=None):
        scalar = z[:, 0].float()
        X = observed_X.float()
        importance = feature_importance.float().clone()
        n, d = X.shape

        balance_factor = 0.5 + 0.25 * torch.distributions.Beta(3.26, 1.44).sample()
        minimum = int(torch.ceil(balance_factor * n / k).item())

        if n < k * minimum:
            return None

        distance_sq = torch.zeros((n, n), device=X.device, dtype=torch.float32)

        for j in range(d):
            xj = X[:, j]

            if bool(feature_type[j]):
                feature_distance_sq = (xj[:, None] != xj[None, :]).float()
            else:
                x_mean = xj.mean()
                x_std = xj.std(unbiased=False)
                if bool(x_std < 1e-12):
                    x_std = torch.tensor(1.0, device=X.device, dtype=X.dtype)
                x_scaled = (xj - x_mean) / x_std
                feature_distance_sq = (x_scaled[:, None] - x_scaled[None, :]).square()

            distance_sq += importance[j] * feature_distance_sq

        distances = torch.sqrt(distance_sq.clamp_min(0.0))
        distances.fill_diagonal_(float("inf"))

        num_neighbors = min(n_neighbors, n - 1)
        neighbor_distances, neighbor_indices = torch.topk(distances, k=num_neighbors, dim=1, largest=False)

        sigma = neighbor_distances.median().clamp_min(1e-12)
        neighbor_weights = torch.exp(-neighbor_distances.square() / (2.0 * sigma.square()))

        order = torch.argsort(scalar)
        scalar_sorted = scalar[order]

        sorted_position = torch.empty(n, dtype=torch.long, device=z.device)
        sorted_position[order] = torch.arange(n, device=z.device)

        src = torch.arange(n, device=z.device)[:, None].expand(-1, num_neighbors).reshape(-1)
        dst = neighbor_indices.reshape(-1)
        edge_weights = neighbor_weights.reshape(-1)

        mask = src < dst
        src = src[mask]
        dst = dst[mask]
        edge_weights = edge_weights[mask]

        pos_src = sorted_position[src]
        pos_dst = sorted_position[dst]
        left = torch.minimum(pos_src, pos_dst)
        right = torch.maximum(pos_src, pos_dst)

        difference = torch.zeros(n + 1, device=z.device, dtype=torch.float32)
        difference.scatter_add_(0, left + 1, edge_weights)
        difference.scatter_add_(0, right + 1, -edge_weights)

        x_cut_cost = torch.cumsum(difference, dim=0)
        valid_x_cost = x_cut_cost[minimum:n - minimum + 1]
        x_scale = valid_x_cost.mean().clamp_min(1e-12)
        x_cut_cost = x_cut_cost / x_scale

        prefix = torch.zeros(n + 1, device=z.device, dtype=torch.float32)
        prefix_sq = torch.zeros(n + 1, device=z.device, dtype=torch.float32)
        prefix[1:] = torch.cumsum(scalar_sorted, dim=0)
        prefix_sq[1:] = torch.cumsum(scalar_sorted.square(), dim=0)

        total_sum = prefix[n]
        total_sq_sum = prefix_sq[n]
        total_sse = (total_sq_sum - total_sum.square() / n).clamp_min(1e-12)

        starts = torch.arange(n + 1, device=z.device)[:, None]
        ends = torch.arange(n + 1, device=z.device)[None, :]
        counts = ends - starts
        safe_counts = counts.clamp_min(1)

        segment_sum = prefix[ends] - prefix[starts]
        segment_sq_sum = prefix_sq[ends] - prefix_sq[starts]
        segment_sse = segment_sq_sum - segment_sum.square() / safe_counts.float()
        segment_costs = segment_sse / total_sse

        invalid_segment = counts < minimum
        segment_costs = segment_costs.masked_fill(invalid_segment, float("inf"))

        valid_cut = torch.zeros(n + 1, dtype=torch.bool, device=z.device)
        valid_cut[1:n] = scalar_sorted[:-1] < scalar_sorted[1:]

        inf = float("inf")
        dp = torch.full((k + 1, n + 1), inf, device=z.device, dtype=torch.float32)
        back = torch.full((k + 1, n + 1), -1, device=z.device, dtype=torch.long)
        dp[0, 0] = 0.0

        for groups in range(1, k + 1):
            min_end = groups * minimum
            max_end = n - (k - groups) * minimum

            previous = dp[groups - 1]
            start_penalty = previous.clone()

            if groups > 1:
                start_penalty = start_penalty + x_weight * x_cut_cost
                start_penalty = start_penalty.masked_fill(~valid_cut, inf)

            candidate_costs = start_penalty[:, None] + segment_costs

            end_mask = torch.zeros(n + 1, dtype=torch.bool, device=z.device)
            end_mask[min_end:max_end + 1] = True
            candidate_costs[:, ~end_mask] = inf

            best_costs, best_starts = candidate_costs.min(dim=0)
            dp[groups] = best_costs
            back[groups] = best_starts

        if not bool(torch.isfinite(dp[k, n])):
            return None

        cuts = []
        end = n

        for groups in range(k, 1, -1):
            start = int(back[groups, end].item())
            if start < 0:
                return None
            cuts.append(start)
            end = start

        cuts.reverse()

        cut_tensor = torch.tensor(cuts, device=z.device, dtype=torch.long)
        thresholds = (scalar_sorted[cut_tensor - 1] + scalar_sorted[cut_tensor]) * 0.5

        labels = torch.bucketize(scalar, thresholds).long()
        if torch.rand((), generator=generator, device=z.device) < 0.5:
            permutation = torch.randperm(k, generator=generator, device=z.device)
            labels = permutation[labels]

        counts = torch.bincount(labels, minlength=k)
        if bool((counts < minimum).any()):
            return None

        smallest_fraction = float((counts.float() / counts.sum()).min().item())
        retention = self._categorical_retention(scalar, labels)

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=k,
            observation_type_id=-1,
            observation_type_name="target_discretization",
            quality_score=smallest_fraction,
            retention=retention,
            prototypes=torch.empty(0, 1, device=z.device, dtype=z.dtype),
            thresholds=thresholds,
        )
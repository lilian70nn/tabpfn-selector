
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from src.data.helper import make_gen, stratified_classification_split
from src.data.synthetic_task import GenerateTask


def _randn(*shape, generator, device):
    return torch.randn(*shape, generator=generator, device=device)


def _rand(*shape, generator, device):
    return torch.rand(*shape, generator=generator, device=device)


def _randint(low, high, shape, generator, device):
    return torch.randint(low, high, shape, generator=generator, device=device)


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6):
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(
        dim=dim, unbiased=False, keepdim=True
    ).clamp_min(eps)


def _normalize_probs(values, device, expected_len=None, name="probabilities"):
    probs = torch.tensor(values, device=device, dtype=torch.float32)
    if expected_len is not None and probs.numel() != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values.")
    if probs.numel() == 0 or bool((probs < 0).any()) or probs.sum() <= 0:
        raise ValueError(f"Invalid {name}.")
    return probs / probs.sum()


# ============================================================================
# Latent edge: [N, h] -> [N, h]
# ============================================================================


class LatentEdge:
    LINEAR = 0
    MLP = 1
    SOFT_TREE = 2

    ACTIVATIONS = ("identity", "tanh", "relu", "sigmoid", "sin", "square", "softplus")

    def __init__(
        self,
        latent_dim: int,
        generator: torch.Generator,
        device: torch.device,
        linear_activation_prob: float = 0.60,
        small_mlp_prob: float = 0.25,
        soft_tree_prob: float = 0.15,
        small_mlp_hidden_dim: Optional[int] = None,
        soft_tree_depth: int = 2,
        soft_tree_temperature: float = 0.5,
    ):
        self.latent_dim = int(latent_dim)
        self.device = device
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_temperature = float(soft_tree_temperature)

        probs = _normalize_probs(
            (linear_activation_prob, small_mlp_prob, soft_tree_prob),
            device,
            3,
            "edge-family probabilities",
        )
        self.edge_type = int(torch.multinomial(probs, 1, generator=generator).item())

        h = self.latent_dim
        self.linear_W = h**-0.5 * _randn(h, h, generator=generator, device=device)
        self.linear_b = _randn(h, generator=generator, device=device)
        self.activation_name = self.ACTIVATIONS[
            int(_randint(0, len(self.ACTIVATIONS), (), generator, device).item())
        ]

        hidden = small_mlp_hidden_dim or 2 * h
        self.mlp_W1 = h**-0.5 * _randn(hidden, h, generator=generator, device=device)
        self.mlp_b1 = _randn(hidden, generator=generator, device=device)
        self.mlp_W2 = hidden**-0.5 * _randn(h, hidden, generator=generator, device=device)
        self.mlp_b2 = _randn(h, generator=generator, device=device)

        n_internal = 2**self.soft_tree_depth - 1
        n_leaves = 2**self.soft_tree_depth
        self.tree_gate_W = h**-0.5 * _randn(
            n_internal, h, generator=generator, device=device
        )
        self.tree_gate_b = _randn(n_internal, generator=generator, device=device)
        self.tree_leaf_values = _randn(
            n_leaves, h, generator=generator, device=device
        )

    def _activation(self, x):
        if self.activation_name == "identity":
            return x
        if self.activation_name == "tanh":
            return torch.tanh(x)
        if self.activation_name == "relu":
            return torch.relu(x)
        if self.activation_name == "sigmoid":
            return torch.sigmoid(x) - 0.5
        if self.activation_name == "sin":
            return torch.sin(x)
        if self.activation_name == "square":
            return x.square()
        if self.activation_name == "softplus":
            return F.softplus(x)
        raise RuntimeError("Unknown activation.")

    def _soft_tree(self, x):
        logits = (x @ self.tree_gate_W.T + self.tree_gate_b) / self.soft_tree_temperature
        right = torch.sigmoid(logits)
        left = 1.0 - right
        paths = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        offset = 0
        for depth in range(self.soft_tree_depth):
            width = 2**depth
            l = left[:, offset : offset + width]
            r = right[:, offset : offset + width]
            paths = torch.stack((paths * l, paths * r), dim=-1).reshape(x.shape[0], -1)
            offset += width
        return paths @ self.tree_leaf_values

    def __call__(self, parent_latent):
        x = parent_latent.float()
        if self.edge_type == self.LINEAR:
            return self._activation(x @ self.linear_W.T + self.linear_b)
        if self.edge_type == self.MLP:
            hidden = torch.tanh(x @ self.mlp_W1.T + self.mlp_b1)
            return hidden @ self.mlp_W2.T + self.mlp_b2
        return self._soft_tree(x)


# ============================================================================
# Weighted connection
# ============================================================================


class WeightedLatentLayerConnection:
    """
    For every child:
      1. sample connected parents;
      2. sample a positive weight for every connected parent;
      3. normalize those weights to sum to one;
      4. aggregate weighted edge outputs.

    No n_dom and no pre-labelled dominant parent.
    """

    def __init__(
        self,
        in_width: int,
        out_width: int,
        latent_dim: int,
        connection_prob: float,
        min_parents_per_node: int,
        edge_weight_concentration: float,
        generator: torch.Generator,
        device: torch.device,
        **edge_kwargs,
    ):
        self.in_width = int(in_width)
        self.out_width = int(out_width)
        self.device = device

        self.adj = _rand(
            self.in_width,
            self.out_width,
            generator=generator,
            device=device,
        ) < connection_prob

        minimum = min(int(min_parents_per_node), self.in_width)
        for child in range(self.out_width):
            missing = minimum - int(self.adj[:, child].sum().item())
            if missing > 0:
                candidates = torch.where(~self.adj[:, child])[0]
                order = torch.randperm(
                    candidates.numel(), generator=generator, device=device
                )
                self.adj[candidates[order[:missing]], child] = True

        self.weights = torch.zeros(
            self.in_width, self.out_width, device=device, dtype=torch.float32
        )
        self.edges = [
            [None for _ in range(self.out_width)] for _ in range(self.in_width)
        ]

        for child in range(self.out_width):
            parents = torch.where(self.adj[:, child])[0]
            concentration = torch.full(
                (parents.numel(),),
                float(edge_weight_concentration),
                device=device,
            )
            raw = torch._standard_gamma(concentration, generator=generator).clamp_min(1e-8)
            self.weights[parents, child] = raw / raw.sum()

            for parent in parents.tolist():
                self.edges[parent][child] = LatentEdge(
                    latent_dim=latent_dim,
                    generator=generator,
                    device=device,
                    **edge_kwargs,
                )

    def __call__(self, parent_latents, generator, latent_noise_scale=0.0):
        children = []
        for child in range(self.out_width):
            value = None
            for parent in range(self.in_width):
                edge = self.edges[parent][child]
                if edge is None:
                    continue
                contribution = self.weights[parent, child] * edge(parent_latents[parent])
                value = contribution if value is None else value + contribution

            value = _standardize(value, dim=0)
            if latent_noise_scale > 0:
                value = value + latent_noise_scale * torch.randn(
                    value.shape,
                    generator=generator,
                    device=self.device,
                    dtype=value.dtype,
                )
                value = _standardize(value, dim=0)
            children.append(value)
        return children


# ============================================================================
# Full weighted SCM
# ============================================================================


class WeightedLayeredLatentSCM:
    ROOT_PRIORS = ("gaussian", "uniform", "heavy_tailed", "skewed", "mixture")

    def __init__(
        self,
        g_dag,
        g_x,
        g_aleatoric,
        num_roots=4,
        num_layers=5,
        hidden_width_min=8,
        hidden_width_max=12,
        final_width=1,
        latent_dim=6,
        connection_probs=(0.30, 0.45, 0.65, 0.85),
        min_parents_per_node=2,
        edge_weight_concentration=0.60,
        latent_noise_scale=0.03,
        root_prior_probs=(0.45, 0.20, 0.15, 0.05, 0.15),
        root_mixture_component_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        root_mixture_separation_min=1.5,
        root_mixture_separation_max=3.0,
        root_mixture_scale_min=0.40,
        root_mixture_scale_max=0.90,
        device=None,
        **edge_kwargs,
    ):
        self.device = device or torch.device("cpu")
        self.g_dag = g_dag
        self.g_x = g_x
        self.g_aleatoric = g_aleatoric
        self.num_roots = int(num_roots)
        self.num_layers = int(num_layers)
        self.hidden_width_min = int(hidden_width_min)
        self.hidden_width_max = int(hidden_width_max)
        self.final_width = int(final_width)
        self.latent_dim = int(latent_dim)
        self.connection_probs = tuple(float(p) for p in connection_probs)
        self.latent_noise_scale = float(latent_noise_scale)

        if len(self.connection_probs) != self.num_layers - 1:
            raise ValueError("connection_probs must have num_layers - 1 values.")

        self.root_prior_probs = _normalize_probs(
            root_prior_probs, self.device, 5, "root_prior_probs"
        )
        self.root_mixture_component_probs = _normalize_probs(
            root_mixture_component_probs,
            self.device,
            5,
            "root_mixture_component_probs",
        )
        self.root_mixture_separation_min = float(root_mixture_separation_min)
        self.root_mixture_separation_max = float(root_mixture_separation_max)
        self.root_mixture_scale_min = float(root_mixture_scale_min)
        self.root_mixture_scale_max = float(root_mixture_scale_max)

        self.widths = [self.num_roots]
        for _ in range(self.num_layers - 2):
            self.widths.append(
                int(
                    _randint(
                        self.hidden_width_min,
                        self.hidden_width_max + 1,
                        (),
                        self.g_dag,
                        self.device,
                    ).item()
                )
            )
        self.widths.append(self.final_width)

        self.connections = []
        for layer in range(self.num_layers - 1):
            self.connections.append(
                WeightedLatentLayerConnection(
                    in_width=self.widths[layer],
                    out_width=self.widths[layer + 1],
                    latent_dim=self.latent_dim,
                    connection_prob=self.connection_probs[layer],
                    min_parents_per_node=min_parents_per_node,
                    edge_weight_concentration=edge_weight_concentration,
                    generator=self.g_dag,
                    device=self.device,
                    **edge_kwargs,
                )
            )

        self.root_prior_types = []
        self.root_prior_type_ids = []
        self.root_mixture_components = []


    def _sample_mixture_root(self, n):
        component_idx = int(
            torch.multinomial(
                self.root_mixture_component_probs, 1, generator=self.g_dag
            ).item()
        )
        k = component_idx + 2
        component_weights = torch.softmax(
            0.5 * _randn(k, generator=self.g_dag, device=self.device), dim=0
        )
        ids = torch.multinomial(
            component_weights, n, replacement=True, generator=self.g_x
        )
        directions = _randn(
            k, self.latent_dim, generator=self.g_dag, device=self.device
        )
        directions = directions / torch.linalg.vector_norm(
            directions, dim=1, keepdim=True
        ).clamp_min(1e-6)
        separation = self.root_mixture_separation_min + (
            self.root_mixture_separation_max - self.root_mixture_separation_min
        ) * _rand((), generator=self.g_dag, device=self.device)
        centers = separation * directions
        scales = self.root_mixture_scale_min + (
            self.root_mixture_scale_max - self.root_mixture_scale_min
        ) * _rand(k, generator=self.g_dag, device=self.device)
        noise = _randn(
            n, self.latent_dim, generator=self.g_x, device=self.device
        )
        return centers[ids] + scales[ids, None] * noise, k

    def sample_root_latents(self, n):
        roots = []
        self.root_prior_types = []
        self.root_prior_type_ids = []
        self.root_mixture_components = []

        for _ in range(self.num_roots):
            prior_id = int(
                torch.multinomial(self.root_prior_probs, 1, generator=self.g_dag).item()
            )
            name = self.ROOT_PRIORS[prior_id]
            mixture_k = 0

            if name == "gaussian":
                z = _randn(n, self.latent_dim, generator=self.g_x, device=self.device)
            elif name == "uniform":
                bound = 3.0**0.5
                z = 2 * bound * _rand(
                    n, self.latent_dim, generator=self.g_x, device=self.device
                ) - bound
            elif name == "heavy_tailed":
                df = 4.0
                numerator = _randn(
                    n, self.latent_dim, generator=self.g_x, device=self.device
                )
                concentration = torch.full(
                    (n, 1), df / 2, device=self.device, dtype=numerator.dtype
                )
                chi2 = 2 * torch._standard_gamma(
                    concentration, generator=self.g_x
                )
                z = numerator / torch.sqrt(chi2 / df).clamp_min(1e-4)
            elif name == "skewed":
                normal = _randn(
                    n, self.latent_dim, generator=self.g_x, device=self.device
                )
                strength = 0.4 + 0.6 * _rand(
                    self.latent_dim, generator=self.g_dag, device=self.device
                )
                z = torch.exp(normal * strength[None, :])
            else:
                z, mixture_k = self._sample_mixture_root(n)

            roots.append(_standardize(z.float(), dim=0))
            self.root_prior_types.append(name)
            self.root_prior_type_ids.append(prior_id)
            self.root_mixture_components.append(mixture_k)

        return roots

    def forward(self, n_samples, latent_noise_scale=None):
        current = self.sample_root_latents(n_samples)
        all_latents = [current]
        noise = self.latent_noise_scale if latent_noise_scale is None else latent_noise_scale
        for connection in self.connections:
            current = connection(
                current,
                generator=self.g_aleatoric,
                latent_noise_scale=noise,
            )
            all_latents.append(current)
        return all_latents

    def compute_node_influence(self, target_node_idx=0):
        """
        influence(parent) = sum_child edge_weight(parent, child) * influence(child)

        This automatically sums all path-weight products from each node to target.
        """
        influence = [
            torch.zeros(width, device=self.device, dtype=torch.float32)
            for width in self.widths
        ]
        influence[-1][target_node_idx] = 1.0

        for layer in range(self.num_layers - 2, -1, -1):
            influence[layer] = (
                self.connections[layer].weights @ influence[layer + 1]
            )
        return influence


# ============================================================================
# Feature observation
# ============================================================================


@dataclass(frozen=True)
class FeatureObservation:
    values: torch.Tensor
    is_categorical: bool
    cardinality: int
    observation_type_id: int
    observation_type_name: str
    quality_score: float
    prototypes: torch.Tensor
    thresholds: torch.Tensor
    projection: torch.Tensor


class AdaptiveObservationHead:
    CONTINUOUS = 0
    PROTOTYPE = 1
    BINNING = 2

    NAMES = (
        "continuous_projection",
        "prototype_discretization",
        "threshold_binning",
    )

    def __init__(
        self,
        latent_dim,
        generator,
        device,
        observation_type_probs=(0.60, 0.20, 0.20),
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        min_samples_per_category=8,
        min_component_weight=0.05,
        prototype_max_attempts=8,
        prototype_min_separation=1.0,
        binning_jitter=0.20,
        observation_noise_scale=0.05,
    ):
        self.latent_dim = int(latent_dim)
        self.device = device
        self.min_samples_per_category = int(min_samples_per_category)
        self.min_component_weight = float(min_component_weight)
        self.prototype_max_attempts = int(prototype_max_attempts)
        self.prototype_min_separation = float(prototype_min_separation)
        self.binning_jitter = float(binning_jitter)
        self.observation_noise_scale = float(observation_noise_scale)

        self.observation_type_probs = _normalize_probs(
            observation_type_probs, device, 3, "observation_type_probs"
        )
        self.cardinalities = tuple(int(k) for k in categorical_cardinalities)
        self.cardinality_probs = _normalize_probs(
            categorical_cardinality_probs,
            device,
            len(self.cardinalities),
            "categorical_cardinality_probs",
        )

        self.sampled_type = int(
            torch.multinomial(
                self.observation_type_probs, 1, generator=generator
            ).item()
        )
        h = self.latent_dim
        # self.continuous_W = h**-0.5 * _randn(h, generator=generator, device=device)
        # self.continuous_b = _randn((), generator=generator, device=device)
        # self.binning_W = h**-0.5 * _randn(h, generator=generator, device=device)
        # self.binning_b = _randn((), generator=generator, device=device)

    def _sample_cardinality(self, n, generator):
        feasible = [
            i
            for i, k in enumerate(self.cardinalities)
            if k * self.min_samples_per_category <= n
            and k * self.min_component_weight <= 1.0
        ]
        if not feasible:
            return 0
        probs = self.cardinality_probs[
            torch.tensor(feasible, device=self.device)
        ]
        probs = probs / probs.sum()
        pos = int(torch.multinomial(probs, 1, generator=generator).item())
        return self.cardinalities[feasible[pos]]

    def _continuous(self, z, W, b, generator, name="continuous_projection"):
        score = z @ W + b
        if self.observation_noise_scale > 0:
            score = score + self.observation_noise_scale * torch.randn(
                score.shape,
                generator=generator,
                device=z.device,
                dtype=z.dtype,
            )
        score = _standardize(score)
        return FeatureObservation(
            score,
            False,
            0,
            self.CONTINUOUS,
            name,
            0.0,
            torch.empty(0, self.latent_dim, device=z.device),
            torch.empty(0, device=z.device),
            W.detach().clone(),
        )


    def _select_prototypes(
        self,
        z: torch.Tensor,
        k: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        indices = torch.randperm(
            z.shape[0],
            generator=generator,
            device=z.device,
        )[:k]

        return z[indices]

    def _prototype(
        self,
        z: torch.Tensor,
        W, b, 
        generator: torch.Generator,
    ) -> FeatureObservation:
        k = self._sample_cardinality(
            z.shape[0],
            generator,
        )

        if k == 0:
            return self._continuous(
                z,
                W,
                b,
                generator,
                "continuous_fallback_from_prototype",
            )

        prototypes = self._select_prototypes(
            z,
            k,
            generator,
        )

        labels = torch.cdist(
            z,
            prototypes,
        ).argmin(dim=1)

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=k,
            observation_type_id=self.PROTOTYPE,
            observation_type_name="prototype_discretization",
            quality_score=0.0,
            prototypes=prototypes,
            thresholds=torch.empty(
                0,
                device=z.device,
                dtype=z.dtype,
            ),
            projection=torch.empty(
                0,
                device=z.device,
                dtype=z.dtype,
            ),
        )

    def _binning(self, z, W, b,generator):
        k = self._sample_cardinality(z.shape[0], generator)
        if k == 0:
            return self._continuous(
                z, W, b, generator, "continuous_fallback_from_binning"
            )

        scalar = _standardize(z @ W + b)
        n = scalar.numel()
        minimum = max(
            self.min_samples_per_category,
            int(torch.ceil(torch.tensor(self.min_component_weight * n)).item()),
        )
        remaining = n - k * minimum
        raw = _rand(k, generator=generator, device=z.device)
        extras_float = raw / raw.sum() * remaining
        extras = torch.floor(extras_float).long()
        extras[
            torch.argsort(extras_float - extras.float(), descending=True)[
                : remaining - int(extras.sum().item())
            ]
        ] += 1
        counts = (extras + minimum).tolist()

        sorted_values = torch.sort(scalar).values
        thresholds = []
        cumulative = 0
        for count in counts[:-1]:
            cumulative += int(count)
            thresholds.append(
                0.5 * (sorted_values[cumulative - 1] + sorted_values[cumulative])
            )
        thresholds = torch.stack(thresholds)
        labels = torch.bucketize(scalar, thresholds)
        observed_counts = torch.bincount(labels, minlength=k)
        quality = float(
            (observed_counts.float() / observed_counts.sum()).min().item()
        )
        return FeatureObservation(
            labels,
            True,
            k,
            self.BINNING,
            "threshold_binning",
            quality,
            torch.empty(0, self.latent_dim, device=z.device),
            thresholds,
            W.detach().clone(),
        )

    def observe(self, latent, shared_W, shared_b, generator):
        z = _standardize(latent.float(), dim=0)
        if self.sampled_type == self.CONTINUOUS:
            return self._continuous(z, shared_W, shared_b, generator)
        if self.sampled_type == self.PROTOTYPE:
            return self._prototype(z, shared_W, shared_b, generator)
        return self._binning(z, shared_W, shared_b, generator)


class TargetObservationHead:
    def __init__(self, latent_dim, generator, device, observation_noise_scale=0.03):
        self.device = device
        self.observation_noise_scale = float(observation_noise_scale)
        # self.W = latent_dim**-0.5 * _randn(
        #     latent_dim, generator=generator, device=device
        # )
        # self.b = _randn((), generator=generator, device=device)

    def score(self, latent, W, b, generator):
        value = latent.float() @ W + b
        if self.observation_noise_scale > 0:
            value = value + self.observation_noise_scale * torch.randn(
                value.shape,
                generator=generator,
                device=self.device,
                dtype=value.dtype,
            )
        return _standardize(value)

    @staticmethod
    def balanced_classes(score, num_classes):
        order = torch.argsort(score)
        labels = torch.empty_like(order)
        n = score.numel()
        start = 0
        for class_id in range(num_classes):
            size = n // num_classes + (1 if class_id < n % num_classes else 0)
            labels[order[start : start + size]] = class_id
            start += size
        return labels.long()


# ============================================================================
# Task
# ============================================================================


class WeightedMixedLatentSCMTask(GenerateTask):
    CONTINUOUS = 0
    CATEGORICAL = 1

    def __init__(
        self,
        num_classes=None,
        n_min=400,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_missing=0.05,
        device=None,
        dag_seed=None,
        aleatoric_seed=None,
        x_seed=None,
        num_roots=4,
        num_layers=5,
        hidden_width_min=8,
        hidden_width_max=12,
        final_width=1,
        latent_dim=6,
        connection_probs=(0.30, 0.45, 0.65, 0.85),
        min_parents_per_node=2,
        edge_weight_concentration=0.60,
        latent_noise_scale=0.03,
        observation_noise_scale=0.03,
        dominant_mass_threshold=0.70,
        dominant_feature_fraction=0.70,
        observation_type_probs=(0.60, 0.20, 0.20),
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        min_samples_per_category=8,
        min_component_weight=0.05,
        prototype_max_attempts=8,
        prototype_min_separation=1.0,
        binning_jitter=0.20,
        root_prior_probs=(0.45, 0.20, 0.15, 0.05, 0.15),
        root_mixture_component_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        root_mixture_separation_min=1.5,
        root_mixture_separation_max=3.0,
        root_mixture_scale_min=0.40,
        root_mixture_scale_max=0.90,
        linear_activation_prob=0.60,
        small_mlp_prob=0.25,
        soft_tree_prob=0.15,
        small_mlp_hidden_dim=None,
        soft_tree_depth=2,
        soft_tree_temperature=0.5,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_classes = num_classes
        self.n_min, self.n_max = int(n_min), int(n_max)
        self.d_min, self.d_max = int(d_min), int(d_max)
        self.test_frac = float(test_frac)
        self.p_missing = float(p_missing)

        self.scm_kwargs = dict(
            num_roots=num_roots,
            num_layers=num_layers,
            hidden_width_min=hidden_width_min,
            hidden_width_max=hidden_width_max,
            final_width=final_width,
            latent_dim=latent_dim,
            connection_probs=connection_probs,
            min_parents_per_node=min_parents_per_node,
            edge_weight_concentration=edge_weight_concentration,
            latent_noise_scale=latent_noise_scale,
            root_prior_probs=root_prior_probs,
            root_mixture_component_probs=root_mixture_component_probs,
            root_mixture_separation_min=root_mixture_separation_min,
            root_mixture_separation_max=root_mixture_separation_max,
            root_mixture_scale_min=root_mixture_scale_min,
            root_mixture_scale_max=root_mixture_scale_max,
            linear_activation_prob=linear_activation_prob,
            small_mlp_prob=small_mlp_prob,
            soft_tree_prob=soft_tree_prob,
            small_mlp_hidden_dim=small_mlp_hidden_dim,
            soft_tree_depth=soft_tree_depth,
            soft_tree_temperature=soft_tree_temperature,
            device=self.device,
        )
        self.latent_dim = int(latent_dim)
        self.latent_noise_scale = float(latent_noise_scale)
        self.observation_noise_scale = float(observation_noise_scale)
        self.dominant_mass_threshold = float(dominant_mass_threshold)
        self.dominant_feature_fraction = float(dominant_feature_fraction)

        self.observation_kwargs = dict(
            observation_type_probs=observation_type_probs,
            categorical_cardinalities=categorical_cardinalities,
            categorical_cardinality_probs=categorical_cardinality_probs,
            min_samples_per_category=min_samples_per_category,
            min_component_weight=min_component_weight,
            prototype_max_attempts=prototype_max_attempts,
            prototype_min_separation=prototype_min_separation,
            binning_jitter=binning_jitter,
            observation_noise_scale=observation_noise_scale,
        )

        self.g_dag, self.dag_seed = make_gen(self.device, dag_seed)
        self.g_aleatoric, self.aleatoric_seed = make_gen(
            self.device, aleatoric_seed
        )
        self.g_x, self.x_seed = make_gen(self.device, x_seed)

        self.n = int(
            _randint(
                self.n_min,
                self.n_max + 1,
                (1,),
                self.g_dag,
                self.device,
            ).item()
        )
        self.d = int(
            _randint(
                self.d_min,
                self.d_max + 1,
                (1,),
                self.g_dag,
                self.device,
            ).item()
        )

        h = self.latent_dim

        self.shared_W = (
            h ** -0.5
        ) * _randn(
            h,
            generator=self.g_dag,
            device=self.device,
        )

        self.shared_b = _randn(
            (),
            generator=self.g_dag,
            device=self.device,
        )



        super().__init__()

    @staticmethod
    def _flatten(all_latents):
        values, index = [], []
        for layer_idx, layer in enumerate(all_latents):
            for node_idx, value in enumerate(layer):
                values.append(value)
                index.append((layer_idx, node_idx))
        return values, index

    def _dominant_group(self, candidate_ids, flat_influence):
        influence = flat_influence[candidate_ids]
        mask = influence > 0
        ids = candidate_ids[mask]
        influence = influence[mask]

        order = torch.argsort(influence, descending=True)
        ids = ids[order]
        influence = influence[order]
        cutoff = self.dominant_mass_threshold * influence.sum()
        reached = torch.where(torch.cumsum(influence, dim=0) >= cutoff)[0]
        size = int(reached[0].item()) + 1 if reached.numel() else ids.numel()
        return ids[:size]

    # def _sample_without_replacement(self, ids, weights, count):
    #     if count <= 0 or ids.numel() == 0:
    #         return torch.empty(0, device=self.device, dtype=torch.long)
    #     count = min(int(count), int(ids.numel()))
    #     positions = torch.multinomial(
    #         weights.clamp_min(1e-8),
    #         count,
    #         replacement=False,
    #         generator=self.g_dag,
    #     )
    #     return ids[positions]
    
    def _sample_without_replacement(self,ids,count):
        if count <= 0 or ids.numel() == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        count = min(int(count),int(ids.numel()))
        positions = torch.randperm(
            ids.numel(),
            generator=self.g_dag,
            device=ids.device,
        )[:count]
        return ids[positions]

    def _sample_feature_ids(self, flat_index, flat_influence):
        candidates = torch.tensor(
            [
                i
                for i, (layer, _) in enumerate(flat_index)
                if layer < len(self.scm.widths) - 1
            ],
            device=self.device,
            dtype=torch.long,
        )
        d = min(self.d, candidates.numel())
        dominant = self._dominant_group(candidates, flat_influence)
        dominant_set = set(dominant.tolist())
        other = torch.tensor(
            [i for i in candidates.tolist() if i not in dominant_set],
            device=self.device,
            dtype=torch.long,
        )

        n_dom = min(
            round(self.dominant_feature_fraction * d),
            dominant.numel(),
        )
        n_other = d - n_dom

        if n_other > other.numel():
            n_dom += n_other - other.numel()
            n_other = other.numel()

        selected_dom = self._sample_without_replacement(
            dominant, n_dom
        )
        selected_other = self._sample_without_replacement(
            other, n_other
        )
        selected = torch.cat((selected_dom, selected_other))

        if selected.numel() < d:
            selected_set = set(selected.tolist())
            remaining = torch.tensor(
                [i for i in candidates.tolist() if i not in selected_set],
                device=self.device,
                dtype=torch.long,
            )
            fill = self._sample_without_replacement(
                remaining,
                d - selected.numel(),
            )
            selected = torch.cat((selected, fill))

        selected = selected[
            torch.randperm(selected.numel(), generator=self.g_dag, device=self.device)
        ]
        return selected.tolist(), dominant

    def _observe_features(self, flat_latents, feature_ids):
        n, d = flat_latents[0].shape[0], len(feature_ids)
        X = torch.empty(n, d, device=self.device)
        feature_type = torch.empty(d, device=self.device, dtype=torch.long)
        cardinality = torch.zeros(d, device=self.device, dtype=torch.long)
        type_ids = torch.empty(d, device=self.device, dtype=torch.long)
        type_names, prototypes, thresholds, projections, heads = [], [], [], [], []
        quality = torch.zeros(d, device=self.device)

        for col, node_id in enumerate(feature_ids):
            head = AdaptiveObservationHead(
                self.latent_dim,
                self.g_dag,
                self.device,
                **self.observation_kwargs,
            )
            observed = head.observe(flat_latents[node_id], self.shared_W, self.shared_b, self.g_aleatoric)
            X[:, col] = observed.values.float()
            feature_type[col] = (
                self.CATEGORICAL if observed.is_categorical else self.CONTINUOUS
            )
            cardinality[col] = observed.cardinality
            type_ids[col] = observed.observation_type_id
            type_names.append(observed.observation_type_name)
            quality[col] = observed.quality_score
            prototypes.append(observed.prototypes)
            thresholds.append(observed.thresholds)
            projections.append(observed.projection)
            heads.append(head)

        return (
            X,
            feature_type,
            cardinality,
            type_ids,
            type_names,
            quality,
            prototypes,
            thresholds,
            projections,
            heads,
        )

    def _generate(self):
        self.scm = WeightedLayeredLatentSCM(
            self.g_dag,
            self.g_x,
            self.g_aleatoric,
            **self.scm_kwargs,
        )
        all_latents = self.scm.forward(
            self.n, latent_noise_scale=self.latent_noise_scale
        )
        flat_latents, flat_index = self._flatten(all_latents)

        layer_influence = self.scm.compute_node_influence(target_node_idx=0)
        flat_influence = torch.cat(layer_influence)

        feature_ids, dominant_group = self._sample_feature_ids(
            flat_index, flat_influence
        )
        self.d = len(feature_ids)

        (
            X_clean,
            feature_type,
            cardinality,
            type_ids,
            type_names,
            quality,
            prototypes,
            thresholds,
            projections,
            heads,
        ) = self._observe_features(flat_latents, feature_ids)

        target_global_id = sum(self.scm.widths[:-1])
        target_head = TargetObservationHead(
            self.latent_dim,
            self.g_dag,
            self.device,
            self.observation_noise_scale,
        )
        target_score = target_head.score(
            flat_latents[target_global_id], self.shared_W, self.shared_b, self.g_aleatoric
        )

        if self.num_classes is None:
            y = target_score
            self.n_classes = None
        else:
            y = target_head.balanced_classes(
                target_score, int(self.num_classes)
            )
            self.n_classes = int(self.num_classes)

        feature_ids_t = torch.tensor(
            feature_ids, device=self.device, dtype=torch.long
        )
        feature_strength = flat_influence[feature_ids_t]
        importance_ratio = feature_strength / feature_strength.sum().clamp_min(1e-12)

        dominant_set = set(dominant_group.tolist())
        selected_from_dominant = torch.tensor(
            [float(i in dominant_set) for i in feature_ids],
            device=self.device,
        )

        X_obs = X_clean.clone()
        missing_mask = _rand(
            *X_obs.shape, generator=self.g_x, device=self.device
        ) < self.p_missing
        X_obs[missing_mask] = torch.nan

        if self.num_classes is not None:
            train_idx, test_idx = stratified_classification_split(
                y=y.long(),
                test_frac=self.test_frac,
                generator=self.g_x,
                device=self.device,
            )
        else:
            n_test = min(max(1, round(self.n * self.test_frac)), self.n - 2)
            order = torch.randperm(
                self.n, generator=self.g_x, device=self.device
            )
            train_idx, test_idx = order[:-n_test], order[-n_test:]

        info = {
            "feature_type": feature_type,
            "cardinality": cardinality,
            "feature_observation_type_ids": type_ids,
            "feature_observation_type_names": type_names,
            "feature_observation_quality": quality,
            "feature_prototypes": prototypes,
            "feature_thresholds": thresholds,
            "feature_projections": projections,
            "feature_ids": feature_ids_t,
            "target_id": torch.tensor(target_global_id, device=self.device),
            "feature_strength": feature_strength,
            "importance_ratio": importance_ratio,
            "is_active": (feature_strength > 0).float(),
            "sampled_active": selected_from_dominant,
            "selected_from_dominant_group": selected_from_dominant,
            "dominant_group_ids": dominant_group,
            "all_node_influence": flat_influence,
            "layer_node_influence": layer_influence,
            "layer_widths": torch.tensor(self.scm.widths, device=self.device),
            "connection_probs": torch.tensor(
                self.scm.connection_probs, device=self.device
            ),
            "adjacency_matrices": [c.adj for c in self.scm.connections],
            "edge_weight_matrices": [c.weights for c in self.scm.connections],
            "root_prior_types": list(self.scm.root_prior_types),
            "root_prior_type_ids": torch.tensor(
                self.scm.root_prior_type_ids, device=self.device
            ),
            "root_mixture_components": torch.tensor(
                self.scm.root_mixture_components, device=self.device
            ),
            "missing_mask_train": missing_mask[train_idx],
            "missing_mask_test": missing_mask[test_idx],
        }

        self.feature_type = feature_type
        self.cardinality = cardinality
        self.feature_observation_heads = heads
        self.target_observation_head = target_head
        self.n_features = self.d

        return (
            X_obs[train_idx],
            y[train_idx],
            X_obs[test_idx],
            y[test_idx],
            info,
        )

    def visualize(self):
        return None

    def forward(self, X):
        del X
        return None


# # Compatible names.
# MixedLatentSCMTask = WeightedMixedLatentSCMTask
# MixedSCMTask = WeightedMixedLatentSCMTask
# RandomLayeredLatentSCM = WeightedLayeredLatentSCM
# RandomLayeredSCM = WeightedLayeredLatentSCM

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.data.helper import make_gen, stratified_classification_split
from src.data.synthetic_task import GenerateTask


def _randn(
    *shape: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, device=device)


def _rand(
    *shape: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.rand(*shape, generator=generator, device=device)


def _randint(
    low: int,
    high: int,
    shape,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        low,
        high,
        shape,
        generator=generator,
        device=device,
    )


def _standardize(
    x: torch.Tensor,
    dim: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True)
    return (x - mean) / std.clamp_min(eps)


def _sample_beta_scalar(
    alpha: float,
    beta: float,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive.")

    a = torch.tensor(alpha, device=device, dtype=torch.float32)
    b = torch.tensor(beta, device=device, dtype=torch.float32)
    x = torch._standard_gamma(a, generator=generator)
    y = torch._standard_gamma(b, generator=generator)
    return float((x / (x + y).clamp_min(1e-12)).item())


# ---------------------------------------------------------------------------
# Latent-space SCM
# ---------------------------------------------------------------------------


class LatentEdge:
    """A permanently sampled latent-to-latent edge: [N, h] -> [N, h]."""

    EDGE_LINEAR_ACTIVATION = 0
    EDGE_SMALL_MLP = 1
    EDGE_SOFT_TREE = 2

    ACTIVATION_NAMES = (
        "identity",
        "tanh",
        "relu",
        "sigmoid",
        "sin",
        "square",
        "softplus",
    )

    EDGE_NAMES = {
        EDGE_LINEAR_ACTIVATION: "linear_activation",
        EDGE_SMALL_MLP: "small_mlp",
        EDGE_SOFT_TREE: "soft_tree",
    }

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
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if soft_tree_depth <= 0:
            raise ValueError("soft_tree_depth must be positive.")
        if soft_tree_temperature <= 0:
            raise ValueError("soft_tree_temperature must be positive.")

        probs = torch.tensor(
            [linear_activation_prob, small_mlp_prob, soft_tree_prob],
            device=device,
            dtype=torch.float32,
        )
        if bool((probs < 0).any()) or float(probs.sum().item()) <= 0:
            raise ValueError("Edge-family probabilities must be nonnegative and nonzero.")
        probs = probs / probs.sum()

        self.latent_dim = int(latent_dim)
        self.device = device
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_temperature = float(soft_tree_temperature)

        self.edge_type = int(
            torch.multinomial(
                probs,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).item()
        )
        self.edge_name = self.EDGE_NAMES[self.edge_type]

        scale = self.latent_dim ** -0.5
        self.linear_W = scale * _randn(
            self.latent_dim,
            self.latent_dim,
            generator=generator,
            device=device,
        )
        self.linear_b = _randn(
            self.latent_dim,
            generator=generator,
            device=device,
        )
        self.activation_id = int(
            _randint(
                0,
                len(self.ACTIVATION_NAMES),
                (),
                generator=generator,
                device=device,
            ).item()
        )
        self.activation_name = self.ACTIVATION_NAMES[self.activation_id]

        hidden_dim = (
            int(small_mlp_hidden_dim)
            if small_mlp_hidden_dim is not None
            else 2 * self.latent_dim
        )
        if hidden_dim <= 0:
            raise ValueError("small_mlp_hidden_dim must be positive.")

        self.mlp_W1 = (self.latent_dim ** -0.5) * _randn(
            hidden_dim,
            self.latent_dim,
            generator=generator,
            device=device,
        )
        self.mlp_b1 = _randn(hidden_dim, generator=generator, device=device)
        self.mlp_W2 = (hidden_dim ** -0.5) * _randn(
            self.latent_dim,
            hidden_dim,
            generator=generator,
            device=device,
        )
        self.mlp_b2 = _randn(self.latent_dim, generator=generator, device=device)

        n_internal = 2**self.soft_tree_depth - 1
        n_leaves = 2**self.soft_tree_depth
        self.tree_gate_W = (self.latent_dim ** -0.5) * _randn(
            n_internal,
            self.latent_dim,
            generator=generator,
            device=device,
        )
        self.tree_gate_b = _randn(n_internal, generator=generator, device=device)
        self.tree_leaf_values = _randn(
            n_leaves,
            self.latent_dim,
            generator=generator,
            device=device,
        )

    def name(self) -> str:
        if self.edge_type == self.EDGE_LINEAR_ACTIVATION:
            return f"{self.edge_name}:{self.activation_name}"
        return self.edge_name

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
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
        raise RuntimeError(f"Unknown activation: {self.activation_name}")

    def _soft_tree(self, x: torch.Tensor) -> torch.Tensor:
        logits = (x @ self.tree_gate_W.T + self.tree_gate_b) / self.soft_tree_temperature
        right_prob = torch.sigmoid(logits)
        left_prob = 1.0 - right_prob

        path_probs = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        offset = 0
        for depth in range(self.soft_tree_depth):
            nodes_at_level = 2**depth
            left = left_prob[:, offset : offset + nodes_at_level]
            right = right_prob[:, offset : offset + nodes_at_level]
            path_probs = torch.stack(
                [path_probs * left, path_probs * right],
                dim=-1,
            ).reshape(x.shape[0], -1)
            offset += nodes_at_level

        return path_probs @ self.tree_leaf_values

    def __call__(self, parent_latent: torch.Tensor) -> torch.Tensor:
        if parent_latent.ndim != 2 or parent_latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected parent latent [N, {self.latent_dim}], "
                f"got {tuple(parent_latent.shape)}."
            )

        x = parent_latent.float()

        if self.edge_type == self.EDGE_LINEAR_ACTIVATION:
            return self._activation(x @ self.linear_W.T + self.linear_b)
        if self.edge_type == self.EDGE_SMALL_MLP:
            hidden = torch.tanh(x @ self.mlp_W1.T + self.mlp_b1)
            return hidden @ self.mlp_W2.T + self.mlp_b2
        if self.edge_type == self.EDGE_SOFT_TREE:
            return self._soft_tree(x)

        raise RuntimeError(f"Unknown edge_type={self.edge_type}.")


class LatentLayerConnection:
    """Sparse connection from one latent layer to the next."""

    def __init__(
        self,
        in_width: int,
        out_width: int,
        latent_dim: int,
        edge_prob: float,
        min_parents_per_node: int,
        generator: torch.Generator,
        device: torch.device,
        linear_activation_prob: float = 0.60,
        small_mlp_prob: float = 0.25,
        soft_tree_prob: float = 0.15,
        small_mlp_hidden_dim: Optional[int] = None,
        soft_tree_depth: int = 2,
        soft_tree_temperature: float = 0.5,
    ) -> None:
        if in_width <= 0 or out_width <= 0:
            raise ValueError("Layer widths must be positive.")
        if not 0.0 <= edge_prob <= 1.0:
            raise ValueError("edge_prob must lie in [0, 1].")
        if min_parents_per_node < 1:
            raise ValueError("min_parents_per_node must be at least 1.")

        self.in_width = int(in_width)
        self.out_width = int(out_width)
        self.latent_dim = int(latent_dim)
        self.device = device

        min_parents = min(int(min_parents_per_node), self.in_width)
        self.adj = _rand(
            self.in_width,
            self.out_width,
            generator=generator,
            device=device,
        ) < edge_prob

        for child_idx in range(self.out_width):
            missing = min_parents - int(self.adj[:, child_idx].sum().item())
            if missing <= 0:
                continue
            candidates = torch.where(~self.adj[:, child_idx])[0]
            order = torch.randperm(
                candidates.numel(),
                generator=generator,
                device=device,
            )
            self.adj[candidates[order[:missing]], child_idx] = True

        self.edges: list[list[Optional[LatentEdge]]] = [
            [None for _ in range(self.out_width)] for _ in range(self.in_width)
        ]
        for parent_idx in range(self.in_width):
            for child_idx in range(self.out_width):
                if bool(self.adj[parent_idx, child_idx]):
                    self.edges[parent_idx][child_idx] = LatentEdge(
                        latent_dim=self.latent_dim,
                        generator=generator,
                        device=device,
                        linear_activation_prob=linear_activation_prob,
                        small_mlp_prob=small_mlp_prob,
                        soft_tree_prob=soft_tree_prob,
                        small_mlp_hidden_dim=small_mlp_hidden_dim,
                        soft_tree_depth=soft_tree_depth,
                        soft_tree_temperature=soft_tree_temperature,
                    )

    def __call__(
        self,
        parent_latents: list[torch.Tensor],
        generator: torch.Generator,
        latent_noise_scale: float = 0.0,
    ) -> list[torch.Tensor]:
        if len(parent_latents) != self.in_width:
            raise ValueError(
                f"Expected {self.in_width} parent nodes, got {len(parent_latents)}."
            )
        if latent_noise_scale < 0:
            raise ValueError("latent_noise_scale must be nonnegative.")

        children: list[torch.Tensor] = []
        for child_idx in range(self.out_width):
            incoming = [
                edge(parent_latents[parent_idx])
                for parent_idx in range(self.in_width)
                if (edge := self.edges[parent_idx][child_idx]) is not None
            ]
            if not incoming:
                raise RuntimeError("Every child must have at least one parent.")

            child = torch.stack(incoming, dim=0).sum(dim=0) / (len(incoming) ** 0.5)
            if latent_noise_scale > 0:
                child = child + latent_noise_scale * torch.randn(
                    child.shape,
                    generator=generator,
                    device=self.device,
                    dtype=child.dtype,
                )
            children.append(child)

        return children


class RandomLayeredLatentSCM:
    """
    Sparse layered SCM containing continuous latent vectors only.

    The SCM does not know whether a selected node will later be observed as a
    continuous or categorical table column.
    """

    def __init__(
        self,
        g_dag: torch.Generator,
        g_x: torch.Generator,
        g_aleatoric: torch.Generator,
        num_roots: int = 3,
        num_layers: int = 4,
        max_nodes_per_layer: int = 8,
        latent_dim: int = 8,
        edge_beta_alpha: float = 2.0,
        edge_beta_beta: float = 5.0,
        edge_prob_min: float = 0.05,
        edge_prob_max: float = 0.95,
        min_parents_per_node: int = 1,
        latent_noise_scale: float = 0.05,
        linear_activation_prob: float = 0.60,
        small_mlp_prob: float = 0.25,
        soft_tree_prob: float = 0.15,
        small_mlp_hidden_dim: Optional[int] = None,
        soft_tree_depth: int = 2,
        soft_tree_temperature: float = 0.5,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")

        if num_roots <= 0:
            raise ValueError("num_roots must be positive.")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")
        if max_nodes_per_layer < 5:
            raise ValueError("max_nodes_per_layer must be at least 5.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if not 0.0 <= edge_prob_min <= edge_prob_max <= 1.0:
            raise ValueError("Require 0 <= edge_prob_min <= edge_prob_max <= 1.")

        self.g_dag = g_dag
        self.g_x = g_x
        self.g_aleatoric = g_aleatoric
        self.num_roots = int(num_roots)
        self.num_layers = int(num_layers)
        self.max_nodes_per_layer = int(max_nodes_per_layer)
        self.latent_dim = int(latent_dim)
        self.latent_noise_scale = float(latent_noise_scale)

        raw_edge_prob = _sample_beta_scalar(
            edge_beta_alpha,
            edge_beta_beta,
            generator=self.g_dag,
            device=self.device,
        )
        self.task_edge_prob = edge_prob_min + (
            edge_prob_max - edge_prob_min
        ) * raw_edge_prob

        self.widths = self._sample_widths()
        self.connections: list[LatentLayerConnection] = []
        for layer_idx in range(self.num_layers - 1):
            self.connections.append(
                LatentLayerConnection(
                    in_width=self.widths[layer_idx],
                    out_width=self.widths[layer_idx + 1],
                    latent_dim=self.latent_dim,
                    edge_prob=self.task_edge_prob,
                    min_parents_per_node=min_parents_per_node,
                    generator=self.g_dag,
                    device=self.device,
                    linear_activation_prob=linear_activation_prob,
                    small_mlp_prob=small_mlp_prob,
                    soft_tree_prob=soft_tree_prob,
                    small_mlp_hidden_dim=small_mlp_hidden_dim,
                    soft_tree_depth=soft_tree_depth,
                    soft_tree_temperature=soft_tree_temperature,
                )
            )

    def _sample_widths(self) -> list[int]:
        widths = [self.num_roots]
        for _ in range(self.num_layers - 1):
            widths.append(
                int(
                    _randint(
                        5,
                        self.max_nodes_per_layer + 1,
                        (),
                        generator=self.g_dag,
                        device=self.device,
                    ).item()
                )
            )
        return widths

    def sample_root_latents(self, n_samples: int) -> list[torch.Tensor]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        return [
            _randn(
                n_samples,
                self.latent_dim,
                generator=self.g_x,
                device=self.device,
            )
            for _ in range(self.num_roots)
        ]

    def forward(
        self,
        root_latents: Optional[list[torch.Tensor]] = None,
        n_samples: Optional[int] = None,
        latent_noise_scale: Optional[float] = None,
    ) -> list[list[torch.Tensor]]:
        if root_latents is None:
            if n_samples is None:
                raise ValueError("Either root_latents or n_samples must be provided.")
            current = self.sample_root_latents(n_samples)
        else:
            current = root_latents

        noise_scale = (
            self.latent_noise_scale
            if latent_noise_scale is None
            else float(latent_noise_scale)
        )
        all_latents = [current]
        for connection in self.connections:
            current = connection(
                current,
                generator=self.g_aleatoric,
                latent_noise_scale=noise_scale,
            )
            all_latents.append(current)
        return all_latents

    def reforward_after_intervention(
        self,
        all_latents: list[list[torch.Tensor]],
        start_layer: int,
        latent_noise_scale: float = 0.0,
    ) -> list[list[torch.Tensor]]:
        if not 0 <= start_layer < self.num_layers:
            raise ValueError("Invalid start_layer.")

        new_latents = [list(layer) for layer in all_latents]
        current = new_latents[start_layer]
        for layer_idx in range(start_layer, self.num_layers - 1):
            current = self.connections[layer_idx](
                current,
                generator=self.g_aleatoric,
                latent_noise_scale=latent_noise_scale,
            )
            new_latents[layer_idx + 1] = current
        return new_latents

    def describe(self) -> None:
        print("========== RandomLayeredLatentSCM ==========")
        print(f"widths: {self.widths}")
        print(f"latent_dim: {self.latent_dim}")
        print(f"task_edge_prob: {self.task_edge_prob:.4f}")
        print()

        for layer_idx, connection in enumerate(self.connections):
            print(f"Connection layer {layer_idx} -> layer {layer_idx + 1}:")
            print(connection.adj.long())
            print(f"num_edges = {int(connection.adj.sum().item())}")
            for parent_idx in range(connection.in_width):
                for child_idx in range(connection.out_width):
                    edge = connection.edges[parent_idx][child_idx]
                    if edge is not None:
                        print(f"  edge {parent_idx}->{child_idx}: {edge.name()}")
            print()


# ---------------------------------------------------------------------------
# Data-driven feature observation
# ---------------------------------------------------------------------------


from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class FeatureObservation:
    """Result of observing one selected latent node."""

    # Final table column:
    # - float score for continuous features
    # - integer labels for categorical features
    values: torch.Tensor

    is_categorical: bool

    # Number of categories; 0 for continuous features.
    cardinality: int

    # Underlying one-dimensional continuous observation score.
    score: torch.Tensor

    # BIC(K=1) - BIC(best K).
    # Larger means stronger evidence for multiple clusters.
    cluster_score: float

    # Sorted Gaussian-mixture means. Empty for continuous features.
    cluster_centers: torch.Tensor

    projection_index: int


@dataclass(frozen=True)
class _GaussianMixtureResult:
    """Internal result for a fitted one-dimensional Gaussian mixture."""

    num_components: int
    means: torch.Tensor
    variances: torch.Tensor
    weights: torch.Tensor
    responsibilities: torch.Tensor
    labels: torch.Tensor
    log_likelihood: float
    bic: float


class AdaptiveObservationHead:
    """
    Convert one selected latent node into one table column.

    Procedure
    ---------
    1. Project latent vectors [N, latent_dim] into a scalar score [N].
    2. Standardize the score.
    3. Fit one-dimensional Gaussian mixtures with K=1,...,K_max.
    4. Compare the models using BIC.
    5. Return a categorical column only when:
       - a K >= 2 model improves sufficiently over K=1;
       - every inferred category has enough samples;
       - neighboring mixture components are sufficiently separated;
       - component weights are not degenerate.
    6. Otherwise return the score as a continuous column.

    No feature type or category count is sampled before observing the data.
    """

    def __init__(
        self,
        latent_dim: int,
        generator: torch.Generator,
        device: torch.device,
        max_cardinality: int = 10,
        min_samples_per_category: int = 8,
        min_bic_improvement: float = 10.0,
        min_cluster_separation: float = 1.5,
        min_component_weight: float = 0.02,
        num_em_restarts: int = 5,
        max_em_iterations: int = 100,
        em_tolerance: float = 1e-5,
        variance_floor: float = 1e-3,
        observation_noise_scale: float = 0.05,
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if max_cardinality < 2:
            raise ValueError("max_cardinality must be at least 2.")
        if min_samples_per_category < 2:
            raise ValueError("min_samples_per_category must be at least 2.")
        if min_bic_improvement < 0:
            raise ValueError("min_bic_improvement must be nonnegative.")
        if min_cluster_separation < 0:
            raise ValueError("min_cluster_separation must be nonnegative.")
        if not 0.0 < min_component_weight < 1.0:
            raise ValueError("min_component_weight must be in (0, 1).")
        if num_em_restarts < 1:
            raise ValueError("num_em_restarts must be at least 1.")
        if max_em_iterations < 1:
            raise ValueError("max_em_iterations must be at least 1.")
        if em_tolerance <= 0:
            raise ValueError("em_tolerance must be positive.")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive.")
        if observation_noise_scale < 0:
            raise ValueError("observation_noise_scale must be nonnegative.")

        self.latent_dim = int(latent_dim)
        self.device = device
        self.max_cardinality = int(max_cardinality)
        self.min_samples_per_category = int(min_samples_per_category)
        self.min_bic_improvement = float(min_bic_improvement)
        self.min_cluster_separation = float(min_cluster_separation)
        self.min_component_weight = float(min_component_weight)
        self.num_em_restarts = int(num_em_restarts)
        self.max_em_iterations = int(max_em_iterations)
        self.em_tolerance = float(em_tolerance)
        self.variance_floor = float(variance_floor)
        self.observation_noise_scale = float(observation_noise_scale)

        # Random measurement direction. This belongs to the observation
        # mechanism and does not affect latent SCM propagation.
        self.W = (latent_dim**-0.5) * torch.randn(
            latent_dim,
            generator=generator,
            device=device,
        )
        self.b = torch.randn(
            (),
            generator=generator,
            device=device,
        )

    @staticmethod
    def _standardize(
        value: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        mean = value.mean()
        std = value.std(unbiased=False).clamp_min(eps)
        return (value - mean) / std

    def _make_score(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent [N, {self.latent_dim}], "
                f"got {tuple(latent.shape)}."
            )

        if latent.shape[0] < 2:
            raise ValueError("At least two samples are required.")

        score = latent.float() @ self.W + self.b

        if self.observation_noise_scale > 0:
            score = score + self.observation_noise_scale * torch.randn(
                score.shape,
                generator=generator,
                device=score.device,
                dtype=score.dtype,
            )

        return self._standardize(score)

    @staticmethod
    def _normal_log_probability(
        x: torch.Tensor,
        means: torch.Tensor,
        variances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log N(x_n | mean_k, variance_k).

        Parameters
        ----------
        x:
            Shape [N].
        means:
            Shape [K].
        variances:
            Shape [K].

        Returns
        -------
        Tensor of shape [N, K].
        """
        log_two_pi = torch.log(
            torch.tensor(
                2.0 * torch.pi,
                device=x.device,
                dtype=x.dtype,
            )
        )

        return -0.5 * (
            log_two_pi
            + torch.log(variances)[None, :]
            + (x[:, None] - means[None, :]).square()
            / variances[None, :]
        )

    def _initialize_parameters(
        self,
        score: torch.Tensor,
        num_components: int,
        generator: torch.Generator,
        restart_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize a one-dimensional Gaussian mixture.

        The first restart uses quantiles. Later restarts add randomized
        perturbations so that EM is less likely to remain in a poor optimum.
        """
        n = int(score.numel())
        sorted_score, _ = torch.sort(score)

        quantile_positions = torch.linspace(
            0,
            n - 1,
            steps=num_components + 2,
            device=score.device,
        )[1:-1]

        indices = quantile_positions.round().long().clamp(0, n - 1)
        means = sorted_score[indices].clone()

        if restart_index > 0:
            scale = score.std(unbiased=False).clamp_min(1e-3)
            means = means + 0.15 * scale * torch.randn(
                means.shape,
                generator=generator,
                device=score.device,
                dtype=score.dtype,
            )

        global_variance = (
            score.var(unbiased=False)
            .clamp_min(self.variance_floor)
        )

        variances = global_variance.repeat(num_components)
        weights = torch.full(
            (num_components,),
            1.0 / num_components,
            device=score.device,
            dtype=score.dtype,
        )

        return means, variances, weights

    def _fit_gmm_once(
        self,
        score: torch.Tensor,
        num_components: int,
        generator: torch.Generator,
        restart_index: int,
    ) -> _GaussianMixtureResult:
        means, variances, weights = self._initialize_parameters(
            score=score,
            num_components=num_components,
            generator=generator,
            restart_index=restart_index,
        )

        previous_log_likelihood: Optional[float] = None
        responsibilities: Optional[torch.Tensor] = None

        for _ in range(self.max_em_iterations):
            # E-step
            component_log_prob = self._normal_log_probability(
                score,
                means,
                variances,
            )

            weighted_log_prob = (
                component_log_prob
                + torch.log(weights.clamp_min(1e-12))[None, :]
            )

            log_normalizer = torch.logsumexp(
                weighted_log_prob,
                dim=1,
                keepdim=True,
            )

            responsibilities = torch.exp(
                weighted_log_prob - log_normalizer
            )

            log_likelihood = float(log_normalizer.sum().item())

            # M-step
            effective_counts = responsibilities.sum(dim=0).clamp_min(1e-6)

            weights = effective_counts / effective_counts.sum()

            means = (
                responsibilities * score[:, None]
            ).sum(dim=0) / effective_counts

            centered = score[:, None] - means[None, :]
            variances = (
                responsibilities * centered.square()
            ).sum(dim=0) / effective_counts

            variances = variances.clamp_min(self.variance_floor)

            if previous_log_likelihood is not None:
                improvement = log_likelihood - previous_log_likelihood

                if abs(improvement) <= self.em_tolerance * (
                    1.0 + abs(previous_log_likelihood)
                ):
                    break

            previous_log_likelihood = log_likelihood

        # Recompute after the final M-step.
        component_log_prob = self._normal_log_probability(
            score,
            means,
            variances,
        )

        weighted_log_prob = (
            component_log_prob
            + torch.log(weights.clamp_min(1e-12))[None, :]
        )

        log_normalizer = torch.logsumexp(
            weighted_log_prob,
            dim=1,
            keepdim=True,
        )

        responsibilities = torch.exp(
            weighted_log_prob - log_normalizer
        )

        log_likelihood = float(log_normalizer.sum().item())

        # One-dimensional GMM parameters:
        # K means + K variances + (K - 1) independent mixture weights.
        num_parameters = 3 * num_components - 1
        n = int(score.numel())

        bic = (
            num_parameters
            * float(torch.log(torch.tensor(float(n))).item())
            - 2.0 * log_likelihood
        )

        # Sort components by their means so category IDs are stable and ordered.
        mean_order = torch.argsort(means)
        means = means[mean_order]
        variances = variances[mean_order]
        weights = weights[mean_order]
        responsibilities = responsibilities[:, mean_order]

        labels = torch.argmax(responsibilities, dim=1)

        return _GaussianMixtureResult(
            num_components=num_components,
            means=means,
            variances=variances,
            weights=weights,
            responsibilities=responsibilities,
            labels=labels,
            log_likelihood=log_likelihood,
            bic=bic,
        )

    def _fit_gmm(
        self,
        score: torch.Tensor,
        num_components: int,
        generator: torch.Generator,
    ) -> _GaussianMixtureResult:
        best_result: Optional[_GaussianMixtureResult] = None

        num_restarts = (
            1 if num_components == 1 else self.num_em_restarts
        )

        for restart_index in range(num_restarts):
            result = self._fit_gmm_once(
                score=score,
                num_components=num_components,
                generator=generator,
                restart_index=restart_index,
            )

            if (
                best_result is None
                or result.log_likelihood > best_result.log_likelihood
            ):
                best_result = result

        if best_result is None:
            raise RuntimeError("Gaussian-mixture fitting failed.")

        return best_result

    def _minimum_component_separation(
        self,
        result: _GaussianMixtureResult,
    ) -> float:
        """
        Compute the minimum standardized separation between adjacent means.

        For neighboring components i and j:

            separation =
                |mu_j - mu_i| /
                sqrt(0.5 * (variance_i + variance_j))

        Larger values indicate more clearly separated components.
        """
        if result.num_components < 2:
            return 0.0

        mean_differences = (
            result.means[1:] - result.means[:-1]
        ).abs()

        pooled_std = torch.sqrt(
            0.5
            * (
                result.variances[1:]
                + result.variances[:-1]
            )
        ).clamp_min(1e-6)

        separation = mean_differences / pooled_std
        return float(separation.min().item())

    def _is_valid_categorical_model(
        self,
        result: _GaussianMixtureResult,
    ) -> bool:
        if result.num_components < 2:
            return False

        counts = torch.bincount(
            result.labels,
            minlength=result.num_components,
        )

        if int(counts.min().item()) < self.min_samples_per_category:
            return False

        if float(result.weights.min().item()) < self.min_component_weight:
            return False

        separation = self._minimum_component_separation(result)

        if separation < self.min_cluster_separation:
            return False

        return True

    def _infer_clusters(
        self,
        score: torch.Tensor,
        generator: torch.Generator,
    ) -> tuple[
        Optional[torch.Tensor],
        torch.Tensor,
        float,
    ]:
        """
        Infer whether the score has reliable discrete cluster structure.

        Returns
        -------
        labels:
            Shape [N] if categorical, otherwise None.
        centers:
            Shape [K] if categorical, otherwise an empty tensor.
        cluster_score:
            BIC(K=1) - BIC(best valid K).
        """
        n = int(score.numel())

        max_k_by_sample_size = (
            n // self.min_samples_per_category
        )

        max_k = min(
            self.max_cardinality,
            max_k_by_sample_size,
        )

        if max_k < 2:
            return (
                None,
                torch.empty(
                    0,
                    device=score.device,
                    dtype=score.dtype,
                ),
                0.0,
            )

        single_component = self._fit_gmm(
            score=score,
            num_components=1,
            generator=generator,
        )

        best_categorical: Optional[_GaussianMixtureResult] = None

        for num_components in range(2, max_k + 1):
            candidate = self._fit_gmm(
                score=score,
                num_components=num_components,
                generator=generator,
            )

            if not self._is_valid_categorical_model(candidate):
                continue

            if (
                best_categorical is None
                or candidate.bic < best_categorical.bic
            ):
                best_categorical = candidate

        if best_categorical is None:
            return (
                None,
                torch.empty(
                    0,
                    device=score.device,
                    dtype=score.dtype,
                ),
                0.0,
            )

        bic_improvement = (
            single_component.bic - best_categorical.bic
        )

        if bic_improvement < self.min_bic_improvement:
            return (
                None,
                torch.empty(
                    0,
                    device=score.device,
                    dtype=score.dtype,
                ),
                float(bic_improvement),
            )

        return (
            best_categorical.labels,
            best_categorical.means,
            float(bic_improvement),
        )

    def observe(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> FeatureObservation:
        score = self._make_score(
            latent,
            generator=generator,
        )

        labels, centers, cluster_score = self._infer_clusters(
            score,
            generator=generator,
        )

        if labels is None:
            return FeatureObservation(
                values=score,
                is_categorical=False,
                cardinality=0,
                score=score,
                cluster_score=cluster_score,
                cluster_centers=torch.empty(
                    0,
                    device=score.device,
                    dtype=score.dtype,
                ),
            )

        cardinality = int(centers.numel())

        # Final defensive validation.
        counts = torch.bincount(
            labels,
            minlength=cardinality,
        )

        if int(counts.min().item()) < self.min_samples_per_category:
            return FeatureObservation(
                values=score,
                is_categorical=False,
                cardinality=0,
                score=score,
                cluster_score=cluster_score,
                cluster_centers=torch.empty(
                    0,
                    device=score.device,
                    dtype=score.dtype,
                ),
            )

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=cardinality,
            score=score,
            cluster_score=cluster_score,
            cluster_centers=centers,
        )


class TargetObservationHead:
    """Task-controlled target observation independent of feature typing."""

    def __init__(
        self,
        latent_dim: int,
        generator: torch.Generator,
        device: torch.device,
        observation_noise_scale: float = 0.05,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.device = device
        self.observation_noise_scale = float(observation_noise_scale)
        self.W = (latent_dim ** -0.5) * _randn(
            latent_dim,
            generator=generator,
            device=device,
        )
        self.b = _randn((), generator=generator, device=device)

    def score(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        value = latent.float() @ self.W + self.b
        if self.observation_noise_scale > 0:
            value = value + self.observation_noise_scale * torch.randn(
                value.shape,
                generator=generator,
                device=self.device,
                dtype=value.dtype,
            )
        return _standardize(value, dim=0)

    @staticmethod
    def balanced_classes(score: torch.Tensor, num_classes: int) -> torch.Tensor:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        n = int(score.numel())
        if n < num_classes:
            raise ValueError("n_samples must be at least num_classes.")

        order = torch.argsort(score)
        labels = torch.empty(n, device=score.device, dtype=torch.long)

        base = n // num_classes
        remainder = n % num_classes
        start = 0
        for class_id in range(num_classes):
            size = base + (1 if class_id < remainder else 0)
            labels[order[start : start + size]] = class_id
            start += size
        return labels


# ---------------------------------------------------------------------------
# Tabular task wrapper
# ---------------------------------------------------------------------------



class MixedLatentSCMTask(GenerateTask):
    CONTINUOUS = 0
    CATEGORICAL = 1

    def __init__(
        self,
        num_classes: Optional[int] = None,
        n_max: int = 500,
        d_max: int = 20,
        n_min: int = 128,
        d_min: int = 2,
        test_frac: float = 0.15,
        p_missing: float = 0.05,
        latent_noise_scale: float = 0.05,
        observation_noise_scale: float = 0.05,
        device: Optional[torch.device] = None,
        dag_seed: Optional[int] = None,
        aleatoric_seed: Optional[int] = None,
        x_seed: Optional[int] = None,
        num_roots: int = 3,
        num_layers: int = 4,
        max_nodes_per_layer: int = 8,
        latent_dim: int = 8,
        edge_beta_alpha: float = 2.0,
        edge_beta_beta: float = 5.0,
        edge_prob_min: float = 0.05,
        edge_prob_max: float = 0.95,
        min_parents_per_node: int = 1,
        max_cardinality: int = 10,
        min_samples_per_category: int = 8,
        min_bic_improvement: float = 10.0,
        min_cluster_separation: float = 1.5,
        min_component_weight: float = 0.02,
        num_em_restarts: int = 5,
        max_em_iterations: int = 100,
        em_tolerance: float = 1e-5,
        variance_floor: float = 1e-3,
        linear_activation_prob: float = 0.60,
        small_mlp_prob: float = 0.25,
        soft_tree_prob: float = 0.15,
        small_mlp_hidden_dim: Optional[int] = None,
        soft_tree_depth: int = 2,
        soft_tree_temperature: float = 0.5,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.num_classes = num_classes
        self.n_max = int(n_max)
        self.d_max = int(d_max)
        self.n_min = int(n_min)
        self.d_min = int(d_min)
        self.test_frac = float(test_frac)
        self.p_missing = float(p_missing)
        self.latent_noise_scale = float(latent_noise_scale)
        self.observation_noise_scale = float(observation_noise_scale)

        self.num_roots = int(num_roots)
        self.num_layers = int(num_layers)
        self.max_nodes_per_layer = int(max_nodes_per_layer)
        self.latent_dim = int(latent_dim)
        self.edge_beta_alpha = float(edge_beta_alpha)
        self.edge_beta_beta = float(edge_beta_beta)
        self.edge_prob_min = float(edge_prob_min)
        self.edge_prob_max = float(edge_prob_max)
        self.min_parents_per_node = int(min_parents_per_node)

        # GMM-based categorical inference settings.
        self.max_cardinality = int(max_cardinality)
        self.min_samples_per_category = int(min_samples_per_category)
        self.min_bic_improvement = float(min_bic_improvement)
        self.min_cluster_separation = float(min_cluster_separation)
        self.min_component_weight = float(min_component_weight)
        self.num_em_restarts = int(num_em_restarts)
        self.max_em_iterations = int(max_em_iterations)
        self.em_tolerance = float(em_tolerance)
        self.variance_floor = float(variance_floor)

        self.linear_activation_prob = float(linear_activation_prob)
        self.small_mlp_prob = float(small_mlp_prob)
        self.soft_tree_prob = float(soft_tree_prob)
        self.small_mlp_hidden_dim = small_mlp_hidden_dim
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_temperature = float(soft_tree_temperature)

        self.g_dag, self.dag_seed = make_gen(
            self.device,
            dag_seed,
        )
        self.g_aleatoric, self.aleatoric_seed = make_gen(
            self.device,
            aleatoric_seed,
        )
        self.g_x, self.x_seed = make_gen(
            self.device,
            x_seed,
        )

        self.d = int(
            _randint(
                self.d_min,
                self.d_max + 1,
                (1,),
                generator=self.g_dag,
                device=self.device,
            ).item()
        )

        self.n = int(
            _randint(
                self.n_min,
                self.n_max + 1,
                (1,),
                generator=self.g_dag,
                device=self.device,
            ).item()
        )

        super().__init__()

    @staticmethod
    def _flatten_latents(
        all_latents: list[list[torch.Tensor]],
    ) -> tuple[
        list[torch.Tensor],
        list[tuple[int, int]],
    ]:
        flat_latents: list[torch.Tensor] = []
        flat_index: list[tuple[int, int]] = []

        for layer_idx, layer in enumerate(all_latents):
            for node_idx, latent in enumerate(layer):
                flat_latents.append(latent)
                flat_index.append((layer_idx, node_idx))

        return flat_latents, flat_index

    def _sample_feature_and_target_sources(
        self,
        flat_index: list[tuple[int, int]],
        d: int,
    ) -> tuple[list[int], int]:
        """
        Choose the target from the final layer and features from earlier layers.
        """
        max_layer = max(
            layer_idx
            for layer_idx, _ in flat_index
        )

        target_candidates = [
            node_id
            for node_id, (layer_idx, _) in enumerate(flat_index)
            if layer_idx == max_layer
        ]

        target_id = target_candidates[
            int(
                _randint(
                    0,
                    len(target_candidates),
                    (),
                    generator=self.g_dag,
                    device=self.device,
                ).item()
            )
        ]

        target_layer, _ = flat_index[target_id]

        candidates = [
            node_id
            for node_id, (layer_idx, _) in enumerate(flat_index)
            if layer_idx < target_layer
        ]

        if len(candidates) < self.d_min:
            raise RuntimeError(
                f"Only {len(candidates)} feature candidates exist, "
                f"but d_min={self.d_min}."
            )

        d = min(
            int(d),
            len(candidates),
        )

        permutation = torch.randperm(
            len(candidates),
            generator=self.g_dag,
            device=self.device,
        )

        feature_ids = [
            candidates[int(i)]
            for i in permutation[:d].tolist()
        ]

        return feature_ids, target_id

    def _observe_features(
        self,
        flat_latents: list[torch.Tensor],
        feature_ids: list[int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[AdaptiveObservationHead],
    ]:
        """
        Observe selected latent nodes as table columns.

        Returns
        -------
        X:
            Observed feature matrix of shape [N, d].
        feature_type:
            Per-feature type:
            0 for continuous and 1 for categorical.
        cardinality:
            Number of categories for each feature, or 0 for continuous.
        cluster_score:
            BIC(K=1) - BIC(best valid categorical model).
        cluster_centers:
            Sorted GMM means for categorical features. Empty tensors for
            continuous features.
        heads:
            Observation heads used for each feature.
        """
        n = int(flat_latents[0].shape[0])
        d = len(feature_ids)

        X = torch.empty(
            n,
            d,
            device=self.device,
            dtype=torch.float32,
        )

        feature_type = torch.empty(
            d,
            device=self.device,
            dtype=torch.long,
        )

        cardinality = torch.zeros(
            d,
            device=self.device,
            dtype=torch.long,
        )

        cluster_score = torch.zeros(
            d,
            device=self.device,
            dtype=torch.float32,
        )

        cluster_centers: list[torch.Tensor] = []
        heads: list[AdaptiveObservationHead] = []

        for col, node_id in enumerate(feature_ids):
            head = AdaptiveObservationHead(
                latent_dim=self.latent_dim,
                generator=self.g_dag,
                device=self.device,
                max_cardinality=self.max_cardinality,
                min_samples_per_category=self.min_samples_per_category,
                min_bic_improvement=self.min_bic_improvement,
                min_cluster_separation=self.min_cluster_separation,
                min_component_weight=self.min_component_weight,
                num_em_restarts=self.num_em_restarts,
                max_em_iterations=self.max_em_iterations,
                em_tolerance=self.em_tolerance,
                variance_floor=self.variance_floor,
                observation_noise_scale=self.observation_noise_scale,
            )

            observed = head.observe(
                flat_latents[node_id],
                generator=self.g_aleatoric,
            )

            # Categorical labels are integer-valued, but X is stored as float
            # because continuous and categorical columns share one tensor.
            X[:, col] = observed.values.float()

            feature_type[col] = (
                self.CATEGORICAL
                if observed.is_categorical
                else self.CONTINUOUS
            )

            cardinality[col] = observed.cardinality
            cluster_score[col] = observed.cluster_score

            cluster_centers.append(
                observed.cluster_centers
            )
            heads.append(head)

        return (
            X,
            feature_type,
            cardinality,
            cluster_score,
            cluster_centers,
            heads,
        )

    def _compute_importance_ground_truth(
        self,
        feature_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        d = len(feature_ids)

        nan = torch.full(
            (d,),
            float("nan"),
            device=self.device,
            dtype=torch.float32,
        )

        return {
            "feature_strength": nan.clone(),
            "importance_ratio": nan.clone(),
            "is_active": nan.clone(),
            "sampled_active": nan.clone(),
        }

    def _generate(self):
        device = self.device
        n, d = self.n, self.d

        scm = RandomLayeredLatentSCM(
            g_dag=self.g_dag,
            g_x=self.g_x,
            g_aleatoric=self.g_aleatoric,
            num_roots=self.num_roots,
            num_layers=self.num_layers,
            max_nodes_per_layer=self.max_nodes_per_layer,
            latent_dim=self.latent_dim,
            edge_beta_alpha=self.edge_beta_alpha,
            edge_beta_beta=self.edge_beta_beta,
            edge_prob_min=self.edge_prob_min,
            edge_prob_max=self.edge_prob_max,
            min_parents_per_node=self.min_parents_per_node,
            latent_noise_scale=self.latent_noise_scale,
            linear_activation_prob=self.linear_activation_prob,
            small_mlp_prob=self.small_mlp_prob,
            soft_tree_prob=self.soft_tree_prob,
            small_mlp_hidden_dim=self.small_mlp_hidden_dim,
            soft_tree_depth=self.soft_tree_depth,
            soft_tree_temperature=self.soft_tree_temperature,
            device=device,
        )

        all_latents = scm.forward(
            n_samples=n,
            latent_noise_scale=self.latent_noise_scale,
        )

        flat_latents, flat_index = self._flatten_latents(
            all_latents
        )

        feature_ids, target_id = (
            self._sample_feature_and_target_sources(
                flat_index,
                d,
            )
        )

        self.d = len(feature_ids)

        (
            X_clean,
            feature_type,
            cardinality,
            cluster_score,
            feature_cluster_centers,
            feature_observation_heads,
        ) = self._observe_features(
            flat_latents,
            feature_ids,
        )

        # The target head is constructed after the SCM so that target task type
        # does not influence DAG sampling or latent edge parameters.
        target_head = TargetObservationHead(
            latent_dim=self.latent_dim,
            generator=self.g_dag,
            device=device,
            observation_noise_scale=self.observation_noise_scale,
        )

        target_score = target_head.score(
            flat_latents[target_id],
            generator=self.g_aleatoric,
        )

        if self.num_classes is None:
            y = target_score
            self.n_classes = None
        else:
            num_classes = int(self.num_classes)

            y = target_head.balanced_classes(
                target_score,
                num_classes,
            )

            self.n_classes = num_classes

        importance_info = self._compute_importance_ground_truth(
            feature_ids
        )

        X_obs = X_clean.clone()

        missing_mask = (
            _rand(
                *X_obs.shape,
                generator=self.g_x,
                device=device,
            )
            < self.p_missing
        )

        X_obs[missing_mask] = torch.nan

        if self.num_classes is not None:
            train_idx, test_idx = stratified_classification_split(
                y=y.long(),
                test_frac=self.test_frac,
                generator=self.g_x,
                device=device,
            )
        else:
            n_test = max(
                1,
                int(round(n * self.test_frac)),
            )

            # Preserve at least two training samples.
            n_test = min(
                n_test,
                n - 2,
            )

            permutation = torch.randperm(
                n,
                device=device,
                generator=self.g_x,
            )

            train_idx = permutation[:-n_test]
            test_idx = permutation[-n_test:]

        X_train = X_obs[train_idx]
        y_train = y[train_idx]
        X_test = X_obs[test_idx]
        y_test = y[test_idx]

        info = {
            "feature_type": feature_type,
            "cardinality": cardinality,

            # Positive values indicate that a valid multi-component model
            # fits better than the single-Gaussian model.
            "categorical_cluster_score": cluster_score,

            # List of length d:
            # - categorical feature: tensor of shape [K]
            # - continuous feature: empty tensor
            "feature_cluster_centers": feature_cluster_centers,

            **importance_info,

            "missing_mask_train": missing_mask[train_idx],
            "missing_mask_test": missing_mask[test_idx],

            "feature_ids": torch.tensor(
                feature_ids,
                device=device,
                dtype=torch.long,
            ),

            "target_id": torch.tensor(
                target_id,
                device=device,
                dtype=torch.long,
            ),

            "task_edge_prob": torch.tensor(
                scm.task_edge_prob,
                device=device,
                dtype=torch.float32,
            ),

            "latent_dim": torch.tensor(
                self.latent_dim,
                device=device,
                dtype=torch.long,
            ),
        }

        self.n_features = self.d
        self.feature_type = feature_type
        self.cardinality = cardinality
        self.cluster_score = cluster_score
        self.feature_cluster_centers = feature_cluster_centers

        self.scm = scm
        self.feature_observation_heads = feature_observation_heads
        self.target_observation_head = target_head

        return (
            X_train,
            y_train,
            X_test,
            y_test,
            info,
        )

    def visualize(self):
        return None

    def forward(
        self,
        X: torch.Tensor,
    ):
        del X
        return None

# RandomLayeredSCM = RandomLayeredLatentSCM
# MixedSCMTask = MixedLatentSCMTask

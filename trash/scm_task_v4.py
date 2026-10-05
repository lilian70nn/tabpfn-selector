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

@dataclass(frozen=True)
class FeatureObservation:
    """
    Result of observing one selected latent node.

    Continuous:
        values is a standardized scalar column [N].

    Categorical:
        values contains cluster labels [N].
    """

    values: torch.Tensor
    is_categorical: bool
    cardinality: int

    # Continuous observation score.
    # Empty for categorical features.
    score: torch.Tensor

    # Quality of the selected clustering.
    # Larger means more clearly separated clusters.
    cluster_score: float

    # Latent-space cluster centers [K, h].
    # Empty for continuous features.
    cluster_centers: torch.Tensor


@dataclass(frozen=True)
class _KMeansResult:
    """Internal result of one latent-space K-means fit."""

    num_clusters: int
    centers: torch.Tensor      # [K, h]
    labels: torch.Tensor       # [N]
    distances: torch.Tensor    # [N, K]
    inertia: float
    cluster_score: float


class AdaptiveObservationHead:
    """
    Observe a latent node as either categorical or continuous.

    Procedure
    ---------
    1. Standardize the latent representation dimension-wise.
    2. Fit K-means directly in latent space for K=2,...,K_max.
    3. Validate each candidate using:
       - minimum samples per cluster;
       - minimum cluster proportion;
       - cluster-center separation;
       - silhouette-like clustering score.
    4. If a valid clustering exists:
       return its labels directly as a categorical feature.
    5. Otherwise:
       project latent [N, h] to a scalar and return a continuous feature.

    Node type and cardinality are not decided before seeing the latent data.
    """

    def __init__(
        self,
        latent_dim: int,
        generator: torch.Generator,
        device: torch.device,
        max_cardinality: int = 10,
        min_samples_per_category: int = 8,
        min_component_weight: float = 0.02,
        min_cluster_separation: float = 1.5,
        min_cluster_score: float = 0.25,
        num_kmeans_restarts: int = 5,
        max_kmeans_iterations: int = 100,
        kmeans_tolerance: float = 1e-4,
        observation_noise_scale: float = 0.05,
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if max_cardinality < 2:
            raise ValueError("max_cardinality must be at least 2.")
        if min_samples_per_category < 2:
            raise ValueError("min_samples_per_category must be at least 2.")
        if not 0.0 < min_component_weight < 1.0:
            raise ValueError("min_component_weight must lie in (0, 1).")
        if min_cluster_separation < 0:
            raise ValueError("min_cluster_separation must be nonnegative.")
        if not -1.0 <= min_cluster_score <= 1.0:
            raise ValueError("min_cluster_score must lie in [-1, 1].")
        if num_kmeans_restarts < 1:
            raise ValueError("num_kmeans_restarts must be at least 1.")
        if max_kmeans_iterations < 1:
            raise ValueError("max_kmeans_iterations must be at least 1.")
        if kmeans_tolerance <= 0:
            raise ValueError("kmeans_tolerance must be positive.")
        if observation_noise_scale < 0:
            raise ValueError("observation_noise_scale must be nonnegative.")

        self.latent_dim = int(latent_dim)
        self.device = device

        self.max_cardinality = int(max_cardinality)
        self.min_samples_per_category = int(min_samples_per_category)
        self.min_component_weight = float(min_component_weight)
        self.min_cluster_separation = float(min_cluster_separation)
        self.min_cluster_score = float(min_cluster_score)

        self.num_kmeans_restarts = int(num_kmeans_restarts)
        self.max_kmeans_iterations = int(max_kmeans_iterations)
        self.kmeans_tolerance = float(kmeans_tolerance)
        self.observation_noise_scale = float(observation_noise_scale)

        # Used only when the latent is observed as continuous.
        self.continuous_W = (latent_dim ** -0.5) * _randn(
            latent_dim,
            generator=generator,
            device=device,
        )
        self.continuous_b = _randn(
            (),
            generator=generator,
            device=device,
        )

    def _prepare_latent(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError(
                f"Expected latent [N, h], got {tuple(latent.shape)}."
            )

        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, "
                f"got {latent.shape[1]}."
            )

        if latent.shape[0] < 2:
            raise ValueError("At least two samples are required.")

        # Standardize every latent dimension across samples.
        return _standardize(
            latent.float(),
            dim=0,
        )

    def _make_continuous_score(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Produce one continuous observed feature from latent [N, h].
        """
        score = latent @ self.continuous_W + self.continuous_b

        if self.observation_noise_scale > 0:
            score = score + self.observation_noise_scale * torch.randn(
                score.shape,
                generator=generator,
                device=score.device,
                dtype=score.dtype,
            )

        return _standardize(
            score,
            dim=0,
        )

    def _initialize_kmeans_centers(
        self,
        latent: torch.Tensor,
        num_clusters: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        K-means++-style initialization.

        The first center is random. Every later center is sampled with
        probability proportional to squared distance from the closest
        existing center.
        """
        n = latent.shape[0]

        first_idx = int(
            _randint(
                0,
                n,
                (),
                generator=generator,
                device=latent.device,
            ).item()
        )

        centers = [latent[first_idx]]

        for _ in range(1, num_clusters):
            current_centers = torch.stack(
                centers,
                dim=0,
            )

            squared_distances = torch.cdist(
                latent,
                current_centers,
            ).square()

            nearest_squared_distance = squared_distances.min(
                dim=1
            ).values

            total = nearest_squared_distance.sum()

            if float(total.item()) <= 1e-12:
                # Degenerate fallback: pick a random sample.
                idx = int(
                    _randint(
                        0,
                        n,
                        (),
                        generator=generator,
                        device=latent.device,
                    ).item()
                )
            else:
                probabilities = (
                    nearest_squared_distance
                    / total
                )

                idx = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        replacement=True,
                        generator=generator,
                    ).item()
                )

            centers.append(latent[idx])

        return torch.stack(
            centers,
            dim=0,
        )

    def _compute_cluster_score(
        self,
        distances: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Compute a silhouette-like score.

        For each sample:
            a = distance to assigned center
            b = distance to closest other center

            score = (b - a) / max(a, b)

        Range is approximately [-1, 1].
        Larger values mean samples are much closer to their assigned center
        than to competing centers.
        """
        assigned_distance = distances.gather(
            1,
            labels[:, None],
        ).squeeze(1)

        other_distances = distances.clone()
        other_distances.scatter_(
            1,
            labels[:, None],
            float("inf"),
        )

        nearest_other_distance = other_distances.min(
            dim=1
        ).values

        denominator = torch.maximum(
            assigned_distance,
            nearest_other_distance,
        ).clamp_min(1e-6)

        sample_scores = (
            nearest_other_distance - assigned_distance
        ) / denominator

        return float(
            sample_scores.mean().item()
        )

    def _fit_kmeans_once(
        self,
        latent: torch.Tensor,
        num_clusters: int,
        generator: torch.Generator,
    ) -> _KMeansResult:
        centers = self._initialize_kmeans_centers(
            latent=latent,
            num_clusters=num_clusters,
            generator=generator,
        )

        previous_inertia: Optional[float] = None

        for _ in range(self.max_kmeans_iterations):
            distances = torch.cdist(
                latent,
                centers,
            )

            labels = torch.argmin(
                distances,
                dim=1,
            )

            new_centers = []

            for cluster_id in range(num_clusters):
                mask = labels == cluster_id

                if bool(mask.any()):
                    new_center = latent[mask].mean(
                        dim=0
                    )
                else:
                    # Empty cluster: reinitialize it to a random sample.
                    idx = int(
                        _randint(
                            0,
                            latent.shape[0],
                            (),
                            generator=generator,
                            device=latent.device,
                        ).item()
                    )
                    new_center = latent[idx]

                new_centers.append(new_center)

            new_centers_tensor = torch.stack(
                new_centers,
                dim=0,
            )

            assigned_squared_distance = (
                latent - new_centers_tensor[labels]
            ).square().sum(dim=1)

            inertia = float(
                assigned_squared_distance.sum().item()
            )

            center_shift = (
                new_centers_tensor - centers
            ).square().sum().sqrt()

            centers = new_centers_tensor

            if float(center_shift.item()) <= self.kmeans_tolerance:
                break

            if previous_inertia is not None:
                improvement = previous_inertia - inertia

                if abs(improvement) <= self.kmeans_tolerance * (
                    1.0 + abs(previous_inertia)
                ):
                    break

            previous_inertia = inertia

        # Recompute final assignments using final centers.
        distances = torch.cdist(
            latent,
            centers,
        )

        labels = torch.argmin(
            distances,
            dim=1,
        )

        assigned_squared_distance = (
            latent - centers[labels]
        ).square().sum(dim=1)

        inertia = float(
            assigned_squared_distance.sum().item()
        )

        cluster_score = self._compute_cluster_score(
            distances=distances,
            labels=labels,
        )

        return _KMeansResult(
            num_clusters=num_clusters,
            centers=centers,
            labels=labels,
            distances=distances,
            inertia=inertia,
            cluster_score=cluster_score,
        )

    def _fit_kmeans(
        self,
        latent: torch.Tensor,
        num_clusters: int,
        generator: torch.Generator,
    ) -> _KMeansResult:
        """
        Fit the same K several times and keep the result with minimum inertia.
        """
        best_result: Optional[_KMeansResult] = None

        for _ in range(self.num_kmeans_restarts):
            result = self._fit_kmeans_once(
                latent=latent,
                num_clusters=num_clusters,
                generator=generator,
            )

            if (
                best_result is None
                or result.inertia < best_result.inertia
            ):
                best_result = result

        if best_result is None:
            raise RuntimeError("K-means fitting failed.")

        return best_result

    def _minimum_center_separation(
        self,
        result: _KMeansResult,
        latent: torch.Tensor,
    ) -> float:
        """
        Measure cluster-center distance relative to within-cluster spread.

        For every pair of clusters:
            center distance /
            sqrt(0.5 * (variance_i + variance_j))

        Returns the smallest pairwise separation.
        """
        cluster_variances = []

        for cluster_id in range(result.num_clusters):
            mask = result.labels == cluster_id

            if not bool(mask.any()):
                return 0.0

            centered = (
                latent[mask]
                - result.centers[cluster_id]
            )

            variance = centered.square().sum(
                dim=1
            ).mean()

            cluster_variances.append(
                variance
            )

        cluster_variances_tensor = torch.stack(
            cluster_variances
        ).clamp_min(1e-6)

        center_distances = torch.cdist(
            result.centers,
            result.centers,
        )

        separations = []

        for i in range(result.num_clusters):
            for j in range(i + 1, result.num_clusters):
                pooled_scale = torch.sqrt(
                    0.5
                    * (
                        cluster_variances_tensor[i]
                        + cluster_variances_tensor[j]
                    )
                ).clamp_min(1e-6)

                separations.append(
                    center_distances[i, j]
                    / pooled_scale
                )

        if not separations:
            return 0.0

        return float(
            torch.stack(separations).min().item()
        )

    def _is_valid_categorical_model(
        self,
        result: _KMeansResult,
        latent: torch.Tensor,
    ) -> bool:
        counts = torch.bincount(
            result.labels,
            minlength=result.num_clusters,
        )

        if int(counts.min().item()) < self.min_samples_per_category:
            return False

        proportions = counts.float() / counts.sum()

        if float(proportions.min().item()) < self.min_component_weight:
            return False

        if result.cluster_score < self.min_cluster_score:
            return False

        separation = self._minimum_center_separation(
            result=result,
            latent=latent,
        )

        if separation < self.min_cluster_separation:
            return False

        return True

    def _detect_latent_clusters(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> Optional[_KMeansResult]:
        """
        Detect the best valid clustering directly in latent space.
        """
        n = latent.shape[0]

        max_k_by_sample_size = (
            n // self.min_samples_per_category
        )

        max_k = min(
            self.max_cardinality,
            max_k_by_sample_size,
        )

        if max_k < 2:
            return None

        best_result: Optional[_KMeansResult] = None

        for num_clusters in range(2, max_k + 1):
            candidate = self._fit_kmeans(
                latent=latent,
                num_clusters=num_clusters,
                generator=generator,
            )

            if not self._is_valid_categorical_model(
                result=candidate,
                latent=latent,
            ):
                continue

            # Prefer the candidate with the strongest cluster quality.
            # If scores are nearly equal, prefer fewer categories.
            if best_result is None:
                best_result = candidate
                continue

            if candidate.cluster_score > best_result.cluster_score + 1e-6:
                best_result = candidate
            elif (
                abs(
                    candidate.cluster_score
                    - best_result.cluster_score
                )
                <= 1e-6
                and candidate.num_clusters < best_result.num_clusters
            ):
                best_result = candidate

        return best_result

    def observe(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> FeatureObservation:
        prepared_latent = self._prepare_latent(
            latent
        )

        clustering = self._detect_latent_clusters(
            latent=prepared_latent,
            generator=generator,
        )

        # The latent node itself contains reliable discrete regimes.
        if clustering is not None:
            return FeatureObservation(
                values=clustering.labels,
                is_categorical=True,
                cardinality=clustering.num_clusters,
                score=torch.empty(
                    0,
                    device=prepared_latent.device,
                    dtype=prepared_latent.dtype,
                ),
                cluster_score=clustering.cluster_score,
                cluster_centers=clustering.centers,
            )

        # No stable latent clustering: observe it as a continuous scalar.
        score = self._make_continuous_score(
            latent=prepared_latent,
            generator=generator,
        )

        return FeatureObservation(
            values=score,
            is_categorical=False,
            cardinality=0,
            score=score,
            cluster_score=0.0,
            cluster_centers=torch.empty(
                0,
                self.latent_dim,
                device=prepared_latent.device,
                dtype=prepared_latent.dtype,
            ),
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

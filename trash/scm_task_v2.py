# scm_task_latent.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math
import torch
import torch.nn.functional as F

from src.data.helper import (
    discretize_latent_random_bins,
    make_gen,
    stratified_classification_split,
)
from src.data.synthetic_task import GenerateTask


NodeKind = Literal["cont", "cat"]


@dataclass(frozen=True)
class NodeSpec:
    """
    Node metadata used only by the observation layer.

    Internal SCM propagation is identical for continuous and categorical nodes:
        latent state: [N, latent_dim]

    Observation:
        cont -> one standardized continuous column
        cat  -> one integer category-ID column
    """

    kind: NodeKind
    K: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == "cat":
            if self.K is None or self.K < 2:
                raise ValueError("Categorical nodes require K >= 2.")
        elif self.kind == "cont":
            if self.K is not None:
                raise ValueError("Continuous nodes must use K=None.")
        else:
            raise ValueError(f"Unknown node kind: {self.kind}")


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
    """
    Generator-aware Beta sampling via Gamma variables.

    torch.distributions.Beta.sample() does not accept a torch.Generator,
    so Gamma(alpha, 1) / (Gamma(alpha, 1) + Gamma(beta, 1)) is used.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive.")

    concentration_a = torch.tensor(alpha, device=device, dtype=torch.float32)
    concentration_b = torch.tensor(beta, device=device, dtype=torch.float32)

    x = torch._standard_gamma(concentration_a, generator=generator)
    y = torch._standard_gamma(concentration_b, generator=generator)
    return float((x / (x + y).clamp_min(1e-12)).item())


# ---------------------------------------------------------------------------
# Unified latent-space edges
# ---------------------------------------------------------------------------

class LatentEdge:
    """
    Unified edge:
        [N, h] -> [N, h]

    Edge-family prior:
        linear_activation: 0.60
        small_mlp:         0.25
        soft_tree:         0.15
    """

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
    ):
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if soft_tree_depth <= 0:
            raise ValueError("soft_tree_depth must be positive.")
        if soft_tree_temperature <= 0:
            raise ValueError("soft_tree_temperature must be positive.")

        probs = torch.tensor(
            [
                linear_activation_prob,
                small_mlp_prob,
                soft_tree_prob,
            ],
            device=device,
            dtype=torch.float32,
        )
        if bool((probs < 0).any()):
            raise ValueError("Edge-family probabilities must be nonnegative.")
        if float(probs.sum().item()) <= 0:
            raise ValueError("At least one edge-family probability must be positive.")
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

        if self.edge_type == self.EDGE_LINEAR_ACTIVATION:
            self._init_linear_activation(generator)

        elif self.edge_type == self.EDGE_SMALL_MLP:
            self._init_small_mlp(
                generator,
                small_mlp_hidden_dim,
            )

        elif self.edge_type == self.EDGE_SOFT_TREE:
            self._init_soft_tree(generator)

        else:
            raise RuntimeError(f"Unknown edge_type={self.edge_type}.")
        

    def name(self) -> str:
        if self.edge_type == self.EDGE_LINEAR_ACTIVATION:
            return f"{self.edge_name}:{self.activation_name}"
        return self.edge_name


    def _init_linear_activation(
        self,
        generator: torch.Generator,
    ) -> None:
        scale = self.latent_dim ** -0.5

        self.linear_W = scale * _randn(
            self.latent_dim,
            self.latent_dim,
            generator=generator,
            device=self.device,
        )
        self.linear_b = _randn(
            self.latent_dim,
            generator=generator,
            device=self.device,
        )

        self.activation_id = int(
            _randint(
                0,
                len(self.ACTIVATION_NAMES),
                (),
                generator=generator,
                device=self.device,
            ).item()
        )
        self.activation_name = self.ACTIVATION_NAMES[self.activation_id]

    
    def _init_small_mlp(
        self,
        generator: torch.Generator,
        hidden_dim: Optional[int],
    ) -> None:
        hidden_dim = (
            int(hidden_dim)
            if hidden_dim is not None
            else 2 * self.latent_dim
        )

        if hidden_dim <= 0:
            raise ValueError("small_mlp_hidden_dim must be positive.")
        
        self.mlp_hidden_dim = hidden_dim

        self.mlp_W1 = (self.latent_dim ** -0.5) * _randn(
            hidden_dim,
            self.latent_dim,
            generator=generator,
            device=self.device,
        )
        self.mlp_b1 = _randn(
            hidden_dim,
            generator=generator,
            device=self.device,
        )

        self.mlp_W2 = (hidden_dim ** -0.5) * _randn(
            self.latent_dim,
            hidden_dim,
            generator=generator,
            device=self.device,
        )
        self.mlp_b2 = _randn(
            self.latent_dim,
            generator=generator,
            device=self.device,
        )
    

    def _init_soft_tree(
        self,
        generator: torch.Generator,
    ) -> None:
        n_internal = 2**self.soft_tree_depth - 1
        n_leaves = 2**self.soft_tree_depth

        self.tree_gate_W = (self.latent_dim ** -0.5) * _randn(
            n_internal,
            self.latent_dim,
            generator=generator,
            device=self.device,
        )
        self.tree_gate_b = _randn(
            n_internal,
            generator=generator,
            device=self.device,
        )
        self.tree_leaf_values = _randn(
            n_leaves,
            self.latent_dim,
            generator=generator,
            device=self.device,
        )


    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        name = self.activation_name

        if name == "identity":
            return x
        if name == "tanh":
            return torch.tanh(x)
        if name == "relu":
            return torch.relu(x)
        if name == "sigmoid":
            return torch.sigmoid(x) - 0.5
        if name == "sin":
            return torch.sin(x)
        if name == "square":
            return x.square()
        if name == "softplus":
            return F.softplus(x)

        raise RuntimeError(f"Unknown activation: {name}")

    def _soft_tree(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable full binary tree.

        Each internal node produces a soft right-branch probability.
        Leaf probabilities are products of left/right routing probabilities.
        """
        gate_logits = (
            x @ self.tree_gate_W.T + self.tree_gate_b
        ) / self.soft_tree_temperature
        right_prob = torch.sigmoid(gate_logits)
        left_prob = 1.0 - right_prob

        # Start with probability 1 at the root, then expand level-by-level.
        path_probs = torch.ones(
            x.shape[0],
            1,
            device=x.device,
            dtype=x.dtype,
        )
        offset = 0

        for depth in range(self.soft_tree_depth):
            nodes_at_level = 2 ** depth
            level_left = left_prob[:, offset : offset + nodes_at_level]
            level_right = right_prob[:, offset : offset + nodes_at_level]

            path_probs = torch.stack(
                [
                    path_probs * level_left,
                    path_probs * level_right,
                ],
                dim=-1,
            ).reshape(x.shape[0], -1)

            offset += nodes_at_level

        return path_probs @ self.tree_leaf_values


    def __call__(self, parent_latent: torch.Tensor) -> torch.Tensor:
        if parent_latent.ndim != 2:
            raise ValueError(
                f"Expected parent latent [N, h], got {tuple(parent_latent.shape)}."
            )

        if parent_latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, "
                f"got {parent_latent.shape[1]}."
            )

        x = parent_latent.float()

        if self.edge_type == self.EDGE_LINEAR_ACTIVATION:
            return self._activation(
                x @ self.linear_W.T + self.linear_b
            )

        if self.edge_type == self.EDGE_SMALL_MLP:
            h = torch.tanh(
                x @ self.mlp_W1.T + self.mlp_b1
            )
            return h @ self.mlp_W2.T + self.mlp_b2

        if self.edge_type == self.EDGE_SOFT_TREE:
            return self._soft_tree(x)

        raise RuntimeError(f"Unknown edge_type={self.edge_type}.")


# ---------------------------------------------------------------------------
# Per-node observation heads
# ---------------------------------------------------------------------------

class ObservationHead:
    """
    Per-node observation head.

    Continuous:
        latent [N, h] -> linear -> [N]
        optional Gaussian observation noise
        standardize

    Categorical:
        latent [N, h] -> linear -> logits [N, K]
        optional Gaussian logit noise
        softmax(logits / temperature)
        sample or argmax -> category ID [N]
    """

    def __init__(
        self,
        spec: NodeSpec,
        latent_dim: int,
        generator: torch.Generator,
        device: torch.device,
        categorical_temperature: float = 1.0,
    ):
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if categorical_temperature <= 0:
            raise ValueError("categorical_temperature must be positive.")

        self.spec = spec
        self.latent_dim = int(latent_dim)
        self.device = device
        self.temperature = float(categorical_temperature)

        if spec.kind == "cont":
            self.W = (latent_dim ** -0.5) * _randn(
                latent_dim,
                generator=generator,
                device=device,
            )
            self.b = _randn((), generator=generator, device=device)

        else:
            assert spec.K is not None
            self.W = (latent_dim ** -0.5) * _randn(
                spec.K,
                latent_dim,
                generator=generator,
                device=device,
            )
            self.b = _randn(
                spec.K,
                generator=generator,
                device=device,
            )

    def observe(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
        sample_categorical: bool = True,
        observation_noise_scale: float = 0.0,
    ) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent [N, {self.latent_dim}], "
                f"got {tuple(latent.shape)}."
            )
        if observation_noise_scale < 0:
            raise ValueError("observation_noise_scale must be nonnegative.")

        z = _standardize(latent.float(), dim=0)

        if self.spec.kind == "cont":
            value = z @ self.W + self.b

            if observation_noise_scale > 0:
                value = value + observation_noise_scale * torch.randn(
                    value.shape,
                    generator=generator,
                    device=self.device,
                    dtype=value.dtype,
                )

            return _standardize(value, dim=0).squeeze(-1)

        assert self.spec.K is not None
        logits = z @ self.W.T + self.b

        if observation_noise_scale > 0:
            logits = logits + observation_noise_scale * torch.randn(
                logits.shape,
                generator=generator,
                device=self.device,
                dtype=logits.dtype,
            )

        probs = torch.softmax(logits / self.temperature, dim=-1)

        if sample_categorical:
            return torch.multinomial(
                probs,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)

        return torch.argmax(probs, dim=-1)


# ---------------------------------------------------------------------------
# Sparse layered latent SCM
# ---------------------------------------------------------------------------

class LatentLayerConnection:
    """
    Sparse connection from layer l to layer l+1.

    adj[i, j] == True:
        parent node i connects to child node j.

    Every active edge maps:
        [N, h] -> [N, h]
    """

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
    ):
        if in_width <= 0 or out_width <= 0:
            raise ValueError("Layer widths must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if not 0.0 <= edge_prob <= 1.0:
            raise ValueError("edge_prob must lie in [0, 1].")
        if min_parents_per_node < 1:
            raise ValueError("min_parents_per_node must be at least 1.")

        self.in_width = int(in_width)
        self.out_width = int(out_width)
        self.latent_dim = int(latent_dim)
        self.device = device

        min_parents = min(min_parents_per_node, self.in_width)

        self.adj = _rand(
            self.in_width,
            self.out_width,
            generator=generator,
            device=device,
        ) < edge_prob

        # Guarantee that every child has enough parents.
        for child_idx in range(self.out_width):
            current = int(self.adj[:, child_idx].sum().item())
            missing = min_parents - current

            if missing <= 0:
                continue

            candidates = torch.where(~self.adj[:, child_idx])[0]
            order = torch.randperm(
                candidates.numel(),
                generator=generator,
                device=device,
            )
            chosen = candidates[order[:missing]]
            self.adj[chosen, child_idx] = True

        self.edges: list[list[Optional[LatentEdge]]] = [
            [None for _ in range(self.out_width)]
            for _ in range(self.in_width)
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
                f"Expected {self.in_width} parent nodes, "
                f"got {len(parent_latents)}."
            )
        if latent_noise_scale < 0:
            raise ValueError("latent_noise_scale must be nonnegative.")

        child_latents: list[torch.Tensor] = []

        for child_idx in range(self.out_width):
            incoming = []

            for parent_idx in range(self.in_width):
                edge = self.edges[parent_idx][child_idx]
                if edge is not None:
                    incoming.append(edge(parent_latents[parent_idx]))

            if not incoming:
                raise RuntimeError("Every child should have at least one parent.")

            combined = torch.stack(incoming, dim=0).sum(dim=0)
            combined /= math.sqrt(len(incoming))
            # combined = combined / (len(incoming) ** 0.5)
            # combined = _standardize(combined, dim=0)

            if latent_noise_scale > 0:
                combined = combined + latent_noise_scale * torch.randn(
                    combined.shape,
                    generator=generator,
                    device=self.device,
                    dtype=combined.dtype,
                )
                #combined = _standardize(combined, dim=0)

            child_latents.append(combined)

        return child_latents


class RandomLayeredLatentSCM:
    """
    Random sparse layered SCM with a unified latent state for every node.

    Internal state:
        node latent: [N, latent_dim]

    Node kind affects only observation:
        cont -> scalar continuous feature
        cat  -> integer category ID

    Graph density:
        task_edge_prob ~ Beta(edge_beta_alpha, edge_beta_beta)
        adj[i, j] ~ Bernoulli(task_edge_prob)
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
        p_cat: float = 0.3,
        max_cardinality: int = 10,
        min_parents_per_node: int = 1,
        latent_noise_scale: float = 0.05,
        observation_noise_scale: float = 0.05,
        categorical_temperature: float = 1.0,
        linear_activation_prob: float = 0.60,
        small_mlp_prob: float = 0.25,
        soft_tree_prob: float = 0.15,
        small_mlp_hidden_dim: Optional[int] = None,
        soft_tree_depth: int = 2,
        soft_tree_temperature: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")

        if num_roots <= 0:
            raise ValueError("num_roots must be positive.")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")
        if max_nodes_per_layer < 1:
            raise ValueError("max_nodes_per_layer must be positive.")
        if max_nodes_per_layer < 5:
            raise ValueError(
                "max_nodes_per_layer must be >= 5 because hidden widths "
                "are sampled from [5, max_nodes_per_layer]."
            )
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if not 0.0 <= edge_prob_min <= edge_prob_max <= 1.0:
            raise ValueError(
                "Require 0 <= edge_prob_min <= edge_prob_max <= 1."
            )
        if not 0.0 <= p_cat <= 1.0:
            raise ValueError("p_cat must lie in [0, 1].")
        if max_cardinality < 2:
            raise ValueError("max_cardinality must be at least 2.")

        self.g_dag = g_dag
        self.g_x = g_x
        self.g_aleatoric = g_aleatoric

        self.num_roots = int(num_roots)
        self.num_layers = int(num_layers)
        self.max_nodes_per_layer = int(max_nodes_per_layer)
        self.latent_dim = int(latent_dim)
        self.p_cat = float(p_cat)
        self.max_cardinality = int(max_cardinality)
        self.min_parents_per_node = int(min_parents_per_node)
        self.latent_noise_scale = float(latent_noise_scale)
        self.observation_noise_scale = float(observation_noise_scale)
        self.categorical_temperature = float(categorical_temperature)

        raw_edge_prob = _sample_beta_scalar(
            alpha=edge_beta_alpha,
            beta=edge_beta_beta,
            generator=self.g_dag,
            device=self.device,
        )
        self.task_edge_prob = (
            edge_prob_min
            + (edge_prob_max - edge_prob_min) * raw_edge_prob
        )

        self.widths = self._sample_widths()

        self.layers: list[list[NodeSpec]] = [
            [self._sample_node_spec() for _ in range(width)]
            for width in self.widths
        ]

        self.observation_heads: list[list[ObservationHead]] = [
            [
                ObservationHead(
                    spec=spec,
                    latent_dim=self.latent_dim,
                    generator=self.g_dag,
                    device=self.device,
                    categorical_temperature=self.categorical_temperature,
                )
                for spec in layer_specs
            ]
            for layer_specs in self.layers
        ]

        self.connections: list[LatentLayerConnection] = []

        for layer_idx in range(self.num_layers - 1):
            self.connections.append(
                LatentLayerConnection(
                    in_width=self.widths[layer_idx],
                    out_width=self.widths[layer_idx + 1],
                    latent_dim=self.latent_dim,
                    edge_prob=self.task_edge_prob,
                    min_parents_per_node=self.min_parents_per_node,
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

    def _sample_node_spec(self) -> NodeSpec:
        is_cat = float(
            _rand(
                (),
                generator=self.g_dag,
                device=self.device,
            ).item()
        ) < self.p_cat

        if is_cat:
            K = int(
                _randint(
                    2,
                    self.max_cardinality + 1,
                    (),
                    generator=self.g_dag,
                    device=self.device,
                ).item()
            )
            return NodeSpec(kind="cat", K=K)

        return NodeSpec(kind="cont", K=None)

    def sample_root_latents(self, n_samples: int) -> list[torch.Tensor]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        return [
            _standardize(
                _randn(
                    n_samples,
                    self.latent_dim,
                    generator=self.g_x,
                    device=self.device,
                ),
                dim=0,
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
                raise ValueError(
                    "Either root_latents or n_samples must be provided."
                )
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

    def observe_node(
        self,
        layer_idx: int,
        node_idx: int,
        latent: torch.Tensor,
        sample_categorical: bool = True,
        observation_noise_scale: Optional[float] = None,
    ) -> torch.Tensor:
        noise_scale = (
            self.observation_noise_scale
            if observation_noise_scale is None
            else float(observation_noise_scale)
        )

        return self.observation_heads[layer_idx][node_idx].observe(
            latent=latent,
            generator=self.g_aleatoric,
            sample_categorical=sample_categorical,
            observation_noise_scale=noise_scale,
        )

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

        for layer_idx, specs in enumerate(self.layers):
            print(f"Layer {layer_idx}:")
            for node_idx, spec in enumerate(specs):
                if spec.kind == "cont":
                    print(f"  node {node_idx}: cont")
                else:
                    print(f"  node {node_idx}: cat, K={spec.K}")
            print()

        for layer_idx, connection in enumerate(self.connections):
            print(
                f"Connection layer {layer_idx} -> layer {layer_idx + 1}:"
            )
            print(connection.adj.long())
            print(f"num_edges = {int(connection.adj.sum().item())}")

            for parent_idx in range(connection.in_width):
                for child_idx in range(connection.out_width):
                    edge = connection.edges[parent_idx][child_idx]
                    if edge is not None:
                        print(
                            f"  edge {parent_idx}->{child_idx}: {edge.name()}"
                        )
            print()


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
        p_cat: float = 0.3,
        max_cardinality: int = 10,
        min_parents_per_node: int = 1,
        categorical_temperature: float = 1.0,
        linear_activation_prob: float = 0.60,
        small_mlp_prob: float = 0.25,
        soft_tree_prob: float = 0.15,
        small_mlp_hidden_dim: Optional[int] = None,
        soft_tree_depth: int = 2,
        soft_tree_temperature: float = 0.5,
    ):
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

        self.p_cat = float(p_cat)
        self.max_cardinality = int(max_cardinality)
        self.min_parents_per_node = int(min_parents_per_node)
        self.categorical_temperature = float(categorical_temperature)

        self.linear_activation_prob = float(linear_activation_prob)
        self.small_mlp_prob = float(small_mlp_prob)
        self.soft_tree_prob = float(soft_tree_prob)
        self.small_mlp_hidden_dim = small_mlp_hidden_dim
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_temperature = float(soft_tree_temperature)

        self.g_dag, self.dag_seed = make_gen(self.device, dag_seed)
        self.g_aleatoric, self.aleatoric_seed = make_gen(
            self.device,
            aleatoric_seed,
        )
        self.g_x, self.x_seed = make_gen(self.device, x_seed)

        self.d = int(
            torch.randint(
                self.d_min,
                self.d_max + 1,
                (1,),
                device=self.device,
                generator=self.g_dag,
            ).item()
        )
        self.n = int(
            torch.randint(
                self.n_min,
                self.n_max + 1,
                (1,),
                device=self.device,
                generator=self.g_dag,
            ).item()
        )

        super().__init__()

    def _flatten_latents(
        self,
        scm: RandomLayeredLatentSCM,
        all_latents: list[list[torch.Tensor]],
    ):
        flat_latents: list[torch.Tensor] = []
        flat_specs: list[NodeSpec] = []
        flat_index: list[tuple[int, int]] = []

        for layer_idx, layer_latents in enumerate(all_latents):
            for node_idx, latent in enumerate(layer_latents):
                flat_latents.append(latent)
                flat_specs.append(scm.layers[layer_idx][node_idx])
                flat_index.append((layer_idx, node_idx))

        return flat_latents, flat_specs, flat_index
    

    def _sample_feature_and_target_sources(
        self,
        flat_index: list[tuple[int, int]],
        d: int,
    ):
        """
        Target:
            one random node from the final layer, regardless of kind.

        Features:
            random nodes from strictly earlier layers.
        """

        max_layer = max(layer_idx for layer_idx, _ in flat_index)

        target_candidates = [
            node_id
            for node_id, (layer_idx, _) in enumerate(flat_index)
            if layer_idx == max_layer
        ]

        if not target_candidates:
            raise RuntimeError("The final layer contains no nodes.")

        target_pos = int(
            _randint(
                0,
                len(target_candidates),
                (),
                generator=self.g_dag,
                device=self.device,
            ).item()
        )
        target_id = target_candidates[target_pos]

        feature_candidates = [
            node_id
            for node_id, (layer_idx, _) in enumerate(flat_index)
            if layer_idx < max_layer
        ]

        if len(feature_candidates) < self.d_min:
            raise RuntimeError(
                f"Only {len(feature_candidates)} pre-target feature candidates exist, "
                f"but d_min={self.d_min}."
            )

        d = min(int(d), len(feature_candidates))

        permutation = torch.randperm(
            len(feature_candidates),
            generator=self.g_dag,
            device=self.device,
        )

        feature_ids = [
            feature_candidates[int(idx)]
            for idx in permutation[:d].tolist()
        ]

        return feature_ids, target_id


    def _observe_selected_nodes(
        self,
        scm: RandomLayeredLatentSCM,
        flat_latents: list[torch.Tensor],
        flat_specs: list[NodeSpec],
        flat_index: list[tuple[int, int]],
        feature_ids: list[int],
        target_id: int,
    ):
        n = flat_latents[0].shape[0]
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

        # Noise is added only when selected nodes are observed.
        for feature_col, node_id in enumerate(feature_ids):
            layer_idx, node_idx = flat_index[node_id]
            spec = flat_specs[node_id]

            observed = scm.observe_node(
                layer_idx=layer_idx,
                node_idx=node_idx,
                latent=flat_latents[node_id],
                sample_categorical=True,
                observation_noise_scale=self.observation_noise_scale,
            )

            X[:, feature_col] = observed.float()

            if spec.kind == "cont":
                feature_type[feature_col] = self.CONTINUOUS
                cardinality[feature_col] = 0
            else:
                assert spec.K is not None
                feature_type[feature_col] = self.CATEGORICAL
                cardinality[feature_col] = int(spec.K)

        target_latent = flat_latents[target_id]

        y = self.target_observation_head.observe(
            latent=target_latent,
            generator=self.g_aleatoric,
            sample_categorical=True,
            observation_noise_scale=self.observation_noise_scale,
        )

        return X, y, feature_type, cardinality

    def _compute_importance_ground_truth(
        self,
        scm: RandomLayeredLatentSCM,
        all_latents: list[list[torch.Tensor]],
        flat_latents: list[torch.Tensor],
        flat_specs: list[NodeSpec],
        flat_index: list[tuple[int, int]],
        feature_ids: list[int],
        target_id: int,
    ):
        """
        Placeholder for future importance-ground-truth logic.

        Current behavior:
            feature_strength  = NaN
            importance_ratio = NaN
            is_active         = NaN

        This intentionally avoids asserting that non-ancestor features have zero
        importance or selecting a particular intervention definition prematurely.
        """
        del (
            scm,
            all_latents,
            flat_latents,
            flat_specs,
            flat_index,
            target_id,
        )

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
            p_cat=self.p_cat,
            max_cardinality=self.max_cardinality,
            min_parents_per_node=self.min_parents_per_node,
            latent_noise_scale=self.latent_noise_scale,
            observation_noise_scale=self.observation_noise_scale,
            categorical_temperature=self.categorical_temperature,
            linear_activation_prob=self.linear_activation_prob,
            small_mlp_prob=self.small_mlp_prob,
            soft_tree_prob=self.soft_tree_prob,
            small_mlp_hidden_dim=self.small_mlp_hidden_dim,
            soft_tree_depth=self.soft_tree_depth,
            soft_tree_temperature=self.soft_tree_temperature,
            device=device,
        )

        if self.num_classes is None:
            target_spec = NodeSpec(
                kind="cont",
                K=None,
            )
        else:
            if self.num_classes < 2:
                raise ValueError("num_classes must be at least 2.")

            target_spec = NodeSpec(
                kind="cat",
                K=int(self.num_classes),
            )

        self.target_observation_head = ObservationHead(
            spec=target_spec,
            latent_dim=self.latent_dim,
            generator=self.g_dag,
            device=self.device,
            categorical_temperature=self.categorical_temperature,
        )

        all_latents = scm.forward(
            n_samples=n,
            latent_noise_scale=self.latent_noise_scale,
        )

        flat_latents, flat_specs, flat_index = self._flatten_latents(
            scm,
            all_latents,
        )

        feature_ids, target_id = self._sample_feature_and_target_sources(
            flat_index=flat_index,
            d=d,
        )

        self.d = len(feature_ids)

        X_clean, y, feature_type, cardinality = (
            self._observe_selected_nodes(
                scm=scm,
                flat_latents=flat_latents,
                flat_specs=flat_specs,
                flat_index=flat_index,
                feature_ids=feature_ids,
                target_id=target_id,
            )
        )

        importance_info = self._compute_importance_ground_truth(
            scm=scm,
            all_latents=all_latents,
            flat_latents=flat_latents,
            flat_specs=flat_specs,
            flat_index=flat_index,
            feature_ids=feature_ids,
            target_id=target_id,
        )

        if self.num_classes is None:
            y = y.float()
            self.n_classes = None
        else:
            y = y.long()
            self.n_classes = int(self.num_classes)

        # Missingness is applied after observation and before splitting.
        X_obs = X_clean.clone()
        missing_mask = (
            torch.rand(
                X_obs.shape,
                device=device,
                generator=self.g_x,
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
            n_test = max(1, int(round(n * self.test_frac)))
            n_test = min(n_test, n - 2)

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
        self.scm = scm

        return X_train, y_train, X_test, y_test, info

    def visualize(self):
        return None

    def forward(self, X: torch.Tensor):
        del X
        return None


# Backward-compatible aliases, if desired.
RandomLayeredSCM = RandomLayeredLatentSCM
MixedSCMTask = MixedLatentSCMTask

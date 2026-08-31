from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.data.helper import make_gen, stratified_classification_split
from src.data.synthetic_task import GenerateTask


def _randn(*shape: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, device=device)


def _rand(*shape: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.rand(*shape, generator=generator, device=device)


def _randint(
    low: int,
    high: int,
    shape,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(low, high, shape, generator=generator, device=device)


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
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


def _normalize_probs(
    values: tuple[float, ...],
    *,
    expected_len: Optional[int],
    name: str,
    device: torch.device,
) -> torch.Tensor:
    probs = torch.tensor(values, device=device, dtype=torch.float32)
    if expected_len is not None and probs.numel() != expected_len:
        raise ValueError(f"{name} must contain exactly {expected_len} values.")
    if probs.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if bool((probs < 0).any()) or float(probs.sum().item()) <= 0:
        raise ValueError(f"{name} must be nonnegative and have positive sum.")
    return probs / probs.sum()


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

        probs = _normalize_probs(
            (linear_activation_prob, small_mlp_prob, soft_tree_prob),
            expected_len=3,
            name="edge-family probabilities",
            device=device,
        )

        self.latent_dim = int(latent_dim)
        self.device = device
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_temperature = float(soft_tree_temperature)
        self.edge_type = int(
            torch.multinomial(probs, 1, replacement=True, generator=generator).item()
        )
        self.edge_name = self.EDGE_NAMES[self.edge_type]

        scale = self.latent_dim ** -0.5
        self.linear_W = scale * _randn(
            self.latent_dim,
            self.latent_dim,
            generator=generator,
            device=device,
        )
        self.linear_b = _randn(self.latent_dim, generator=generator, device=device)
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
                [path_probs * left, path_probs * right], dim=-1
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
    """Sparse latent-layer connection with optional dominant-parent aggregation."""

    def __init__(
        self,
        in_width: int,
        out_width: int,
        latent_dim: int,
        edge_prob: float,
        min_parents_per_node: int,
        generator: torch.Generator,
        device: torch.device,
        dominant_parent_prob: float = 0.40,
        dominant_parent_weight: float = 0.80,
        linear_activation_prob: float = 0.60,
        small_mlp_prob: float = 0.25,
        soft_tree_prob: float = 0.15,
        small_mlp_hidden_dim: Optional[int] = None,
        soft_tree_depth: int = 2,
        soft_tree_temperature: float = 0.5,
    ) -> None:
        if in_width <= 0 or out_width <= 0:
            raise ValueError("Layer widths must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if not 0.0 <= edge_prob <= 1.0:
            raise ValueError("edge_prob must lie in [0, 1].")
        if min_parents_per_node < 1:
            raise ValueError("min_parents_per_node must be at least 1.")
        if not 0.0 <= dominant_parent_prob <= 1.0:
            raise ValueError("dominant_parent_prob must lie in [0, 1].")
        if not 0.5 <= dominant_parent_weight <= 1.0:
            raise ValueError("dominant_parent_weight must lie in [0.5, 1.0].")

        self.in_width = int(in_width)
        self.out_width = int(out_width)
        self.latent_dim = int(latent_dim)
        self.device = device
        self.dominant_parent_prob = float(dominant_parent_prob)
        self.dominant_parent_weight = float(dominant_parent_weight)

        min_parents = min(int(min_parents_per_node), self.in_width)
        self.adj = (
            _rand(
                self.in_width,
                self.out_width,
                generator=generator,
                device=device,
            )
            < edge_prob
        )

        for child_idx in range(self.out_width):
            missing = min_parents - int(self.adj[:, child_idx].sum().item())
            if missing <= 0:
                continue
            candidates = torch.where(~self.adj[:, child_idx])[0]
            order = torch.randperm(
                candidates.numel(), generator=generator, device=device
            )
            self.adj[candidates[order[:missing]], child_idx] = True

        self.edges: list[list[Optional[LatentEdge]]] = [
            [None for _ in range(self.out_width)] for _ in range(self.in_width)
        ]
        for parent_idx in range(self.in_width):
            for child_idx in range(self.out_width):
                if not bool(self.adj[parent_idx, child_idx]):
                    continue
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

        self.uses_dominant_parent = torch.zeros(
            self.out_width, device=device, dtype=torch.bool
        )
        self.dominant_parent_indices = torch.full(
            (self.out_width,), -1, device=device, dtype=torch.long
        )

        for child_idx in range(self.out_width):
            parent_indices = torch.where(self.adj[:, child_idx])[0]
            num_parents = int(parent_indices.numel())
            if num_parents < 2:
                continue
            use_dominant = bool(
                (
                    _rand((), generator=generator, device=device)
                    < self.dominant_parent_prob
                ).item()
            )
            if not use_dominant:
                continue
            selected_position = int(
                _randint(
                    0,
                    num_parents,
                    (),
                    generator=generator,
                    device=device,
                ).item()
            )
            self.uses_dominant_parent[child_idx] = True
            self.dominant_parent_indices[child_idx] = parent_indices[
                selected_position
            ]

    @staticmethod
    def _aggregate_standard(incoming: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(incoming, dim=0).sum(dim=0) / (len(incoming) ** 0.5)

    def _aggregate_with_dominant_parent(
        self,
        parent_outputs: dict[int, torch.Tensor],
        dominant_parent_idx: int,
    ) -> torch.Tensor:
        dominant_output = parent_outputs[dominant_parent_idx]
        other_outputs = [
            output
            for parent_idx, output in parent_outputs.items()
            if parent_idx != dominant_parent_idx
        ]
        if not other_outputs:
            return dominant_output
        normalized_other = torch.stack(other_outputs, dim=0).sum(dim=0) / (
            len(other_outputs) ** 0.5
        )
        w = self.dominant_parent_weight
        return w * dominant_output + (1.0 - w) * normalized_other

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
            parent_outputs: dict[int, torch.Tensor] = {}
            for parent_idx in range(self.in_width):
                edge = self.edges[parent_idx][child_idx]
                if edge is not None:
                    parent_outputs[parent_idx] = edge(parent_latents[parent_idx])
            if not parent_outputs:
                raise RuntimeError("Every child must have at least one parent.")

            if bool(self.uses_dominant_parent[child_idx].item()):
                child = self._aggregate_with_dominant_parent(
                    parent_outputs,
                    int(self.dominant_parent_indices[child_idx].item()),
                )
            else:
                child = self._aggregate_standard(list(parent_outputs.values()))

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
    """Sparse layered SCM whose internal node states remain continuous."""

    ROOT_PRIOR_NAMES = (
        "gaussian",
        "uniform",
        "heavy_tailed",
        "skewed",
        "mixture",
    )

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
        root_prior_probs: tuple[float, float, float, float, float] = (
            0.45,
            0.20,
            0.15,
            0.05,
            0.15,
        ),
        root_mixture_component_probs: tuple[float, float, float, float, float] = (
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),
        root_mixture_separation_min: float = 1.5,
        root_mixture_separation_max: float = 3.0,
        root_mixture_scale_min: float = 0.40,
        root_mixture_scale_max: float = 0.90,
        dominant_parent_prob: float = 0.40,
        dominant_parent_weight: float = 0.75,
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
        self.root_prior_probs = _normalize_probs(
            root_prior_probs,
            expected_len=5,
            name="root_prior_probs",
            device=self.device,
        )
        self.root_mixture_component_probs = _normalize_probs(
            root_mixture_component_probs,
            expected_len=5,
            name="root_mixture_component_probs",
            device=self.device,
        )
        self.root_mixture_separation_min = float(root_mixture_separation_min)
        self.root_mixture_separation_max = float(root_mixture_separation_max)
        self.root_mixture_scale_min = float(root_mixture_scale_min)
        self.root_mixture_scale_max = float(root_mixture_scale_max)
        if self.root_mixture_separation_min <= 0:
            raise ValueError("root_mixture_separation_min must be positive.")
        if self.root_mixture_separation_max < self.root_mixture_separation_min:
            raise ValueError("Invalid mixture separation range.")
        if self.root_mixture_scale_min <= 0:
            raise ValueError("root_mixture_scale_min must be positive.")
        if self.root_mixture_scale_max < self.root_mixture_scale_min:
            raise ValueError("Invalid mixture scale range.")

        raw_edge_prob = _sample_beta_scalar(
            edge_beta_alpha, edge_beta_beta, self.g_dag, self.device
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
                    dominant_parent_prob=dominant_parent_prob,
                    dominant_parent_weight=dominant_parent_weight,
                    linear_activation_prob=linear_activation_prob,
                    small_mlp_prob=small_mlp_prob,
                    soft_tree_prob=soft_tree_prob,
                    small_mlp_hidden_dim=small_mlp_hidden_dim,
                    soft_tree_depth=soft_tree_depth,
                    soft_tree_temperature=soft_tree_temperature,
                )
            )

        self.root_prior_types: list[str] = []
        self.root_prior_type_ids: list[int] = []
        self.root_mixture_components: list[int] = []

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

    def _sample_gaussian_root(self, n_samples: int) -> torch.Tensor:
        return _randn(
            n_samples,
            self.latent_dim,
            generator=self.g_x,
            device=self.device,
        )

    def _sample_uniform_root(self, n_samples: int) -> torch.Tensor:
        bound = 3.0**0.5
        return (
            2.0
            * bound
            * _rand(
                n_samples,
                self.latent_dim,
                generator=self.g_x,
                device=self.device,
            )
            - bound
        )

    def _sample_heavy_tailed_root(self, n_samples: int) -> torch.Tensor:
        df = 4.0
        numerator = _randn(
            n_samples,
            self.latent_dim,
            generator=self.g_x,
            device=self.device,
        )
        concentration = torch.full(
            (n_samples, 1),
            df / 2.0,
            device=self.device,
            dtype=numerator.dtype,
        )
        chi_square = 2.0 * torch._standard_gamma(
            concentration, generator=self.g_x
        )
        return numerator / torch.sqrt(chi_square / df).clamp_min(1e-4)

    def _sample_skewed_root(self, n_samples: int) -> torch.Tensor:
        gaussian = _randn(
            n_samples,
            self.latent_dim,
            generator=self.g_x,
            device=self.device,
        )
        strength = 0.40 + 0.60 * _rand(
            self.latent_dim, generator=self.g_dag, device=self.device
        )
        skewed = torch.exp(gaussian * strength[None, :])
        signs = torch.where(
            _rand(self.latent_dim, generator=self.g_dag, device=self.device) < 0.5,
            -torch.ones(self.latent_dim, device=self.device, dtype=skewed.dtype),
            torch.ones(self.latent_dim, device=self.device, dtype=skewed.dtype),
        )
        return skewed * signs[None, :]

    def _sample_mixture_root(self, n_samples: int) -> tuple[torch.Tensor, int]:
        component_choice = int(
            torch.multinomial(
                self.root_mixture_component_probs,
                1,
                replacement=True,
                generator=self.g_dag,
            ).item()
        )
        num_components = component_choice + 2
        weight_logits = 0.5 * _randn(
            num_components, generator=self.g_dag, device=self.device
        )
        component_weights = torch.softmax(weight_logits, dim=0)
        component_ids = torch.multinomial(
            component_weights,
            n_samples,
            replacement=True,
            generator=self.g_x,
        )
        directions = _randn(
            num_components,
            self.latent_dim,
            generator=self.g_dag,
            device=self.device,
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
        ) * _rand(num_components, generator=self.g_dag, device=self.device)
        noise = _randn(
            n_samples,
            self.latent_dim,
            generator=self.g_x,
            device=self.device,
        )
        return centers[component_ids] + scales[component_ids, None] * noise, num_components

    def sample_root_latents(self, n_samples: int) -> list[torch.Tensor]:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        roots: list[torch.Tensor] = []
        self.root_prior_types = []
        self.root_prior_type_ids = []
        self.root_mixture_components = []

        for _ in range(self.num_roots):
            prior_id = int(
                torch.multinomial(
                    self.root_prior_probs,
                    1,
                    replacement=True,
                    generator=self.g_dag,
                ).item()
            )
            prior_name = self.ROOT_PRIOR_NAMES[prior_id]
            mixture_components = 0
            if prior_name == "gaussian":
                latent = self._sample_gaussian_root(n_samples)
            elif prior_name == "uniform":
                latent = self._sample_uniform_root(n_samples)
            elif prior_name == "heavy_tailed":
                latent = self._sample_heavy_tailed_root(n_samples)
            elif prior_name == "skewed":
                latent = self._sample_skewed_root(n_samples)
            elif prior_name == "mixture":
                latent, mixture_components = self._sample_mixture_root(n_samples)
            else:
                raise RuntimeError(f"Unknown root prior: {prior_name}.")

            roots.append(_standardize(latent.float(), dim=0))
            self.root_prior_types.append(prior_name)
            self.root_prior_type_ids.append(prior_id)
            self.root_mixture_components.append(mixture_components)
        return roots

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
            if len(root_latents) != self.num_roots:
                raise ValueError(
                    f"Expected {self.num_roots} root nodes, got {len(root_latents)}."
                )
            current = root_latents
            self.root_prior_types = ["provided"] * self.num_roots
            self.root_prior_type_ids = [-1] * self.num_roots
            self.root_mixture_components = [0] * self.num_roots

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
        if self.root_prior_types:
            print(f"root_prior_types: {self.root_prior_types}")
            print(f"root_mixture_components: {self.root_mixture_components}")
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
# Feature observation: continuous / prototype / threshold-binning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureObservation:
    values: torch.Tensor
    is_categorical: bool
    cardinality: int
    score: torch.Tensor
    observation_type_id: int
    observation_type_name: str
    quality_score: float
    prototypes: torch.Tensor
    thresholds: torch.Tensor
    projection: torch.Tensor


class AdaptiveObservationHead:
    """
    Each selected latent node independently samples one observation mechanism:

    1. continuous_projection
    2. prototype_discretization
    3. threshold_binning

    The SCM latent remains h-dimensional. Only the observation head maps it to
    one scalar continuous column or one integer-valued categorical column.
    """

    CONTINUOUS = 0
    PROTOTYPE = 1
    BINNING = 2
    OBSERVATION_TYPE_NAMES = (
        "continuous_projection",
        "prototype_discretization",
        "threshold_binning",
    )

    def __init__(
        self,
        latent_dim: int,
        generator: torch.Generator,
        device: torch.device,
        observation_type_probs: tuple[float, float, float] = (0.50, 0.25, 0.25),
        categorical_cardinalities: tuple[int, ...] = (2, 3, 4, 5, 6),
        categorical_cardinality_probs: tuple[float, ...] = (
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),
        min_samples_per_category: int = 8,
        min_component_weight: float = 0.05,
        prototype_max_attempts: int = 8,
        prototype_min_separation: float = 1.0,
        binning_jitter: float = 0.20,
        observation_noise_scale: float = 0.05,
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if min_samples_per_category < 2:
            raise ValueError("min_samples_per_category must be at least 2.")
        if not 0.0 < min_component_weight < 1.0:
            raise ValueError("min_component_weight must lie in (0, 1).")
        if prototype_max_attempts < 1:
            raise ValueError("prototype_max_attempts must be at least 1.")
        if prototype_min_separation < 0:
            raise ValueError("prototype_min_separation must be nonnegative.")
        if not 0.0 <= binning_jitter <= 0.45:
            raise ValueError("binning_jitter must lie in [0, 0.45].")
        if observation_noise_scale < 0:
            raise ValueError("observation_noise_scale must be nonnegative.")
        if len(categorical_cardinalities) == 0:
            raise ValueError("categorical_cardinalities must not be empty.")
        if any(int(k) < 2 for k in categorical_cardinalities):
            raise ValueError("All categorical cardinalities must be at least 2.")
        if len(categorical_cardinalities) != len(categorical_cardinality_probs):
            raise ValueError(
                "categorical_cardinalities and categorical_cardinality_probs "
                "must have the same length."
            )

        self.latent_dim = int(latent_dim)
        self.device = device
        self.min_samples_per_category = int(min_samples_per_category)
        self.min_component_weight = float(min_component_weight)
        self.prototype_max_attempts = int(prototype_max_attempts)
        self.prototype_min_separation = float(prototype_min_separation)
        self.binning_jitter = float(binning_jitter)
        self.observation_noise_scale = float(observation_noise_scale)

        self.observation_type_probs = _normalize_probs(
            observation_type_probs,
            expected_len=3,
            name="observation_type_probs",
            device=device,
        )
        self.categorical_cardinalities = tuple(
            int(k) for k in categorical_cardinalities
        )
        self.categorical_cardinality_probs = _normalize_probs(
            categorical_cardinality_probs,
            expected_len=len(self.categorical_cardinalities),
            name="categorical_cardinality_probs",
            device=device,
        )

        self.sampled_observation_type_id = int(
            torch.multinomial(
                self.observation_type_probs,
                1,
                replacement=True,
                generator=generator,
            ).item()
        )

        self.continuous_W = (self.latent_dim ** -0.5) * _randn(
            self.latent_dim, generator=generator, device=device
        )
        self.continuous_b = _randn((), generator=generator, device=device)

        self.binning_W = (self.latent_dim ** -0.5) * _randn(
            self.latent_dim, generator=generator, device=device
        )
        self.binning_b = _randn((), generator=generator, device=device)

    def _prepare_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent [N, {self.latent_dim}], got {tuple(latent.shape)}."
            )
        if latent.shape[0] < 2:
            raise ValueError("At least two samples are required.")
        return _standardize(latent.float(), dim=0)

    def _sample_cardinality(self, n_samples: int, generator: torch.Generator) -> int:
        feasible_indices = [
            i
            for i, k in enumerate(self.categorical_cardinalities)
            if k * self.min_samples_per_category <= n_samples
            and k * self.min_component_weight <= 1.0 + 1e-12
        ]
        if not feasible_indices:
            return 0
        feasible_probs = self.categorical_cardinality_probs[
            torch.tensor(feasible_indices, device=self.device, dtype=torch.long)
        ]
        feasible_probs = feasible_probs / feasible_probs.sum()
        selected = int(
            torch.multinomial(
                feasible_probs,
                1,
                replacement=True,
                generator=generator,
            ).item()
        )
        return self.categorical_cardinalities[feasible_indices[selected]]

    def _observe_continuous(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
        *,
        fallback_name: Optional[str] = None,
    ) -> FeatureObservation:
        score = latent @ self.continuous_W + self.continuous_b
        if self.observation_noise_scale > 0:
            score = score + self.observation_noise_scale * torch.randn(
                score.shape,
                generator=generator,
                device=score.device,
                dtype=score.dtype,
            )
        score = _standardize(score, dim=0)
        name = fallback_name or self.OBSERVATION_TYPE_NAMES[self.CONTINUOUS]
        return FeatureObservation(
            values=score,
            is_categorical=False,
            cardinality=0,
            score=score,
            observation_type_id=self.CONTINUOUS,
            observation_type_name=name,
            quality_score=0.0,
            prototypes=torch.empty(
                0, self.latent_dim, device=latent.device, dtype=latent.dtype
            ),
            thresholds=torch.empty(0, device=latent.device, dtype=latent.dtype),
            projection=self.continuous_W.detach().clone(),
        )

    def _select_prototypes(
        self,
        latent: torch.Tensor,
        cardinality: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Choose actual latent samples as prototypes using k-means++ sampling."""
        n = latent.shape[0]
        first_idx = int(
            _randint(0, n, (), generator=generator, device=latent.device).item()
        )
        selected = [latent[first_idx]]
        for _ in range(1, cardinality):
            current = torch.stack(selected, dim=0)
            nearest_sq = torch.cdist(latent, current).square().min(dim=1).values
            total = nearest_sq.sum()
            if float(total.item()) <= 1e-12:
                idx = int(
                    _randint(0, n, (), generator=generator, device=latent.device).item()
                )
            else:
                idx = int(
                    torch.multinomial(
                        nearest_sq / total,
                        1,
                        replacement=True,
                        generator=generator,
                    ).item()
                )
            selected.append(latent[idx])
        return torch.stack(selected, dim=0)

    def _prototype_quality(
        self,
        latent: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        if prototypes.shape[0] < 2:
            return 0.0
        within = torch.linalg.vector_norm(
            latent - prototypes[labels], dim=1
        ).mean()
        pairwise = torch.cdist(prototypes, prototypes)
        pairwise.fill_diagonal_(float("inf"))
        between = pairwise.min()
        return float((between / within.clamp_min(1e-6)).item())

    def _prototype_is_valid(
        self,
        labels: torch.Tensor,
        cardinality: int,
        quality: float,
    ) -> bool:
        counts = torch.bincount(labels, minlength=cardinality)
        if int(counts.min().item()) < self.min_samples_per_category:
            return False
        if float((counts.float() / counts.sum()).min().item()) < self.min_component_weight:
            return False
        if quality < self.prototype_min_separation:
            return False
        return True

    def _observe_prototype(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> FeatureObservation:
        cardinality = self._sample_cardinality(latent.shape[0], generator)
        if cardinality == 0:
            return self._observe_continuous(
                latent,
                generator,
                fallback_name="continuous_fallback_from_prototype",
            )

        best: Optional[tuple[float, torch.Tensor, torch.Tensor]] = None
        for _ in range(self.prototype_max_attempts):
            prototypes = self._select_prototypes(latent, cardinality, generator)
            distances = torch.cdist(latent, prototypes)
            labels = torch.argmin(distances, dim=1)
            quality = self._prototype_quality(latent, prototypes, labels)
            if best is None or quality > best[0]:
                best = (quality, prototypes, labels)
            if self._prototype_is_valid(labels, cardinality, quality):
                return FeatureObservation(
                    values=labels,
                    is_categorical=True,
                    cardinality=cardinality,
                    score=torch.empty(0, device=latent.device, dtype=latent.dtype),
                    observation_type_id=self.PROTOTYPE,
                    observation_type_name=self.OBSERVATION_TYPE_NAMES[self.PROTOTYPE],
                    quality_score=quality,
                    prototypes=prototypes,
                    thresholds=torch.empty(
                        0, device=latent.device, dtype=latent.dtype
                    ),
                    projection=torch.empty(
                        0, device=latent.device, dtype=latent.dtype
                    ),
                )

        # A bad prototype draw is not silently accepted as categorical.
        # It falls back to a continuous observation.
        return self._observe_continuous(
            latent,
            generator,
            fallback_name="continuous_fallback_from_prototype",
        )

    def _sample_binning_cut_counts(
        self,
        n_samples: int,
        cardinality: int,
        generator: torch.Generator,
    ) -> list[int]:
        """Generate random but non-tiny bin sizes summing exactly to n_samples."""
        min_count = max(
            self.min_samples_per_category,
            int(torch.ceil(torch.tensor(self.min_component_weight * n_samples)).item()),
        )
        required = cardinality * min_count
        if required > n_samples:
            raise RuntimeError("Infeasible binning cardinality.")

        remaining = n_samples - required
        raw = _rand(cardinality, generator=generator, device=self.device)
        raw = raw / raw.sum().clamp_min(1e-12)
        extras_float = raw * remaining
        extras = torch.floor(extras_float).long()
        missing = remaining - int(extras.sum().item())
        if missing > 0:
            fractional = extras_float - extras.float()
            order = torch.argsort(fractional, descending=True)
            extras[order[:missing]] += 1
        counts = (extras + min_count).tolist()

        # Small local perturbation keeps bins from being too regular while
        # preserving all minimum-count constraints.
        max_shift = int(round(self.binning_jitter * n_samples / cardinality))
        if max_shift > 0 and cardinality > 1:
            for boundary in range(cardinality - 1):
                shift = int(
                    _randint(
                        -max_shift,
                        max_shift + 1,
                        (),
                        generator=generator,
                        device=self.device,
                    ).item()
                )
                if shift > 0:
                    allowed = counts[boundary + 1] - min_count
                    actual = min(shift, allowed)
                    counts[boundary] += actual
                    counts[boundary + 1] -= actual
                elif shift < 0:
                    allowed = counts[boundary] - min_count
                    actual = min(-shift, allowed)
                    counts[boundary] -= actual
                    counts[boundary + 1] += actual
        return [int(v) for v in counts]

    def _observe_binning(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> FeatureObservation:
        cardinality = self._sample_cardinality(latent.shape[0], generator)
        if cardinality == 0:
            return self._observe_continuous(
                latent,
                generator,
                fallback_name="continuous_fallback_from_binning",
            )

        scalar = latent @ self.binning_W + self.binning_b
        if self.observation_noise_scale > 0:
            scalar = scalar + self.observation_noise_scale * torch.randn(
                scalar.shape,
                generator=generator,
                device=scalar.device,
                dtype=scalar.dtype,
            )
        scalar = _standardize(scalar, dim=0)

        counts = self._sample_binning_cut_counts(
            latent.shape[0], cardinality, generator
        )
        sorted_values, _ = torch.sort(scalar)
        cumulative = 0
        thresholds: list[torch.Tensor] = []
        for count in counts[:-1]:
            cumulative += count
            left = sorted_values[cumulative - 1]
            right = sorted_values[cumulative]
            thresholds.append(0.5 * (left + right))
        threshold_tensor = torch.stack(thresholds)
        labels = torch.bucketize(scalar, threshold_tensor)

        observed_counts = torch.bincount(labels, minlength=cardinality)
        min_fraction = float(
            (observed_counts.float() / observed_counts.sum()).min().item()
        )
        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=cardinality,
            score=torch.empty(0, device=latent.device, dtype=latent.dtype),
            observation_type_id=self.BINNING,
            observation_type_name=self.OBSERVATION_TYPE_NAMES[self.BINNING],
            quality_score=min_fraction,
            prototypes=torch.empty(
                0, self.latent_dim, device=latent.device, dtype=latent.dtype
            ),
            thresholds=threshold_tensor,
            projection=self.binning_W.detach().clone(),
        )

    def observe(
        self,
        latent: torch.Tensor,
        generator: torch.Generator,
    ) -> FeatureObservation:
        prepared = self._prepare_latent(latent)
        if self.sampled_observation_type_id == self.CONTINUOUS:
            return self._observe_continuous(prepared, generator)
        if self.sampled_observation_type_id == self.PROTOTYPE:
            return self._observe_prototype(prepared, generator)
        if self.sampled_observation_type_id == self.BINNING:
            return self._observe_binning(prepared, generator)
        raise RuntimeError(
            f"Unknown observation type id: {self.sampled_observation_type_id}."
        )


class TargetObservationHead:
    """Target mechanism controlled only by task type."""

    def __init__(
        self,
        latent_dim: int,
        generator: torch.Generator,
        device: torch.device,
        observation_noise_scale: float = 0.05,
    ) -> None:
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if observation_noise_scale < 0:
            raise ValueError("observation_noise_scale must be nonnegative.")
        self.latent_dim = int(latent_dim)
        self.device = device
        self.observation_noise_scale = float(observation_noise_scale)
        self.W = (latent_dim ** -0.5) * _randn(
            latent_dim, generator=generator, device=device
        )
        self.b = _randn((), generator=generator, device=device)

    def score(self, latent: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent [N, {self.latent_dim}], got {tuple(latent.shape)}."
            )
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
        root_prior_probs: tuple[float, float, float, float, float] = (
            0.45,
            0.20,
            0.15,
            0.05,
            0.15,
        ),
        root_mixture_component_probs: tuple[float, float, float, float, float] = (
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),
        root_mixture_separation_min: float = 1.5,
        root_mixture_separation_max: float = 3.0,
        root_mixture_scale_min: float = 0.40,
        root_mixture_scale_max: float = 0.90,
        dominant_parent_prob: float = 0.40,
        dominant_parent_weight: float = 0.75,
        observation_type_probs: tuple[float, float, float] = (0.50, 0.25, 0.25),
        categorical_cardinalities: tuple[int, ...] = (2, 3, 4, 5, 6),
        categorical_cardinality_probs: tuple[float, ...] = (
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),
        min_samples_per_category: int = 8,
        min_component_weight: float = 0.05,
        prototype_max_attempts: int = 8,
        prototype_min_separation: float = 1.0,
        binning_jitter: float = 0.20,
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
        self.root_prior_probs = tuple(float(v) for v in root_prior_probs)
        self.root_mixture_component_probs = tuple(
            float(v) for v in root_mixture_component_probs
        )
        self.root_mixture_separation_min = float(root_mixture_separation_min)
        self.root_mixture_separation_max = float(root_mixture_separation_max)
        self.root_mixture_scale_min = float(root_mixture_scale_min)
        self.root_mixture_scale_max = float(root_mixture_scale_max)
        self.dominant_parent_prob = float(dominant_parent_prob)
        self.dominant_parent_weight = float(dominant_parent_weight)
        self.observation_type_probs = tuple(float(v) for v in observation_type_probs)
        self.categorical_cardinalities = tuple(
            int(v) for v in categorical_cardinalities
        )
        self.categorical_cardinality_probs = tuple(
            float(v) for v in categorical_cardinality_probs
        )
        self.min_samples_per_category = int(min_samples_per_category)
        self.min_component_weight = float(min_component_weight)
        self.prototype_max_attempts = int(prototype_max_attempts)
        self.prototype_min_separation = float(prototype_min_separation)
        self.binning_jitter = float(binning_jitter)
        self.linear_activation_prob = float(linear_activation_prob)
        self.small_mlp_prob = float(small_mlp_prob)
        self.soft_tree_prob = float(soft_tree_prob)
        self.small_mlp_hidden_dim = small_mlp_hidden_dim
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_temperature = float(soft_tree_temperature)

        if not 0.0 < self.test_frac < 1.0:
            raise ValueError("test_frac must lie in (0, 1).")
        if not 0.0 <= self.p_missing < 1.0:
            raise ValueError("p_missing must lie in [0, 1).")
        if self.n_min < 3 or self.n_max < self.n_min:
            raise ValueError("Require 3 <= n_min <= n_max.")
        if self.d_min < 1 or self.d_max < self.d_min:
            raise ValueError("Require 1 <= d_min <= d_max.")

        self.g_dag, self.dag_seed = make_gen(self.device, dag_seed)
        self.g_aleatoric, self.aleatoric_seed = make_gen(
            self.device, aleatoric_seed
        )
        self.g_x, self.x_seed = make_gen(self.device, x_seed)
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
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
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
        max_layer = max(layer_idx for layer_idx, _ in flat_index)
        target_candidates = [
            node_id
            for node_id, (layer_idx, _) in enumerate(flat_index)
            if layer_idx == max_layer
        ]
        if not target_candidates:
            raise RuntimeError("The final layer contains no nodes.")
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
        d = min(int(d), len(candidates))
        permutation = torch.randperm(
            len(candidates), generator=self.g_dag, device=self.device
        )
        feature_ids = [candidates[int(i)] for i in permutation[:d].tolist()]
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
        torch.Tensor,
        list[str],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[AdaptiveObservationHead],
    ]:
        n = int(flat_latents[0].shape[0])
        d = len(feature_ids)
        X = torch.empty(n, d, device=self.device, dtype=torch.float32)
        feature_type = torch.empty(d, device=self.device, dtype=torch.long)
        cardinality = torch.zeros(d, device=self.device, dtype=torch.long)
        quality_score = torch.zeros(d, device=self.device, dtype=torch.float32)
        observation_type_ids = torch.empty(d, device=self.device, dtype=torch.long)
        observation_type_names: list[str] = []
        prototypes: list[torch.Tensor] = []
        thresholds: list[torch.Tensor] = []
        projections: list[torch.Tensor] = []
        heads: list[AdaptiveObservationHead] = []

        for col, node_id in enumerate(feature_ids):
            head = AdaptiveObservationHead(
                latent_dim=self.latent_dim,
                generator=self.g_dag,
                device=self.device,
                observation_type_probs=self.observation_type_probs,
                categorical_cardinalities=self.categorical_cardinalities,
                categorical_cardinality_probs=self.categorical_cardinality_probs,
                min_samples_per_category=self.min_samples_per_category,
                min_component_weight=self.min_component_weight,
                prototype_max_attempts=self.prototype_max_attempts,
                prototype_min_separation=self.prototype_min_separation,
                binning_jitter=self.binning_jitter,
                observation_noise_scale=self.observation_noise_scale,
            )
            observed = head.observe(
                flat_latents[node_id], generator=self.g_aleatoric
            )
            X[:, col] = observed.values.float()
            feature_type[col] = (
                self.CATEGORICAL if observed.is_categorical else self.CONTINUOUS
            )
            cardinality[col] = observed.cardinality
            quality_score[col] = observed.quality_score
            observation_type_ids[col] = observed.observation_type_id
            observation_type_names.append(observed.observation_type_name)
            prototypes.append(observed.prototypes)
            thresholds.append(observed.thresholds)
            projections.append(observed.projection)
            heads.append(head)

        return (
            X,
            feature_type,
            cardinality,
            quality_score,
            observation_type_ids,
            observation_type_names,
            prototypes,
            thresholds,
            projections,
            heads,
        )

    def _compute_importance_ground_truth(
        self, feature_ids: list[int]
    ) -> dict[str, torch.Tensor]:
        d = len(feature_ids)
        nan = torch.full(
            (d,), float("nan"), device=self.device, dtype=torch.float32
        )
        return {
            "feature_strength": nan.clone(),
            "importance_ratio": nan.clone(),
            "is_active": nan.clone(),
            "sampled_active": nan.clone(),
        }

    def _generate(self):
        device = self.device
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
            root_prior_probs=self.root_prior_probs,
            root_mixture_component_probs=self.root_mixture_component_probs,
            root_mixture_separation_min=self.root_mixture_separation_min,
            root_mixture_separation_max=self.root_mixture_separation_max,
            root_mixture_scale_min=self.root_mixture_scale_min,
            root_mixture_scale_max=self.root_mixture_scale_max,
            dominant_parent_prob=self.dominant_parent_prob,
            dominant_parent_weight=self.dominant_parent_weight,
            linear_activation_prob=self.linear_activation_prob,
            small_mlp_prob=self.small_mlp_prob,
            soft_tree_prob=self.soft_tree_prob,
            small_mlp_hidden_dim=self.small_mlp_hidden_dim,
            soft_tree_depth=self.soft_tree_depth,
            soft_tree_temperature=self.soft_tree_temperature,
            device=device,
        )

        all_latents = scm.forward(
            n_samples=self.n,
            latent_noise_scale=self.latent_noise_scale,
        )
        flat_latents, flat_index = self._flatten_latents(all_latents)
        feature_ids, target_id = self._sample_feature_and_target_sources(
            flat_index, self.d
        )
        self.d = len(feature_ids)

        (
            X_clean,
            feature_type,
            cardinality,
            quality_score,
            observation_type_ids,
            observation_type_names,
            feature_prototypes,
            feature_thresholds,
            feature_projections,
            feature_observation_heads,
        ) = self._observe_features(flat_latents, feature_ids)

        target_head = TargetObservationHead(
            latent_dim=self.latent_dim,
            generator=self.g_dag,
            device=device,
            observation_noise_scale=self.observation_noise_scale,
        )
        target_score = target_head.score(
            flat_latents[target_id], generator=self.g_aleatoric
        )

        if self.num_classes is None:
            y = target_score.float()
            self.n_classes = None
        else:
            num_classes = int(self.num_classes)
            if num_classes < 2:
                raise ValueError("num_classes must be at least 2 or None.")
            y = target_head.balanced_classes(target_score, num_classes)
            self.n_classes = num_classes

        importance_info = self._compute_importance_ground_truth(feature_ids)
        X_obs = X_clean.clone()
        missing_mask = (
            _rand(*X_obs.shape, generator=self.g_x, device=device) < self.p_missing
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
            n_test = max(1, int(round(self.n * self.test_frac)))
            n_test = min(n_test, self.n - 2)
            permutation = torch.randperm(
                self.n, device=device, generator=self.g_x
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
            "feature_observation_type_ids": observation_type_ids,
            "feature_observation_type_names": observation_type_names,
            "feature_observation_quality": quality_score,
            # Compatibility alias used by earlier sanity scripts.
            "categorical_cluster_score": quality_score,
            "feature_prototypes": feature_prototypes,
            "feature_thresholds": feature_thresholds,
            "feature_projections": feature_projections,
            # Compatibility alias: for prototype features these are prototypes;
            # otherwise the tensor is empty.
            "feature_cluster_centers": feature_prototypes,
            **importance_info,
            "missing_mask_train": missing_mask[train_idx],
            "missing_mask_test": missing_mask[test_idx],
            "feature_ids": torch.tensor(
                feature_ids, device=device, dtype=torch.long
            ),
            "target_id": torch.tensor(target_id, device=device, dtype=torch.long),
            "task_edge_prob": torch.tensor(
                scm.task_edge_prob, device=device, dtype=torch.float32
            ),
            "latent_dim": torch.tensor(
                self.latent_dim, device=device, dtype=torch.long
            ),
            "root_prior_types": list(scm.root_prior_types),
            "root_prior_type_ids": torch.tensor(
                scm.root_prior_type_ids, device=device, dtype=torch.long
            ),
            "root_mixture_components": torch.tensor(
                scm.root_mixture_components, device=device, dtype=torch.long
            ),
            "root_prior_probs": torch.tensor(
                self.root_prior_probs, device=device, dtype=torch.float32
            ),
            "root_mixture_component_probs": torch.tensor(
                self.root_mixture_component_probs,
                device=device,
                dtype=torch.float32,
            ),
            "observation_type_probs": torch.tensor(
                self.observation_type_probs, device=device, dtype=torch.float32
            ),
            "categorical_cardinalities": torch.tensor(
                self.categorical_cardinalities, device=device, dtype=torch.long
            ),
            "categorical_cardinality_probs": torch.tensor(
                self.categorical_cardinality_probs,
                device=device,
                dtype=torch.float32,
            ),
        }

        self.n_features = self.d
        self.feature_type = feature_type
        self.cardinality = cardinality
        self.observation_type_ids = observation_type_ids
        self.observation_type_names = observation_type_names
        self.observation_quality = quality_score
        self.feature_prototypes = feature_prototypes
        self.feature_thresholds = feature_thresholds
        self.feature_projections = feature_projections
        self.root_prior_types = list(scm.root_prior_types)
        self.root_prior_type_ids = list(scm.root_prior_type_ids)
        self.root_mixture_components = list(scm.root_mixture_components)
        self.scm = scm
        self.feature_observation_heads = feature_observation_heads
        self.target_observation_head = target_head

        return X_train, y_train, X_test, y_test, info

    def visualize(self):
        return None

    def forward(self, X: torch.Tensor):
        del X
        return None


# Backward-compatible aliases if the rest of the project imports them.
RandomLayeredSCM = RandomLayeredLatentSCM
MixedSCMTask = MixedLatentSCMTask

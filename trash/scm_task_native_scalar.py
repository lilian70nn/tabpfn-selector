
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.data.helper import make_gen, stratified_classification_split
from src.data.synthetic_task import GenerateTask


# ============================================================================
# Random helpers
# ============================================================================


def _randn(*shape, generator, device):
    return torch.randn(
        *shape,
        generator=generator,
        device=device,
    )


def _rand(*shape, generator, device):
    return torch.rand(
        *shape,
        generator=generator,
        device=device,
    )


def _randint(
    low,
    high,
    shape,
    generator,
    device,
):
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
    return (
        x
        - x.mean(
            dim=dim,
            keepdim=True,
        )
    ) / x.std(
        dim=dim,
        unbiased=False,
        keepdim=True,
    ).clamp_min(eps)


def _normalize_probs(
    values,
    device,
    expected_len=None,
    name="probabilities",
):
    probs = torch.tensor(
        values,
        device=device,
        dtype=torch.float32,
    )

    if (
        expected_len is not None
        and probs.numel() != expected_len
    ):
        raise ValueError(
            f"{name} must contain "
            f"{expected_len} values."
        )

    if (
        probs.numel() == 0
        or bool((probs < 0).any())
        or probs.sum() <= 0
    ):
        raise ValueError(
            f"Invalid {name}."
        )

    return probs / probs.sum()


# ============================================================================
# Scalar latent edge: [N] -> [N]
# ============================================================================


class ScalarLatentEdge:
    """
    One scalar parent node is transformed into one scalar contribution.

    Supported mechanisms:
      1. linear + activation
      2. small scalar-input MLP
      3. scalar-input soft tree
    """

    LINEAR = 0
    MLP = 1
    SOFT_TREE = 2

    ACTIVATIONS = (
        "identity",
        "tanh",
        "relu",
        "sigmoid",
        "sin",
        "square",
        "softplus",
    )

    def __init__(
        self,
        generator: torch.Generator,
        device: torch.device,
        linear_activation_prob: float = 0.60,
        small_mlp_prob: float = 0.25,
        soft_tree_prob: float = 0.15,
        small_mlp_hidden_dim: Optional[int] = None,
        soft_tree_depth: int = 2,
        soft_tree_temperature: float = 0.5,
    ):
        self.device = device
        self.soft_tree_depth = int(
            soft_tree_depth
        )
        self.soft_tree_temperature = float(
            soft_tree_temperature
        )

        if self.soft_tree_depth < 1:
            raise ValueError(
                "soft_tree_depth must be at least 1."
            )

        if self.soft_tree_temperature <= 0:
            raise ValueError(
                "soft_tree_temperature must be positive."
            )

        edge_probs = _normalize_probs(
            (
                linear_activation_prob,
                small_mlp_prob,
                soft_tree_prob,
            ),
            device=device,
            expected_len=3,
            name="edge-family probabilities",
        )

        self.edge_type = int(
            torch.multinomial(
                edge_probs,
                1,
                generator=generator,
            ).item()
        )

        # ------------------------------------------------------------------
        # Linear + activation:
        #   y = activation(a * x + b)
        # ------------------------------------------------------------------

        self.linear_weight = _randn(
            (),
            generator=generator,
            device=device,
        )

        self.linear_bias = _randn(
            (),
            generator=generator,
            device=device,
        )

        activation_id = int(
            _randint(
                0,
                len(self.ACTIVATIONS),
                (),
                generator,
                device,
            ).item()
        )

        self.activation_name = self.ACTIVATIONS[
            activation_id
        ]

        # ------------------------------------------------------------------
        # Scalar -> hidden -> scalar MLP
        # ------------------------------------------------------------------

        hidden_dim = (
            int(small_mlp_hidden_dim)
            if small_mlp_hidden_dim is not None
            else 4
        )

        if hidden_dim < 1:
            raise ValueError(
                "small_mlp_hidden_dim must be positive."
            )

        self.mlp_weight_1 = _randn(
            hidden_dim,
            generator=generator,
            device=device,
        )

        self.mlp_bias_1 = _randn(
            hidden_dim,
            generator=generator,
            device=device,
        )

        self.mlp_weight_2 = (
            hidden_dim ** -0.5
        ) * _randn(
            hidden_dim,
            generator=generator,
            device=device,
        )

        self.mlp_bias_2 = _randn(
            (),
            generator=generator,
            device=device,
        )

        # ------------------------------------------------------------------
        # Scalar-input soft tree
        # ------------------------------------------------------------------

        num_internal_nodes = (
            2**self.soft_tree_depth - 1
        )

        num_leaves = (
            2**self.soft_tree_depth
        )

        self.tree_gate_weight = _randn(
            num_internal_nodes,
            generator=generator,
            device=device,
        )

        self.tree_gate_bias = _randn(
            num_internal_nodes,
            generator=generator,
            device=device,
        )

        self.tree_leaf_values = _randn(
            num_leaves,
            generator=generator,
            device=device,
        )

    def _activation(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
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

        raise RuntimeError(
            f"Unknown activation: "
            f"{self.activation_name}"
        )

    def _linear(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._activation(
            self.linear_weight * x
            + self.linear_bias
        )

    def _mlp(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.tanh(
            x[:, None]
            * self.mlp_weight_1[None, :]
            + self.mlp_bias_1[None, :]
        )

        return (
            hidden
            * self.mlp_weight_2[None, :]
        ).sum(
            dim=1
        ) + self.mlp_bias_2

    def _soft_tree(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        logits = (
            x[:, None]
            * self.tree_gate_weight[None, :]
            + self.tree_gate_bias[None, :]
        ) / self.soft_tree_temperature

        right_probability = torch.sigmoid(
            logits
        )

        left_probability = (
            1.0 - right_probability
        )

        path_probability = torch.ones(
            x.shape[0],
            1,
            device=x.device,
            dtype=x.dtype,
        )

        offset = 0

        for depth in range(
            self.soft_tree_depth
        ):
            width = 2**depth

            left = left_probability[
                :,
                offset : offset + width,
            ]

            right = right_probability[
                :,
                offset : offset + width,
            ]

            path_probability = torch.stack(
                (
                    path_probability * left,
                    path_probability * right,
                ),
                dim=-1,
            ).reshape(
                x.shape[0],
                -1,
            )

            offset += width

        return (
            path_probability
            @ self.tree_leaf_values
        )

    def __call__(
        self,
        parent_latent: torch.Tensor,
    ) -> torch.Tensor:
        if parent_latent.ndim != 1:
            raise ValueError(
                "ScalarLatentEdge expects "
                f"shape [N], got "
                f"{tuple(parent_latent.shape)}."
            )

        x = parent_latent.float()

        if self.edge_type == self.LINEAR:
            return self._linear(x)

        if self.edge_type == self.MLP:
            return self._mlp(x)

        return self._soft_tree(x)


# ============================================================================
# Weighted scalar layer connection
# ============================================================================


class WeightedScalarLayerConnection:
    """
    For every child node:

      1. sample its connected parents;
      2. sample a positive structural weight for each parent;
      3. normalize parent weights to sum to one;
      4. transform every parent through its own edge mechanism;
      5. take the weighted sum of all parent contributions.

    Every latent node is a tensor of shape [N].
    """

    def __init__(
        self,
        in_width: int,
        out_width: int,
        connection_prob: float,
        min_parents_per_node: int,
        edge_weight_concentration: float,
        generator: torch.Generator,
        device: torch.device,
        **edge_kwargs,
    ):
        self.in_width = int(
            in_width
        )

        self.out_width = int(
            out_width
        )

        self.device = device

        if self.in_width < 1:
            raise ValueError(
                "in_width must be positive."
            )

        if self.out_width < 1:
            raise ValueError(
                "out_width must be positive."
            )

        if not (
            0.0 <= connection_prob <= 1.0
        ):
            raise ValueError(
                "connection_prob must be in [0, 1]."
            )

        if edge_weight_concentration <= 0:
            raise ValueError(
                "edge_weight_concentration "
                "must be positive."
            )

        self.adj = (
            _rand(
                self.in_width,
                self.out_width,
                generator=generator,
                device=device,
            )
            < float(connection_prob)
        )

        minimum = min(
            int(min_parents_per_node),
            self.in_width,
        )

        for child in range(
            self.out_width
        ):
            current_parent_count = int(
                self.adj[
                    :,
                    child,
                ].sum().item()
            )

            missing = (
                minimum
                - current_parent_count
            )

            if missing <= 0:
                continue

            candidates = torch.where(
                ~self.adj[
                    :,
                    child,
                ]
            )[0]

            order = torch.randperm(
                candidates.numel(),
                generator=generator,
                device=device,
            )

            selected = candidates[
                order[:missing]
            ]

            self.adj[
                selected,
                child,
            ] = True

        self.weights = torch.zeros(
            self.in_width,
            self.out_width,
            device=device,
            dtype=torch.float32,
        )

        self.edges = [
            [
                None
                for _ in range(
                    self.out_width
                )
            ]
            for _ in range(
                self.in_width
            )
        ]

        for child in range(
            self.out_width
        ):
            parents = torch.where(
                self.adj[
                    :,
                    child,
                ]
            )[0]

            concentration = torch.full(
                (
                    parents.numel(),
                ),
                float(
                    edge_weight_concentration
                ),
                device=device,
                dtype=torch.float32,
            )

            raw_weights = (
                torch._standard_gamma(
                    concentration,
                    generator=generator,
                )
                .clamp_min(1e-8)
            )

            normalized_weights = (
                raw_weights
                / raw_weights.sum()
            )

            self.weights[
                parents,
                child,
            ] = normalized_weights

            for parent in parents.tolist():
                self.edges[
                    parent
                ][
                    child
                ] = ScalarLatentEdge(
                    generator=generator,
                    device=device,
                    **edge_kwargs,
                )

    def __call__(
        self,
        parent_latents,
        generator: torch.Generator,
        latent_noise_scale: float = 0.0,
    ):
        if len(parent_latents) != self.in_width:
            raise ValueError(
                "Number of parent latent tensors "
                f"is {len(parent_latents)}, "
                f"expected {self.in_width}."
            )

        children = []

        for child in range(
            self.out_width
        ):
            child_value = None

            for parent in range(
                self.in_width
            ):
                edge = self.edges[
                    parent
                ][
                    child
                ]

                if edge is None:
                    continue

                contribution = (
                    self.weights[
                        parent,
                        child,
                    ]
                    * edge(
                        parent_latents[
                            parent
                        ]
                    )
                )

                child_value = (
                    contribution
                    if child_value is None
                    else child_value
                    + contribution
                )

            if child_value is None:
                raise RuntimeError(
                    f"Child {child} has no parents."
                )

            child_value = _standardize(
                child_value,
                dim=0,
            )

            if latent_noise_scale > 0:
                child_value = (
                    child_value
                    + float(
                        latent_noise_scale
                    )
                    * torch.randn(
                        child_value.shape,
                        generator=generator,
                        device=self.device,
                        dtype=child_value.dtype,
                    )
                )

                child_value = _standardize(
                    child_value,
                    dim=0,
                )

            children.append(
                child_value
            )

        return children


# ============================================================================
# Full scalar SCM
# ============================================================================


class WeightedLayeredScalarSCM:
    ROOT_PRIORS = (
        "gaussian",
        "uniform",
        "heavy_tailed",
        "skewed",
        "mixture",
    )

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
        connection_probs=(
            0.30,
            0.45,
            0.65,
            0.85,
        ),
        min_parents_per_node=2,
        edge_weight_concentration=0.60,
        latent_noise_scale=0.03,
        root_prior_probs=(
            0.45,
            0.20,
            0.15,
            0.05,
            0.15,
        ),
        root_mixture_component_probs=(
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),
        root_mixture_separation_min=1.5,
        root_mixture_separation_max=3.0,
        root_mixture_scale_min=0.40,
        root_mixture_scale_max=0.90,
        device=None,
        **edge_kwargs,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cpu")
        )

        self.g_dag = g_dag
        self.g_x = g_x
        self.g_aleatoric = g_aleatoric

        self.num_roots = int(
            num_roots
        )

        self.num_layers = int(
            num_layers
        )

        self.hidden_width_min = int(
            hidden_width_min
        )

        self.hidden_width_max = int(
            hidden_width_max
        )

        self.final_width = int(
            final_width
        )

        self.connection_probs = tuple(
            float(probability)
            for probability in connection_probs
        )

        self.latent_noise_scale = float(
            latent_noise_scale
        )

        if self.num_roots < 1:
            raise ValueError(
                "num_roots must be positive."
            )

        if self.num_layers < 2:
            raise ValueError(
                "num_layers must be at least 2."
            )

        if (
            self.hidden_width_min < 1
            or self.hidden_width_max
            < self.hidden_width_min
        ):
            raise ValueError(
                "Invalid hidden width range."
            )

        if self.final_width < 1:
            raise ValueError(
                "final_width must be positive."
            )

        if (
            len(self.connection_probs)
            != self.num_layers - 1
        ):
            raise ValueError(
                "connection_probs must contain "
                "num_layers - 1 values."
            )

        self.root_prior_probs = _normalize_probs(
            root_prior_probs,
            device=self.device,
            expected_len=5,
            name="root_prior_probs",
        )

        self.root_mixture_component_probs = (
            _normalize_probs(
                root_mixture_component_probs,
                device=self.device,
                expected_len=5,
                name=(
                    "root_mixture_component_probs"
                ),
            )
        )

        self.root_mixture_separation_min = float(
            root_mixture_separation_min
        )

        self.root_mixture_separation_max = float(
            root_mixture_separation_max
        )

        self.root_mixture_scale_min = float(
            root_mixture_scale_min
        )

        self.root_mixture_scale_max = float(
            root_mixture_scale_max
        )

        if (
            self.root_mixture_separation_min
            <= 0
            or self.root_mixture_separation_max
            < self.root_mixture_separation_min
        ):
            raise ValueError(
                "Invalid mixture separation range."
            )

        if (
            self.root_mixture_scale_min
            <= 0
            or self.root_mixture_scale_max
            < self.root_mixture_scale_min
        ):
            raise ValueError(
                "Invalid mixture scale range."
            )

        self.widths = [
            self.num_roots
        ]

        for _ in range(
            self.num_layers - 2
        ):
            width = int(
                _randint(
                    self.hidden_width_min,
                    self.hidden_width_max + 1,
                    (),
                    self.g_dag,
                    self.device,
                ).item()
            )

            self.widths.append(
                width
            )

        self.widths.append(
            self.final_width
        )

        self.connections = []

        for layer in range(
            self.num_layers - 1
        ):
            connection = (
                WeightedScalarLayerConnection(
                    in_width=self.widths[
                        layer
                    ],
                    out_width=self.widths[
                        layer + 1
                    ],
                    connection_prob=(
                        self.connection_probs[
                            layer
                        ]
                    ),
                    min_parents_per_node=(
                        min_parents_per_node
                    ),
                    edge_weight_concentration=(
                        edge_weight_concentration
                    ),
                    generator=self.g_dag,
                    device=self.device,
                    **edge_kwargs,
                )
            )

            self.connections.append(
                connection
            )

        self.root_prior_types = []
        self.root_prior_type_ids = []
        self.root_mixture_components = []

    def _sample_mixture_root(
        self,
        n: int,
    ):
        component_index = int(
            torch.multinomial(
                self.root_mixture_component_probs,
                1,
                generator=self.g_dag,
            ).item()
        )

        num_components = (
            component_index + 2
        )

        component_weights = torch.softmax(
            0.5
            * _randn(
                num_components,
                generator=self.g_dag,
                device=self.device,
            ),
            dim=0,
        )

        component_ids = torch.multinomial(
            component_weights,
            n,
            replacement=True,
            generator=self.g_x,
        )

        separation = (
            self.root_mixture_separation_min
            + (
                self.root_mixture_separation_max
                - self.root_mixture_separation_min
            )
            * _rand(
                (),
                generator=self.g_dag,
                device=self.device,
            )
        )

        # In one dimension, use distinct ordered centers.
        centers = separation * torch.linspace(
            -1.0,
            1.0,
            steps=num_components,
            device=self.device,
        )

        scales = (
            self.root_mixture_scale_min
            + (
                self.root_mixture_scale_max
                - self.root_mixture_scale_min
            )
            * _rand(
                num_components,
                generator=self.g_dag,
                device=self.device,
            )
        )

        noise = _randn(
            n,
            generator=self.g_x,
            device=self.device,
        )

        values = (
            centers[
                component_ids
            ]
            + scales[
                component_ids
            ]
            * noise
        )

        return (
            values,
            num_components,
        )

    def sample_root_latents(
        self,
        n: int,
    ):
        roots = []

        self.root_prior_types = []
        self.root_prior_type_ids = []
        self.root_mixture_components = []

        for _ in range(
            self.num_roots
        ):
            prior_id = int(
                torch.multinomial(
                    self.root_prior_probs,
                    1,
                    generator=self.g_dag,
                ).item()
            )

            prior_name = self.ROOT_PRIORS[
                prior_id
            ]

            mixture_components = 0

            if prior_name == "gaussian":
                root = _randn(
                    n,
                    generator=self.g_x,
                    device=self.device,
                )

            elif prior_name == "uniform":
                bound = 3.0**0.5

                root = (
                    2.0
                    * bound
                    * _rand(
                        n,
                        generator=self.g_x,
                        device=self.device,
                    )
                    - bound
                )

            elif prior_name == "heavy_tailed":
                degrees_of_freedom = 4.0

                numerator = _randn(
                    n,
                    generator=self.g_x,
                    device=self.device,
                )

                concentration = torch.full(
                    (n,),
                    degrees_of_freedom / 2.0,
                    device=self.device,
                    dtype=numerator.dtype,
                )

                chi_square = (
                    2.0
                    * torch._standard_gamma(
                        concentration,
                        generator=self.g_x,
                    )
                )

                root = numerator / torch.sqrt(
                    chi_square
                    / degrees_of_freedom
                ).clamp_min(
                    1e-4
                )

            elif prior_name == "skewed":
                normal = _randn(
                    n,
                    generator=self.g_x,
                    device=self.device,
                )

                strength = (
                    0.4
                    + 0.6
                    * _rand(
                        (),
                        generator=self.g_dag,
                        device=self.device,
                    )
                )

                root = torch.exp(
                    normal * strength
                )

            else:
                (
                    root,
                    mixture_components,
                ) = self._sample_mixture_root(
                    n
                )

            root = _standardize(
                root.float(),
                dim=0,
            )

            roots.append(
                root
            )

            self.root_prior_types.append(
                prior_name
            )

            self.root_prior_type_ids.append(
                prior_id
            )

            self.root_mixture_components.append(
                mixture_components
            )

        return roots

    def forward(
        self,
        n_samples: int,
        latent_noise_scale=None,
    ):
        current_layer = self.sample_root_latents(
            n_samples
        )

        all_latents = [
            current_layer
        ]

        noise_scale = (
            self.latent_noise_scale
            if latent_noise_scale is None
            else float(latent_noise_scale)
        )

        for connection in self.connections:
            current_layer = connection(
                current_layer,
                generator=self.g_aleatoric,
                latent_noise_scale=noise_scale,
            )

            all_latents.append(
                current_layer
            )

        return all_latents

    def compute_node_influence(
        self,
        target_node_idx: int = 0,
    ):
        """
        Structural path score:

            influence(parent)
              = sum_child [
                    edge_weight(parent, child)
                    * influence(child)
                ]

        This sums products of structural edge weights
        over all directed paths to the target.
        """

        if not (
            0
            <= target_node_idx
            < self.widths[-1]
        ):
            raise ValueError(
                "target_node_idx is outside "
                "the final layer."
            )

        influence = [
            torch.zeros(
                width,
                device=self.device,
                dtype=torch.float32,
            )
            for width in self.widths
        ]

        influence[-1][
            target_node_idx
        ] = 1.0

        for layer in range(
            self.num_layers - 2,
            -1,
            -1,
        ):
            influence[
                layer
            ] = (
                self.connections[
                    layer
                ].weights
                @ influence[
                    layer + 1
                ]
            )

        return influence


# ============================================================================
# Observed feature conversion
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


class ScalarFeatureObservation:
    """
    A latent node is already scalar.

    CONTINUOUS:
        return the scalar node directly, optionally with noise.

    PROTOTYPE:
        turn the scalar node into categorical labels
        using nearest sampled scalar prototypes.

    BINNING:
        turn the scalar node into categorical labels
        using thresholds.

    No projection is used anywhere.
    """

    CONTINUOUS = 0
    PROTOTYPE = 1
    BINNING = 2

    NAMES = (
        "continuous_scalar",
        "prototype_discretization",
        "threshold_binning",
    )

    def __init__(
        self,
        generator: torch.Generator,
        device: torch.device,
        observation_type_probs=(
            0.60,
            0.20,
            0.20,
        ),
        categorical_cardinalities=(
            2,
            3,
            4,
            5,
            6,
        ),
        categorical_cardinality_probs=(
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),
        min_samples_per_category=8,
        min_component_weight=0.05,
        prototype_max_attempts=8,
        prototype_min_separation=1.0,
        binning_jitter=0.20,
        observation_noise_scale=0.05,
    ):
        self.device = device

        self.min_samples_per_category = int(
            min_samples_per_category
        )

        self.min_component_weight = float(
            min_component_weight
        )

        self.prototype_max_attempts = int(
            prototype_max_attempts
        )

        self.prototype_min_separation = float(
            prototype_min_separation
        )

        self.binning_jitter = float(
            binning_jitter
        )

        self.observation_noise_scale = float(
            observation_noise_scale
        )

        self.observation_type_probs = (
            _normalize_probs(
                observation_type_probs,
                device=device,
                expected_len=3,
                name="observation_type_probs",
            )
        )

        self.cardinalities = tuple(
            int(cardinality)
            for cardinality
            in categorical_cardinalities
        )

        if any(
            cardinality < 2
            for cardinality in self.cardinalities
        ):
            raise ValueError(
                "All categorical cardinalities "
                "must be at least 2."
            )

        self.cardinality_probs = (
            _normalize_probs(
                categorical_cardinality_probs,
                device=device,
                expected_len=len(
                    self.cardinalities
                ),
                name=(
                    "categorical_cardinality_probs"
                ),
            )
        )

        self.sampled_type = int(
            torch.multinomial(
                self.observation_type_probs,
                1,
                generator=generator,
            ).item()
        )

    def _sample_cardinality(
        self,
        n: int,
        generator: torch.Generator,
    ) -> int:
        feasible_positions = [
            position
            for position, cardinality
            in enumerate(
                self.cardinalities
            )
            if (
                cardinality
                * self.min_samples_per_category
                <= n
            )
            and (
                cardinality
                * self.min_component_weight
                <= 1.0
            )
        ]

        if not feasible_positions:
            return 0

        feasible_tensor = torch.tensor(
            feasible_positions,
            device=self.device,
            dtype=torch.long,
        )

        probabilities = self.cardinality_probs[
            feasible_tensor
        ]

        probabilities = (
            probabilities
            / probabilities.sum()
        )

        sampled_position = int(
            torch.multinomial(
                probabilities,
                1,
                generator=generator,
            ).item()
        )

        return self.cardinalities[
            feasible_positions[
                sampled_position
            ]
        ]

    def _continuous(
        self,
        scalar: torch.Tensor,
        generator: torch.Generator,
        name: str = "continuous_scalar",
    ) -> FeatureObservation:
        values = scalar.float().clone()

        if self.observation_noise_scale > 0:
            values = (
                values
                + self.observation_noise_scale
                * torch.randn(
                    values.shape,
                    generator=generator,
                    device=values.device,
                    dtype=values.dtype,
                )
            )

        values = _standardize(
            values,
            dim=0,
        )

        return FeatureObservation(
            values=values,
            is_categorical=False,
            cardinality=0,
            observation_type_id=(
                self.CONTINUOUS
            ),
            observation_type_name=name,
            quality_score=0.0,
            prototypes=torch.empty(
                0,
                device=values.device,
                dtype=values.dtype,
            ),
            thresholds=torch.empty(
                0,
                device=values.device,
                dtype=values.dtype,
            ),
            projection=torch.empty(
                0,
                device=values.device,
                dtype=values.dtype,
            ),
        )

    def _sample_prototypes(
        self,
        scalar: torch.Tensor,
        cardinality: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Try several random prototype sets and retain
        the set with the largest minimum separation.
        """

        best_prototypes = None
        best_separation = None

        attempts = max(
            1,
            self.prototype_max_attempts,
        )

        for _ in range(attempts):
            indices = torch.randperm(
                scalar.numel(),
                generator=generator,
                device=scalar.device,
            )[
                :cardinality
            ]

            prototypes = scalar[
                indices
            ]

            if cardinality == 1:
                minimum_separation = torch.tensor(
                    float("inf"),
                    device=scalar.device,
                )
            else:
                distances = torch.abs(
                    prototypes[:, None]
                    - prototypes[None, :]
                )

                distances.fill_diagonal_(
                    float("inf")
                )

                minimum_separation = (
                    distances.min()
                )

            if (
                best_separation is None
                or minimum_separation
                > best_separation
            ):
                best_prototypes = prototypes
                best_separation = (
                    minimum_separation
                )

            if (
                float(
                    minimum_separation.item()
                )
                >= self.prototype_min_separation
            ):
                break

        return best_prototypes

    def _prototype(
        self,
        scalar: torch.Tensor,
        generator: torch.Generator,
    ) -> FeatureObservation:
        cardinality = self._sample_cardinality(
            scalar.numel(),
            generator,
        )

        if cardinality == 0:
            return self._continuous(
                scalar,
                generator,
                name=(
                    "continuous_fallback_from_prototype"
                ),
            )

        prototypes = self._sample_prototypes(
            scalar,
            cardinality,
            generator,
        )

        distances = torch.abs(
            scalar[:, None]
            - prototypes[None, :]
        )

        labels = distances.argmin(
            dim=1
        )

        category_counts = torch.bincount(
            labels,
            minlength=cardinality,
        )

        quality = float(
            (
                category_counts.float()
                / category_counts.sum().clamp_min(
                    1
                )
            ).min().item()
        )

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=cardinality,
            observation_type_id=(
                self.PROTOTYPE
            ),
            observation_type_name=(
                "prototype_discretization"
            ),
            quality_score=quality,
            prototypes=prototypes,
            thresholds=torch.empty(
                0,
                device=scalar.device,
                dtype=scalar.dtype,
            ),
            projection=torch.empty(
                0,
                device=scalar.device,
                dtype=scalar.dtype,
            ),
        )

    def _binning(
        self,
        scalar: torch.Tensor,
        generator: torch.Generator,
    ) -> FeatureObservation:
        cardinality = self._sample_cardinality(
            scalar.numel(),
            generator,
        )

        if cardinality == 0:
            return self._continuous(
                scalar,
                generator,
                name=(
                    "continuous_fallback_from_binning"
                ),
            )

        scalar = _standardize(
            scalar.float(),
            dim=0,
        )

        n = scalar.numel()

        minimum_count = max(
            self.min_samples_per_category,
            int(
                torch.ceil(
                    torch.tensor(
                        self.min_component_weight
                        * n,
                        device=scalar.device,
                    )
                ).item()
            ),
        )

        remaining = (
            n
            - cardinality
            * minimum_count
        )

        if remaining < 0:
            return self._continuous(
                scalar,
                generator,
                name=(
                    "continuous_fallback_from_binning"
                ),
            )

        random_mass = _rand(
            cardinality,
            generator=generator,
            device=scalar.device,
        )

        extra_counts_float = (
            random_mass
            / random_mass.sum().clamp_min(
                1e-12
            )
            * remaining
        )

        extra_counts = torch.floor(
            extra_counts_float
        ).long()

        leftover = (
            remaining
            - int(
                extra_counts.sum().item()
            )
        )

        if leftover > 0:
            residual_order = torch.argsort(
                extra_counts_float
                - extra_counts.float(),
                descending=True,
            )

            extra_counts[
                residual_order[:leftover]
            ] += 1

        counts = (
            extra_counts
            + minimum_count
        ).tolist()

        sorted_values = torch.sort(
            scalar
        ).values

        thresholds = []

        cumulative_count = 0

        for count in counts[:-1]:
            cumulative_count += int(
                count
            )

            lower = sorted_values[
                cumulative_count - 1
            ]

            upper = sorted_values[
                cumulative_count
            ]

            threshold = (
                0.5
                * (
                    lower + upper
                )
            )

            if self.binning_jitter > 0:
                local_gap = (
                    upper - lower
                ).abs()

                jitter = (
                    2.0
                    * _rand(
                        (),
                        generator=generator,
                        device=scalar.device,
                    )
                    - 1.0
                )

                threshold = (
                    threshold
                    + self.binning_jitter
                    * local_gap
                    * jitter
                )

                threshold = torch.minimum(
                    torch.maximum(
                        threshold,
                        lower,
                    ),
                    upper,
                )

            thresholds.append(
                threshold
            )

        thresholds = torch.stack(
            thresholds
        )

        thresholds = torch.sort(
            thresholds
        ).values

        labels = torch.bucketize(
            scalar,
            thresholds,
        )

        category_counts = torch.bincount(
            labels,
            minlength=cardinality,
        )

        quality = float(
            (
                category_counts.float()
                / category_counts.sum().clamp_min(
                    1
                )
            ).min().item()
        )

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=cardinality,
            observation_type_id=(
                self.BINNING
            ),
            observation_type_name=(
                "threshold_binning"
            ),
            quality_score=quality,
            prototypes=torch.empty(
                0,
                device=scalar.device,
                dtype=scalar.dtype,
            ),
            thresholds=thresholds,
            projection=torch.empty(
                0,
                device=scalar.device,
                dtype=scalar.dtype,
            ),
        )

    def observe(
        self,
        scalar_latent: torch.Tensor,
        generator: torch.Generator,
    ) -> FeatureObservation:
        if scalar_latent.ndim != 1:
            raise ValueError(
                "ScalarFeatureObservation expects "
                f"shape [N], got "
                f"{tuple(scalar_latent.shape)}."
            )

        scalar = _standardize(
            scalar_latent.float(),
            dim=0,
        )

        if self.sampled_type == self.CONTINUOUS:
            return self._continuous(
                scalar,
                generator,
            )

        if self.sampled_type == self.PROTOTYPE:
            return self._prototype(
                scalar,
                generator,
            )

        return self._binning(
            scalar,
            generator,
        )


# ============================================================================
# Target handling
# ============================================================================


class ScalarTargetHead:
    """
    The target node is already scalar.

    No projection is used.
    """

    def __init__(
        self,
        device: torch.device,
        observation_noise_scale: float = 0.03,
    ):
        self.device = device

        self.observation_noise_scale = float(
            observation_noise_scale
        )

    def score(
        self,
        scalar_latent: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if scalar_latent.ndim != 1:
            raise ValueError(
                "ScalarTargetHead expects "
                f"shape [N], got "
                f"{tuple(scalar_latent.shape)}."
            )

        score = scalar_latent.float().clone()

        if self.observation_noise_scale > 0:
            score = (
                score
                + self.observation_noise_scale
                * torch.randn(
                    score.shape,
                    generator=generator,
                    device=self.device,
                    dtype=score.dtype,
                )
            )

        return _standardize(
            score,
            dim=0,
        )

    @staticmethod
    def balanced_classes(
        score: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2."
            )

        order = torch.argsort(
            score
        )

        labels = torch.empty_like(
            order
        )

        n = score.numel()
        start = 0

        for class_id in range(
            num_classes
        ):
            class_size = (
                n // num_classes
                + (
                    1
                    if class_id
                    < n % num_classes
                    else 0
                )
            )

            labels[
                order[
                    start
                    : start + class_size
                ]
            ] = class_id

            start += class_size

        return labels.long()


# ============================================================================
# Task
# ============================================================================


class WeightedMixedScalarSCMTask(
    GenerateTask
):
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
        connection_probs=(
            0.30,
            0.45,
            0.65,
            0.85,
        ),
        min_parents_per_node=2,
        edge_weight_concentration=0.60,
        latent_noise_scale=0.03,
        observation_noise_scale=0.03,
        dominant_mass_threshold=0.70,
        dominant_feature_fraction=0.70,
        observation_type_probs=(
            0.60,
            0.20,
            0.20,
        ),
        categorical_cardinalities=(
            2,
            3,
            4,
            5,
            6,
        ),
        categorical_cardinality_probs=(
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),
        min_samples_per_category=8,
        min_component_weight=0.05,
        prototype_max_attempts=8,
        prototype_min_separation=1.0,
        binning_jitter=0.20,
        root_prior_probs=(
            0.45,
            0.20,
            0.15,
            0.05,
            0.15,
        ),
        root_mixture_component_probs=(
            0.40,
            0.30,
            0.18,
            0.08,
            0.04,
        ),
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
        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.num_classes = (
            None
            if num_classes is None
            else int(num_classes)
        )

        if (
            self.num_classes is not None
            and self.num_classes < 2
        ):
            raise ValueError(
                "num_classes must be None "
                "or at least 2."
            )

        self.n_min = int(
            n_min
        )

        self.n_max = int(
            n_max
        )

        self.d_min = int(
            d_min
        )

        self.d_max = int(
            d_max
        )

        self.test_frac = float(
            test_frac
        )

        self.p_missing = float(
            p_missing
        )

        self.latent_noise_scale = float(
            latent_noise_scale
        )

        self.observation_noise_scale = float(
            observation_noise_scale
        )

        self.dominant_mass_threshold = float(
            dominant_mass_threshold
        )

        self.dominant_feature_fraction = float(
            dominant_feature_fraction
        )

        if (
            self.n_min < 3
            or self.n_max < self.n_min
        ):
            raise ValueError(
                "Invalid sample-size range."
            )

        if (
            self.d_min < 1
            or self.d_max < self.d_min
        ):
            raise ValueError(
                "Invalid feature-count range."
            )

        if not (
            0.0 < self.test_frac < 1.0
        ):
            raise ValueError(
                "test_frac must be in (0, 1)."
            )

        if not (
            0.0 <= self.p_missing <= 1.0
        ):
            raise ValueError(
                "p_missing must be in [0, 1]."
            )

        if not (
            0.0
            <= self.dominant_mass_threshold
            <= 1.0
        ):
            raise ValueError(
                "dominant_mass_threshold "
                "must be in [0, 1]."
            )

        if not (
            0.0
            <= self.dominant_feature_fraction
            <= 1.0
        ):
            raise ValueError(
                "dominant_feature_fraction "
                "must be in [0, 1]."
            )

        self.scm_kwargs = dict(
            num_roots=num_roots,
            num_layers=num_layers,
            hidden_width_min=(
                hidden_width_min
            ),
            hidden_width_max=(
                hidden_width_max
            ),
            final_width=final_width,
            connection_probs=(
                connection_probs
            ),
            min_parents_per_node=(
                min_parents_per_node
            ),
            edge_weight_concentration=(
                edge_weight_concentration
            ),
            latent_noise_scale=(
                latent_noise_scale
            ),
            root_prior_probs=(
                root_prior_probs
            ),
            root_mixture_component_probs=(
                root_mixture_component_probs
            ),
            root_mixture_separation_min=(
                root_mixture_separation_min
            ),
            root_mixture_separation_max=(
                root_mixture_separation_max
            ),
            root_mixture_scale_min=(
                root_mixture_scale_min
            ),
            root_mixture_scale_max=(
                root_mixture_scale_max
            ),
            linear_activation_prob=(
                linear_activation_prob
            ),
            small_mlp_prob=(
                small_mlp_prob
            ),
            soft_tree_prob=(
                soft_tree_prob
            ),
            small_mlp_hidden_dim=(
                small_mlp_hidden_dim
            ),
            soft_tree_depth=(
                soft_tree_depth
            ),
            soft_tree_temperature=(
                soft_tree_temperature
            ),
            device=self.device,
        )

        self.observation_kwargs = dict(
            observation_type_probs=(
                observation_type_probs
            ),
            categorical_cardinalities=(
                categorical_cardinalities
            ),
            categorical_cardinality_probs=(
                categorical_cardinality_probs
            ),
            min_samples_per_category=(
                min_samples_per_category
            ),
            min_component_weight=(
                min_component_weight
            ),
            prototype_max_attempts=(
                prototype_max_attempts
            ),
            prototype_min_separation=(
                prototype_min_separation
            ),
            binning_jitter=(
                binning_jitter
            ),
            observation_noise_scale=(
                observation_noise_scale
            ),
        )

        (
            self.g_dag,
            self.dag_seed,
        ) = make_gen(
            self.device,
            dag_seed,
        )

        (
            self.g_aleatoric,
            self.aleatoric_seed,
        ) = make_gen(
            self.device,
            aleatoric_seed,
        )

        (
            self.g_x,
            self.x_seed,
        ) = make_gen(
            self.device,
            x_seed,
        )

        self.n = int(
            _randint(
                self.n_min,
                self.n_max + 1,
                (),
                self.g_dag,
                self.device,
            ).item()
        )

        self.d = int(
            _randint(
                self.d_min,
                self.d_max + 1,
                (),
                self.g_dag,
                self.device,
            ).item()
        )

        super().__init__()

    @staticmethod
    def _flatten(
        all_latents,
    ):
        flat_values = []
        flat_index = []

        for layer_index, layer in enumerate(
            all_latents
        ):
            for node_index, value in enumerate(
                layer
            ):
                flat_values.append(
                    value
                )

                flat_index.append(
                    (
                        layer_index,
                        node_index,
                    )
                )

        return (
            flat_values,
            flat_index,
        )

    def _dominant_group(
        self,
        candidate_ids: torch.Tensor,
        flat_influence: torch.Tensor,
    ) -> torch.Tensor:
        influence = flat_influence[
            candidate_ids
        ]

        active_mask = (
            influence > 0
        )

        ids = candidate_ids[
            active_mask
        ]

        influence = influence[
            active_mask
        ]

        if ids.numel() == 0:
            return torch.empty(
                0,
                device=self.device,
                dtype=torch.long,
            )

        order = torch.argsort(
            influence,
            descending=True,
        )

        ids = ids[
            order
        ]

        influence = influence[
            order
        ]

        cutoff = (
            self.dominant_mass_threshold
            * influence.sum()
        )

        cumulative = torch.cumsum(
            influence,
            dim=0,
        )

        reached = torch.where(
            cumulative >= cutoff
        )[0]

        group_size = (
            int(
                reached[0].item()
            )
            + 1
            if reached.numel() > 0
            else ids.numel()
        )

        return ids[
            :group_size
        ]

    def _sample_without_replacement(
        self,
        ids: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        if (
            count <= 0
            or ids.numel() == 0
        ):
            return torch.empty(
                0,
                device=self.device,
                dtype=torch.long,
            )

        count = min(
            int(count),
            int(
                ids.numel()
            ),
        )

        positions = torch.randperm(
            ids.numel(),
            generator=self.g_dag,
            device=ids.device,
        )[
            :count
        ]

        return ids[
            positions
        ]

    def _sample_feature_ids(
        self,
        flat_index,
        flat_influence: torch.Tensor,
    ):
        candidate_ids = torch.tensor(
            [
                flat_id
                for flat_id, (
                    layer_index,
                    _,
                )
                in enumerate(
                    flat_index
                )
                if (
                    layer_index
                    < len(
                        self.scm.widths
                    )
                    - 1
                )
            ],
            device=self.device,
            dtype=torch.long,
        )

        num_features = min(
            self.d,
            candidate_ids.numel(),
        )

        dominant_ids = self._dominant_group(
            candidate_ids,
            flat_influence,
        )

        dominant_set = set(
            dominant_ids.tolist()
        )

        other_ids = torch.tensor(
            [
                candidate_id
                for candidate_id
                in candidate_ids.tolist()
                if (
                    candidate_id
                    not in dominant_set
                )
            ],
            device=self.device,
            dtype=torch.long,
        )

        num_dominant = min(
            round(
                self.dominant_feature_fraction
                * num_features
            ),
            dominant_ids.numel(),
        )

        num_other = (
            num_features
            - num_dominant
        )

        if (
            num_other
            > other_ids.numel()
        ):
            num_dominant += (
                num_other
                - other_ids.numel()
            )

            num_other = (
                other_ids.numel()
            )

        selected_dominant = (
            self._sample_without_replacement(
                dominant_ids,
                num_dominant,
            )
        )

        selected_other = (
            self._sample_without_replacement(
                other_ids,
                num_other,
            )
        )

        selected = torch.cat(
            (
                selected_dominant,
                selected_other,
            )
        )

        if selected.numel() < num_features:
            selected_set = set(
                selected.tolist()
            )

            remaining_ids = torch.tensor(
                [
                    candidate_id
                    for candidate_id
                    in candidate_ids.tolist()
                    if (
                        candidate_id
                        not in selected_set
                    )
                ],
                device=self.device,
                dtype=torch.long,
            )

            fill = (
                self._sample_without_replacement(
                    remaining_ids,
                    num_features
                    - selected.numel(),
                )
            )

            selected = torch.cat(
                (
                    selected,
                    fill,
                )
            )

        selected = selected[
            torch.randperm(
                selected.numel(),
                generator=self.g_dag,
                device=self.device,
            )
        ]

        return (
            selected.tolist(),
            dominant_ids,
        )

    def _observe_features(
        self,
        flat_latents,
        feature_ids,
    ):
        num_samples = flat_latents[
            0
        ].shape[
            0
        ]

        num_features = len(
            feature_ids
        )

        X = torch.empty(
            num_samples,
            num_features,
            device=self.device,
            dtype=torch.float32,
        )

        feature_type = torch.empty(
            num_features,
            device=self.device,
            dtype=torch.long,
        )

        cardinality = torch.zeros(
            num_features,
            device=self.device,
            dtype=torch.long,
        )

        observation_type_ids = torch.empty(
            num_features,
            device=self.device,
            dtype=torch.long,
        )

        observation_type_names = []

        quality = torch.zeros(
            num_features,
            device=self.device,
            dtype=torch.float32,
        )

        prototypes = []
        thresholds = []
        projections = []
        heads = []

        for column, node_id in enumerate(
            feature_ids
        ):
            head = ScalarFeatureObservation(
                generator=self.g_dag,
                device=self.device,
                **self.observation_kwargs,
            )

            observed = head.observe(
                flat_latents[
                    node_id
                ],
                self.g_aleatoric,
            )

            X[
                :,
                column,
            ] = observed.values.float()

            feature_type[
                column
            ] = (
                self.CATEGORICAL
                if observed.is_categorical
                else self.CONTINUOUS
            )

            cardinality[
                column
            ] = observed.cardinality

            observation_type_ids[
                column
            ] = (
                observed.observation_type_id
            )

            observation_type_names.append(
                observed.observation_type_name
            )

            quality[
                column
            ] = observed.quality_score

            prototypes.append(
                observed.prototypes
            )

            thresholds.append(
                observed.thresholds
            )

            projections.append(
                observed.projection
            )

            heads.append(
                head
            )

        return (
            X,
            feature_type,
            cardinality,
            observation_type_ids,
            observation_type_names,
            quality,
            prototypes,
            thresholds,
            projections,
            heads,
        )

    def _generate(
        self,
    ):
        self.scm = WeightedLayeredScalarSCM(
            self.g_dag,
            self.g_x,
            self.g_aleatoric,
            **self.scm_kwargs,
        )

        all_latents = self.scm.forward(
            self.n,
            latent_noise_scale=(
                self.latent_noise_scale
            ),
        )

        (
            flat_latents,
            flat_index,
        ) = self._flatten(
            all_latents
        )

        layer_influence = (
            self.scm.compute_node_influence(
                target_node_idx=0
            )
        )

        flat_influence = torch.cat(
            layer_influence
        )

        (
            feature_ids,
            dominant_group,
        ) = self._sample_feature_ids(
            flat_index,
            flat_influence,
        )

        self.d = len(
            feature_ids
        )

        (
            X_clean,
            feature_type,
            cardinality,
            observation_type_ids,
            observation_type_names,
            observation_quality,
            prototypes,
            thresholds,
            projections,
            observation_heads,
        ) = self._observe_features(
            flat_latents,
            feature_ids,
        )

        target_global_id = sum(
            self.scm.widths[
                :-1
            ]
        )

        target_head = ScalarTargetHead(
            device=self.device,
            observation_noise_scale=(
                self.observation_noise_scale
            ),
        )

        target_score = target_head.score(
            flat_latents[
                target_global_id
            ],
            self.g_aleatoric,
        )

        if self.num_classes is None:
            y = target_score

            self.n_classes = None

        else:
            y = target_head.balanced_classes(
                target_score,
                self.num_classes,
            )

            self.n_classes = (
                self.num_classes
            )

        feature_ids_tensor = torch.tensor(
            feature_ids,
            device=self.device,
            dtype=torch.long,
        )

        feature_strength = flat_influence[
            feature_ids_tensor
        ]

        importance_ratio = (
            feature_strength
            / feature_strength.sum().clamp_min(
                1e-12
            )
        )

        dominant_set = set(
            dominant_group.tolist()
        )

        selected_from_dominant = (
            torch.tensor(
                [
                    float(
                        feature_id
                        in dominant_set
                    )
                    for feature_id
                    in feature_ids
                ],
                device=self.device,
                dtype=torch.float32,
            )
        )

        X_observed = X_clean.clone()

        missing_mask = (
            _rand(
                *X_observed.shape,
                generator=self.g_x,
                device=self.device,
            )
            < self.p_missing
        )

        X_observed[
            missing_mask
        ] = torch.nan

        if self.num_classes is not None:
            (
                train_indices,
                test_indices,
            ) = stratified_classification_split(
                y=y.long(),
                test_frac=self.test_frac,
                generator=self.g_x,
                device=self.device,
            )

        else:
            num_test = min(
                max(
                    1,
                    round(
                        self.n
                        * self.test_frac
                    ),
                ),
                self.n - 2,
            )

            order = torch.randperm(
                self.n,
                generator=self.g_x,
                device=self.device,
            )

            train_indices = order[
                :-num_test
            ]

            test_indices = order[
                -num_test:
            ]

        info = {
            "feature_type": (
                feature_type
            ),
            "cardinality": (
                cardinality
            ),
            "feature_observation_type_ids": (
                observation_type_ids
            ),
            "feature_observation_type_names": (
                observation_type_names
            ),
            "feature_observation_quality": (
                observation_quality
            ),
            "feature_prototypes": (
                prototypes
            ),
            "feature_thresholds": (
                thresholds
            ),
            "feature_projections": (
                projections
            ),
            "feature_ids": (
                feature_ids_tensor
            ),
            "target_id": torch.tensor(
                target_global_id,
                device=self.device,
                dtype=torch.long,
            ),
            "feature_strength": (
                feature_strength
            ),
            "importance_ratio": (
                importance_ratio
            ),
            "is_active": (
                feature_strength > 0
            ).float(),
            "sampled_active": (
                selected_from_dominant
            ),
            "selected_from_dominant_group": (
                selected_from_dominant
            ),
            "dominant_group_ids": (
                dominant_group
            ),
            "all_node_influence": (
                flat_influence
            ),
            "layer_node_influence": (
                layer_influence
            ),
            "layer_widths": torch.tensor(
                self.scm.widths,
                device=self.device,
                dtype=torch.long,
            ),
            "connection_probs": torch.tensor(
                self.scm.connection_probs,
                device=self.device,
                dtype=torch.float32,
            ),
            "adjacency_matrices": [
                connection.adj
                for connection
                in self.scm.connections
            ],
            "edge_weight_matrices": [
                connection.weights
                for connection
                in self.scm.connections
            ],
            "root_prior_types": list(
                self.scm.root_prior_types
            ),
            "root_prior_type_ids": (
                torch.tensor(
                    self.scm.root_prior_type_ids,
                    device=self.device,
                    dtype=torch.long,
                )
            ),
            "root_mixture_components": (
                torch.tensor(
                    self.scm.root_mixture_components,
                    device=self.device,
                    dtype=torch.long,
                )
            ),
            "missing_mask_train": (
                missing_mask[
                    train_indices
                ]
            ),
            "missing_mask_test": (
                missing_mask[
                    test_indices
                ]
            ),
        }

        self.feature_type = (
            feature_type
        )

        self.cardinality = (
            cardinality
        )

        self.feature_observation_heads = (
            observation_heads
        )

        self.target_observation_head = (
            target_head
        )

        self.n_features = (
            self.d
        )

        return (
            X_observed[
                train_indices
            ],
            y[
                train_indices
            ],
            X_observed[
                test_indices
            ],
            y[
                test_indices
            ],
            info,
        )

    def visualize(
        self,
    ):
        return None

    def forward(
        self,
        X,
    ):
        del X
        return None


# ============================================================================
# Compatible names
# ============================================================================


WeightedMixedLatentSCMTask = (
    WeightedMixedScalarSCMTask
)

MixedLatentSCMTask = (
    WeightedMixedScalarSCMTask
)

MixedSCMTask = (
    WeightedMixedScalarSCMTask
)

WeightedLayeredLatentSCM = (
    WeightedLayeredScalarSCM
)

RandomLayeredLatentSCM = (
    WeightedLayeredScalarSCM
)

RandomLayeredSCM = (
    WeightedLayeredScalarSCM
)

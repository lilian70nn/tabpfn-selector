# replace the dominant sampling method by ancestor/descendant penalty.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.data.helper import make_gen, stratified_classification_split
from src.data.synthetic_task import GenerateTask
from src.data.helper import detach_tree
import time



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


def _randint(low, high, shape, generator, device):
    return torch.randint(
        low,
        high,
        shape,
        generator=generator,
        device=device,
    )


def _standardize(x, dim=0, eps = 1e-6):
    mean = x.mean(dim=dim, keepdim=True).detach()
    std = x.std(dim=dim, unbiased=False, keepdim=True,).clamp_min(eps).detach()
    return (x - mean) / std


def _normalize_probs(values, device, expected_len=None, name="probabilities"):
    probs = torch.as_tensor(values, device=device, dtype=torch.float32)
    if expected_len is not None and probs.numel() != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values.")

    if (probs.numel() == 0 or bool((probs < 0).any()) or probs.sum() <= 0):
        raise ValueError(f"Invalid {name}.")
    return probs / probs.sum()


def _sample_latent(n, prior_probs, g_dag, g_x, device):

    SOURCE_PRIORS = ("gaussian", "uniform", "heavy_tailed", "skewed")
    prior_id = int(torch.multinomial(prior_probs, 1, generator=g_dag).item())
    name = SOURCE_PRIORS[prior_id]

    if name == "gaussian":
        z = _randn(n, 1, generator=g_x, device=device)

    elif name == "uniform":
        bound = 3.0**0.5
        z = 2.0 * bound * _rand(n, 1, generator=g_x, device=device) - bound

    elif name == "heavy_tailed":
        df = 4.0
        numerator = _randn(n, 1, generator=g_x, device=device)
        concentration = torch.full((n, 1), df / 2.0, device=device, dtype=numerator.dtype)
        chi2 = 2.0 * torch._standard_gamma(concentration, generator=g_x)
        z = numerator / torch.sqrt(chi2 / df).clamp_min(1e-4)

    elif name == "skewed":
        normal = _randn(n, 1, generator=g_x, device=device)
        strength = 0.4 + 0.6 * _rand((), generator=g_dag, device=device)
        z = torch.exp(normal * strength)

    z = _standardize(z.float(), dim=0)
    z.requires_grad_(True)
    return z

class ScalarLatentEdge:
    """
    Every node is scalar-valued.
    Input shape: [N, 1]
    Output shape: [N, 1]
    """

    LINEAR = 0
    MLP = 1
    SOFT_TREE = 2

    ACTIVATIONS = (
        "identity", "tanh", "relu", "sigmoid",
        "sin", "square", "softplus",
    )

    def __init__(
        self, generator, device,
        linear_activation_prob = 0.60,
        small_mlp_prob = 0.25,
        soft_tree_prob = 0.15,
        small_mlp_hidden_dim = None,
        soft_tree_depth = 2,
        soft_tree_temperature = 0.5,
    ):
        self.device = device
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_temperature = float(soft_tree_temperature)
        probs = _normalize_probs(
            (linear_activation_prob, small_mlp_prob, soft_tree_prob),
            device,
            expected_len=3,
            name="edge-family probabilities",
        )
        self.edge_type = int(torch.multinomial(probs, 1, generator=generator).item())
        
        # Scalar -> scalar linear + activation.
        self.linear_w = _randn((), generator=generator, device=device)
        self.linear_b = _randn((), generator=generator, device=device)
        self.activation_name = self.ACTIVATIONS[
            int(_randint(0, len(self.ACTIVATIONS), (), generator, device).item())
        ]

        # Scalar -> hidden -> scalar MLP.
        hidden = (int(small_mlp_hidden_dim) if small_mlp_hidden_dim is not None else 8)
        self.mlp_W1 = _randn(hidden, 1, generator=generator, device=device)
        self.mlp_b1 = _randn(hidden, generator=generator, device=device)
        self.mlp_W2 = hidden**-0.5 * _randn(1, hidden, generator=generator, device=device)
        self.mlp_b2 = _randn(1, generator=generator, device=device)

        # Soft tree on scalar input.
        n_internal = 2**self.soft_tree_depth - 1
        n_leaves = 2**self.soft_tree_depth
        self.tree_gate_W = _randn(n_internal, 1, generator=generator, device=device)
        self.tree_gate_b = _randn(n_internal, generator=generator, device=device)
        self.tree_leaf_values = _randn(n_leaves, 1, generator=generator, device=device)

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

        raise RuntimeError(
            f"Unknown activation: {self.activation_name}"
        )

    def _soft_tree(self, x):
        logits = (x @ self.tree_gate_W.T + self.tree_gate_b) / self.soft_tree_temperature
        right = torch.sigmoid(logits)
        left = 1.0 - right
        paths = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        offset = 0

        for depth in range(self.soft_tree_depth):
            width = 2**depth
            left_prob = left[:, offset : offset + width]
            right_prob = right[:, offset : offset + width]
            paths = torch.stack((paths * left_prob,paths * right_prob,), dim=-1).reshape(x.shape[0], -1,)
            offset += width
        return paths @ self.tree_leaf_values

    def __call__(self, parent_latent):
        if (parent_latent.ndim != 2 or parent_latent.shape[1] != 1):
            raise ValueError(f"ScalarLatentEdge expects shape [N, 1], received {tuple(parent_latent.shape)}.")
        x = parent_latent.float()
        if self.edge_type == self.LINEAR:
            value = self.linear_w * x + self.linear_b
            return self._activation(value)
        if self.edge_type == self.MLP:
            hidden = torch.tanh(x @ self.mlp_W1.T + self.mlp_b1)
            return hidden @ self.mlp_W2.T + self.mlp_b2
        return self._soft_tree(x)


# ============================================================================
# Weighted scalar connection
# ============================================================================


class WeightedScalarLayerConnection:
    """
    For each child node:

    1. sample connected parents;
    2. assign positive normalized parent weights;
    3. apply one scalar edge mechanism per parent;
    4. aggregate weighted scalar outputs.
    """

    def __init__(
        self,
        in_width,
        out_width,
        connection_prob,
        edge_weight_concentration,
        g_dag,
        g_x,
        device,
        source_prior_probs=(0.45, 0.20, 0.15, 0.05),
        edgewise_prob = 0.50,
        post_aggregate_prob = 0.25,
        joint_mlp_prob = 0.25,
        joint_mlp_hidden_dim = 8,
        **edge_kwargs,
    ):
        self.in_width = int(in_width)
        self.out_width = int(out_width)
        self.g_dag = g_dag
        self.g_x = g_x
        self.device = device
        self.source_prior_probs = _normalize_probs(source_prior_probs, device, expected_len=4, name="source_prior_probs")

        method_probs = _normalize_probs(
            (edgewise_prob, post_aggregate_prob,joint_mlp_prob),
            device=device,
            expected_len=3,
            name="child-method probabilities",
        )

        self.child_methods = torch.empty(self.out_width, device=device, dtype=torch.long)
        self.adj = _rand(self.in_width, self.out_width, generator=self.g_dag, device=device) < connection_prob


        self.weights = torch.zeros(self.in_width, self.out_width, device=device, dtype=torch.float32)
        self.edges = [
            [None for _ in range(self.out_width)]
            for _ in range(self.in_width)
        ]
        self.child_scalar_edges = [None for _ in range(self.out_width)]
        self.child_joint_mlps = [None for _ in range(self.out_width)]

        for child in range(self.out_width):
            parents = torch.where(self.adj[:, child])[0]
            if parents.numel() == 0:
                self.child_methods[child] = -1
                continue
            concentration = torch.full(
                (parents.numel(),),
                float(edge_weight_concentration),
                device=device,
                dtype=torch.float32,
            )
            raw_weights = torch._standard_gamma(concentration, generator=self.g_dag).clamp_min(1e-8)
            normalized_weights = (raw_weights / raw_weights.sum())
            self.weights[parents, child] = normalized_weights
            method = int(torch.multinomial(method_probs, 1, generator=self.g_dag).item())
            self.child_methods[child] = method

            if method == 0:
                for parent in parents.tolist():
                    self.edges[parent][child] = ScalarLatentEdge(generator=self.g_dag, device=device, **edge_kwargs)
            elif method == 1:
                self.child_scalar_edges[child] = ScalarLatentEdge(generator=self.g_dag, device=device, **edge_kwargs)
            else:
                num_parents = int(parents.numel())
                hidden = int(joint_mlp_hidden_dim)
                self.child_joint_mlps[child] = {
                    "W1": num_parents**-0.5 * _randn(hidden, num_parents, generator=self.g_dag, device=device),
                    "b1": _randn(hidden, generator=self.g_dag, device=device),
                    "W2": hidden**-0.5 * _randn(1, hidden, generator=self.g_dag, device=device),
                    "b2": _randn(1, generator=self.g_dag, device=device),
                }

    def _random_mlp_activation(self, x, generator, device):
        probs = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device, dtype=torch.float32)
        activation_id = int(torch.multinomial(probs, 1, generator=generator).item())

        if activation_id == 0:
            return torch.tanh(x)
        if activation_id == 1:
            return torch.relu(x)
        if activation_id == 2:
            return F.softplus(x)
        return torch.sin(x)


    def __call__(self, parent_latents, generator, latent_noise_scale = 0.0):
        children = []

        for child in range(self.out_width):
            parents = torch.where(self.adj[:, child])[0]

            if parents.numel() == 0:
                value = _sample_latent(parent_latents[0].shape[0], self.source_prior_probs, self.g_dag, self.g_x, self.device)
            else:
                method = int(self.child_methods[child].item())
                if method == 0:
                    value = None
                    for parent in parents.tolist():
                        edge = self.edges[parent][child]
                        if edge is None:
                            raise RuntimeError(
                                f"Missing edge function for "
                                f"parent={parent}, child={child}."
                            )
                        contribution = (self.weights[parent, child] * edge(parent_latents[parent]))
                        value = (contribution if value is None else value + contribution)

                elif method == 1:
                    aggregate = None
                    for parent in parents.tolist():
                        contribution = (self.weights[parent, child] * parent_latents[parent])
                        aggregate = ( contribution if aggregate is None else aggregate + contribution)
                    child_function = self.child_scalar_edges[child]
                    if child_function is None:
                        raise RuntimeError(
                            f"Missing child scalar function "
                            f"for child={child}."
                        )
                    value = child_function(aggregate)

                else:
                    parameters = self.child_joint_mlps[child]
                    if parameters is None:
                        raise RuntimeError(
                            f"Missing joint MLP parameters "
                            f"for child={child}."
                        )
                    weighted_inputs = [self.weights[parent, child] * parent_latents[parent] for parent in parents.tolist()]
                    parent_matrix = torch.cat(weighted_inputs, dim=1)
                    hidden = self._random_mlp_activation(
                        parent_matrix @ parameters["W1"].T + parameters["b1"],
                        generator=generator,
                        device=self.device
                    )
                    value = (hidden @ parameters["W2"].T + parameters["b2"])

            if value is None:
                raise RuntimeError(
                    f"Child {child} produced no value."
                )
            
            value = _standardize(value,  dim=0)
            if latent_noise_scale > 0:
                noise = torch.randn(value.shape, generator=generator, device=self.device, dtype=value.dtype)
                value = (value + float(latent_noise_scale) * noise)
                value = _standardize(value, dim=0)
            children.append(value)
        return children



# ============================================================================
# Full scalar SCM
# ============================================================================


class WeightedLayeredScalarSCM:
    """
    Every SCM node stores one continuous scalar per sample.
    Each node tensor has shape:[N, 1]
    """


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
        connection_probs=(0.30, 0.45, 0.65, 0.85),
        edge_weight_concentration=0.60,
        latent_noise_scale=0.03,
        source_prior_probs=(0.45, 0.20, 0.15, 0.05),
        device=None,
        **edge_kwargs,
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.g_dag = g_dag
        self.g_x = g_x
        self.g_aleatoric = g_aleatoric
        self.num_roots = int(num_roots)
        self.num_layers = int(num_layers)
        self.hidden_width_min = int(hidden_width_min)
        self.hidden_width_max = int(hidden_width_max)
        self.final_width = int(final_width)
        self.connection_probs = tuple(float(p) for p in connection_probs)
        self.latent_noise_scale = float(latent_noise_scale)

        if len(self.connection_probs) != self.num_layers - 1:
            raise ValueError("connection_probs must contain num_layers - 1 values.")

        self.source_prior_probs = _normalize_probs(source_prior_probs, self.device, expected_len=4, name="source_prior_probs")
        self.widths = [self.num_roots]
        for _ in range(self.num_layers - 2):
            width = int(_randint(self.hidden_width_min, self.hidden_width_max + 1, (), self.g_dag, self.device).item())
            self.widths.append(width)

        self.widths.append(self.final_width)
        self.connections = []
        for layer in range(self.num_layers - 1):
            connection = WeightedScalarLayerConnection(
                in_width=self.widths[layer],
                out_width=self.widths[layer + 1],
                connection_prob=self.connection_probs[layer],
                edge_weight_concentration=edge_weight_concentration,
                g_dag=self.g_dag,
                g_x=self.g_x,
                device=self.device,
                source_prior_probs=self.source_prior_probs,
                **edge_kwargs
            )
            self.connections.append(connection)


    def forward(self, n_samples, latent_noise_scale=None):
        current = [_sample_latent(n_samples, self.source_prior_probs, self.g_dag, self.g_x, self.device) for _ in range(self.num_roots)]
        all_latents = [current]
        noise_scale = (
            self.latent_noise_scale
            if latent_noise_scale is None
            else float(latent_noise_scale)
        )
        for connection in self.connections:
            current = connection(current, generator=self.g_aleatoric,latent_noise_scale=noise_scale)
            all_latents.append(current)
        return all_latents
    
    def compute_sampling_influence(self, target_node_idx = 0, decay=1.0):
        """
        Structural influence based only on normalized edge weights.
        influence(parent) = sum_child weight(parent, child) * influence(child)
        This sums products of weights over every path to the target.
        """

        if not (0 <= target_node_idx < self.widths[-1]):
            raise ValueError("Invalid target_node_idx.")
        
        influence = [
            torch.zeros(width, device=self.device, dtype=torch.float32)
            for width in self.widths
        ]
        influence[-1][target_node_idx] = 1.0
        for layer in range(self.num_layers - 2, -1, -1):
            influence[layer] = decay * self.connections[layer].weights @ influence[layer + 1]
        return influence
        

    # def compute_node_influence(self, all_latents, layer_idx, node_idx, target_node_idx = 0):
    #     node = all_latents[layer_idx][node_idx]
    #     target = all_latents[-1][target_node_idx]

    #     grad = torch.autograd.grad(
    #         outputs=target,
    #         inputs=node,
    #         grad_outputs=torch.ones_like(target),
    #         retain_graph=True,
    #         allow_unused=True,
    #     )[0]

    #     if grad is None:
    #         return torch.tensor(0.0, device=self.device, dtype=torch.float32)
    #     return grad.abs().mean().float()

    def compute_node_influence(self, all_latents, node_indices, target_node_idx=0):
        """
        Compute functional influence for multiple nodes in one autograd call.
        node_indices: iterable of (layer_idx, node_idx)
        returns: Tensor of shape [num_nodes]
        """

        target = all_latents[-1][target_node_idx]
        nodes = [all_latents[layer_idx][node_idx] for layer_idx, node_idx in node_indices]
        grads = torch.autograd.grad(
            outputs=target,
            inputs=nodes,
            grad_outputs=torch.ones_like(target),
            retain_graph=False,
            allow_unused=True,
        )
        strengths = [
            torch.zeros((), device=self.device, dtype=torch.float32)
            if grad is None else grad.abs().mean().float() for grad in grads
        ]
        return torch.stack(strengths)


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

    NAMES = ("continuous_scalar", "prototype_discretization", "threshold_binning")

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
        prototype_max_attempts=8,
        prototype_min_separation=1.0,
        binning_jitter=0.20,
    ):
        self.device = device
        self.min_samples_per_category = int(min_samples_per_category)
        self.min_component_weight = float(min_component_weight)
        self.observation_noise_scale = float(observation_noise_scale)

        self.prototype_max_attempts = int(prototype_max_attempts)
        self.prototype_min_separation = float(prototype_min_separation)
        self.binning_jitter = float(binning_jitter)

        self.observation_type_probs = (
            _normalize_probs(
                observation_type_probs,
                device,
                expected_len=3,
                name="observation_type_probs",
            )
        )
        self.cardinalities = tuple(int(k) for k in categorical_cardinalities)
        self.cardinality_probs = (
            _normalize_probs(
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

        score = _standardize(score, dim=0)
        return FeatureObservation(
            values=score,
            is_categorical=False,
            cardinality=0,
            observation_type_id=self.CONTINUOUS,
            observation_type_name=name,
            quality_score=0.0,
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

    def _prototype(self, z, generator):
        scalar = z[:, 0].clone()
        k = self._sample_cardinality(scalar.shape[0], generator)

        if k == 0:
            return self._continuous(z, generator, name="continuous_fallback_from_prototype")

        prototypes = self._select_prototypes(scalar, k, generator)
        distances = torch.abs(scalar[:, None] - prototypes[None, :])
        labels = distances.argmin(dim=1).long()
        counts = torch.bincount(labels, minlength=k)
        smallest_fraction = float((counts.float() / counts.sum().clamp_min(1)).min().item())

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=k,
            observation_type_id=self.PROTOTYPE,
            observation_type_name="prototype_discretization",
            quality_score=smallest_fraction,
            prototypes=prototypes[:, None],
            thresholds=torch.empty(0, device=z.device, dtype=z.dtype),
        )

    def _binning(self, z, generator):
        scalar = _standardize(z[:, 0].clone(), dim=0)
        k = self._sample_cardinality(scalar.shape[0], generator)

        if k == 0:
            return self._continuous(z, generator, name="continuous_fallback_from_binning")

        n = scalar.numel()
        minimum = max(
            self.min_samples_per_category,
            int(torch.ceil(torch.tensor(self.min_component_weight * n, device=z.device)).item()),
        )
        remaining = n - k * minimum
        if remaining < 0:
            return self._continuous(z, generator, name="continuous_fallback_from_binning")

        raw = _rand(k, generator=generator, device=z.device)
        extras_float = raw / raw.sum().clamp_min(1e-12) * remaining
        extras = torch.floor(extras_float).long()
        leftover = remaining - int(extras.sum().item())
        if leftover > 0:
            residual_order = torch.argsort(extras_float - extras.float(), descending=True)
            extras[residual_order[:leftover]] += 1
        counts = (extras + minimum).tolist()
        sorted_values = torch.sort(scalar).values
        thresholds = []
        cumulative = 0

        for count in counts[:-1]:
            cumulative += int(count)
            left_value = sorted_values[cumulative - 1]
            right_value = sorted_values[cumulative]
            threshold = 0.5 * (left_value + right_value)
            thresholds.append(threshold)

        thresholds = torch.stack(thresholds)
        labels = torch.bucketize(scalar,thresholds).long()
        observed_counts = torch.bincount(labels,minlength=k)
        smallest_fraction = float((observed_counts.float()  / observed_counts.sum().clamp_min(1)).min().item())

        return FeatureObservation(
            values=labels,
            is_categorical=True,
            cardinality=k,
            observation_type_id=self.BINNING,
            observation_type_name="threshold_binning",
            quality_score=smallest_fraction,
            prototypes=torch.empty(0, 1, device=z.device, dtype=z.dtype),
            thresholds=thresholds,
        )

    def observe(self, latent, generator):
        z = _standardize(latent.float(),dim=0)

        if self.sampled_type == self.CONTINUOUS:
            return self._continuous(z, generator)
        if self.sampled_type == self.PROTOTYPE:
            return self._prototype(z, generator)
        return self._binning(z, generator)


# ============================================================================
# Target observation
# ============================================================================


class ScalarTargetObservationHead:
    """
    Target is read directly from the scalar target node.
    No projection is used.
    """

    def __init__(
        self,
        device,
        observation_noise_scale=0.03,
    ):
        self.device = device
        self.observation_noise_scale = float(observation_noise_scale)

    def score(self, latent, generator):
        if latent.ndim != 2 or latent.shape[1] != 1:
            raise ValueError(f"Scalar target expects latent shape [N, 1], received {tuple(latent.shape)}.")
        value = latent[:, 0].float()
        if self.observation_noise_scale > 0:
            noise = torch.randn(value.shape, generator=generator, device=self.device, dtype=value.dtype)
            value = (value + self.observation_noise_scale * noise)
        return _standardize(value, dim=0)
    

    @staticmethod
    def balanced_classes(score, num_classes, generator):
        """
        Convert continuous target score into classes using
        mildly randomized quantile thresholds.
        Unlike exact balanced splitting, class proportions
        are allowed to vary while avoiding extremely tiny classes.
        """

        num_classes = int(num_classes)
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

        if num_classes > 4:
            raise ValueError("This implementation currently supports 2 to 4 classes.")

        if num_classes == 2:
            q = 0.40 + 0.20 * torch.rand((), generator=generator,  device=score.device)
            quantiles = torch.stack([q])

        elif num_classes == 3:
            q1 = 0.25 + 0.15 * torch.rand((), generator=generator, device=score.device)
            q2 = 0.60 + 0.15 * torch.rand((), generator=generator, device=score.device)
            quantiles = torch.stack((q1,q2))

        else:
            q1 = 0.15 + 0.15 * torch.rand((), generator=generator, device=score.device)
            q2 = 0.40 + 0.20 * torch.rand((), generator=generator, device=score.device)
            q3 = 0.70 + 0.15 * torch.rand((), generator=generator, device=score.device)
            quantiles = torch.stack((q1, q2, q3,))
        thresholds = torch.quantile(score, quantiles)
        labels = torch.bucketize(score, thresholds)
        return labels.long()


# ============================================================================
# Task
# ============================================================================


class WeightedMixedScalarSCMTask(GenerateTask):
    """
    Mixed tabular SCM with one-dimensional continuous latent nodes.
    Underlying SCM: every node is continuous and scalar.
    Observed feature: continuous scalar, prototype category, or binned category.
    Target: direct scalar readout from final target node.
    """

    use_inference_mode = False

    CONTINUOUS = 0
    CATEGORICAL = 1
    def _sync(self):

        if self.device.type == "cuda":

            torch.cuda.synchronize()

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
        connection_probs=(0.30, 0.45, 0.65, 0.85),
        edge_weight_concentration=0.60,
        latent_noise_scale=0.03,
        sampling_penalty=0.25,
        observation_noise_scale=0.03,
        observation_type_probs=(0.60, 0.20, 0.20),
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        min_samples_per_category=8,
        min_component_weight=0.05,
        prototype_max_attempts=8,
        prototype_min_separation=1.0,
        binning_jitter=0.20,
        source_prior_probs=(0.45, 0.20, 0.15, 0.05),
        linear_activation_prob=0.60,
        small_mlp_prob=0.25,
        soft_tree_prob=0.15,
        small_mlp_hidden_dim=None,
        soft_tree_depth=2,
        soft_tree_temperature=0.5,
        latent_dim=1,
    ):
        if int(latent_dim) != 1:
            raise ValueError("WeightedMixedScalarSCMTask requires latent_dim=1.")
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.num_classes = None if num_classes is None else int(num_classes)
        if self.num_classes is not None and self.num_classes < 2:
            raise ValueError("num_classes must be None or at least 2.")

        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.d_min = int(d_min)
        self.d_max = int(d_max)
        self.test_frac = float(test_frac)
        self.p_missing = float(p_missing)
        self.latent_dim = 1
        self.latent_noise_scale = float(latent_noise_scale)
        self.observation_noise_scale = float(observation_noise_scale)
        self.sampling_penalty = float(sampling_penalty)


        self.scm_kwargs = dict(
            num_roots=num_roots,
            num_layers=num_layers,
            hidden_width_min=hidden_width_min,
            hidden_width_max=hidden_width_max,
            final_width=final_width,
            connection_probs=connection_probs,
            edge_weight_concentration=edge_weight_concentration,
            latent_noise_scale=latent_noise_scale,
            source_prior_probs=source_prior_probs,
            linear_activation_prob=linear_activation_prob,
            small_mlp_prob=small_mlp_prob,
            soft_tree_prob=soft_tree_prob,
            small_mlp_hidden_dim=small_mlp_hidden_dim,
            soft_tree_depth=soft_tree_depth,
            soft_tree_temperature=soft_tree_temperature,
            device=self.device,
        )

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

        self.g_dag, self.dag_seed = make_gen(self.device,dag_seed)
        self.g_aleatoric, self.aleatoric_seed = make_gen(self.device, aleatoric_seed)
        self.g_x, self.x_seed = make_gen(self.device, x_seed)
        self.n = int(_randint(self.n_min, self.n_max + 1, (), self.g_dag, self.device).item())
        self.d = int(_randint(self.d_min, self.d_max + 1, (), self.g_dag, self.device,).item())
        
        super().__init__()

    @staticmethod
    def _flatten(all_latents):
        values = []
        index = []

        for layer_idx, layer in enumerate(all_latents):
            for node_idx, value in enumerate(layer):
                values.append(value)
                index.append((layer_idx, node_idx))
        return values, index


    def _apply_sampling_penalty(self, scores, selected_global_id, flat_index, penalty):
        updated = scores.clone()
        selected_layer, selected_node = flat_index[selected_global_id]
        index_lookup = {key: global_id for global_id, key in enumerate(flat_index)}

        ancestor_distance = {}
        frontier = {(selected_layer, selected_node): 0}
        while frontier:
            next_frontier = {}
            for (layer_idx, node_idx), distance in frontier.items():
                if layer_idx == 0:
                    continue
                connection = self.scm.connections[layer_idx - 1]
                parents = torch.where(connection.adj[:, node_idx])[0].tolist()
                for parent_idx in parents:
                    key = (layer_idx - 1, parent_idx)
                    new_distance = distance + 1
                    if key not in ancestor_distance or new_distance < ancestor_distance[key]:
                        ancestor_distance[key] = new_distance
                        next_frontier[key] = new_distance
            frontier = next_frontier

        descendant_distance = {}
        frontier = {(selected_layer, selected_node): 0}
        while frontier:
            next_frontier = {}
            for (layer_idx, node_idx), distance in frontier.items():
                if layer_idx >= len(self.scm.widths) - 2:
                    continue
                connection = self.scm.connections[layer_idx]
                children = torch.where(connection.adj[node_idx])[0].tolist()
                for child_idx in children:
                    key = (layer_idx + 1, child_idx)
                    new_distance = distance + 1
                    if key not in descendant_distance or new_distance < descendant_distance[key]:
                        descendant_distance[key] = new_distance
                        next_frontier[key] = new_distance
            frontier = next_frontier

        related_distance = {}
        for key, distance in ancestor_distance.items():
            related_distance[key] = distance
        for key, distance in descendant_distance.items():
            related_distance[key] = distance

        for key, distance in related_distance.items():
            global_id = index_lookup[key]
            factor = penalty ** (1.0 / float(distance))
            updated[global_id] *= factor

        updated[selected_global_id] = 0.0
        updated /= updated.sum().clamp_min(1e-12)
        return updated

    def _sample_feature_ids(self, flat_index, flat_influence, penalty):
        candidates = torch.tensor([global_id for global_id, (layer_idx, _) in enumerate(flat_index) if layer_idx < len(self.scm.widths) - 1], device=self.device, dtype=torch.long)
        d = min(self.d, int(candidates.numel()))
        sampling_scores = flat_influence.clone()
        candidate_mask = torch.zeros_like(sampling_scores, dtype=torch.bool)
        candidate_mask[candidates] = True
        sampling_scores[~candidate_mask] = 0.0
        selected = []
        for _ in range(d):
            available_scores = sampling_scores[candidates]
            if available_scores.sum() <= 0:
                remaining = torch.tensor([node_id for node_id in candidates.tolist() if node_id not in selected], device=self.device, dtype=torch.long)
                chosen = remaining[torch.randint(remaining.numel(), (1,), generator=self.g_dag, device=self.device)].item()
            else:
                probs = available_scores / available_scores.sum()
                position = torch.multinomial(probs, 1, generator=self.g_dag).item()
                chosen = int(candidates[position].item())
            selected.append(chosen)
            sampling_scores = self._apply_sampling_penalty(sampling_scores, chosen, flat_index, penalty=penalty)
        permutation = torch.randperm(len(selected), generator=self.g_dag, device=self.device)
        selected_tensor = torch.tensor(selected, device=self.device, dtype=torch.long)[permutation]
        return selected_tensor.tolist()


    def _observe_features(self, flat_latents, feature_ids):
        n = flat_latents[0].shape[0]
        d = len(feature_ids)
        X = torch.empty(n, d, device=self.device, dtype=torch.float32)
        feature_type = torch.empty(d, device=self.device, dtype=torch.long)
        cardinality = torch.zeros(d, device=self.device, dtype=torch.long)
        type_ids = torch.empty(d, device=self.device, dtype=torch.long)
        quality = torch.zeros(d, device=self.device, dtype=torch.float32)

        type_names = []
        prototypes = []
        thresholds = []
        heads = []
        for column, node_id in enumerate(feature_ids):
            head = ScalarObservationHead(
                generator=self.g_dag,
                device=self.device,
                **self.observation_kwargs,
            )
            observed = head.observe(flat_latents[node_id], self.g_aleatoric)
            X[:, column] = observed.values.float()
            feature_type[column] = self.CATEGORICAL if observed.is_categorical else self.CONTINUOUS
            cardinality[column] = observed.cardinality
            type_ids[column] = observed.observation_type_id
            quality[column] = observed.quality_score
            type_names.append(observed.observation_type_name)
            prototypes.append(observed.prototypes)
            thresholds.append(observed.thresholds)
            heads.append(head)

        return (X, feature_type, cardinality, type_ids, type_names, 
                quality, prototypes, thresholds, heads)


    def _generate(self):

        self._sync()
        t0 = time.perf_counter()
        
        self.scm = WeightedLayeredScalarSCM(
            self.g_dag,
            self.g_x,
            self.g_aleatoric,
            **self.scm_kwargs,
        )

        self._sync()
        t1 = time.perf_counter()

        all_latents = self.scm.forward(self.n, latent_noise_scale=self.latent_noise_scale)

        self._sync()
        t2 = time.perf_counter()


        flat_latents, flat_index = self._flatten(all_latents)
        layer_influence = self.scm.compute_sampling_influence(target_node_idx=0)
        flat_influence = torch.cat(layer_influence)
        flat_influence = flat_influence / flat_influence.sum().clamp_min(1e-12)

        # flat_influence_list = []
        # for global_id, (layer_idx, node_idx) in enumerate(flat_index):
        #     if layer_idx == len(self.scm.widths) - 1:
        #         strength = torch.tensor(0.0, device=self.device)
        #     else:
        #         strength = self.scm.compute_node_influence(
        #             all_latents=all_latents,
        #             layer_idx=layer_idx,
        #             node_idx=node_idx,
        #             target_node_idx=0,
        #         )
        #     flat_influence_list.append(strength)

        # flat_influence = torch.stack(flat_influence_list)
        # flat_influence = flat_influence / flat_influence.sum().clamp_min(1e-12)

        feature_ids = self._sample_feature_ids(flat_index, flat_influence, penalty=self.sampling_penalty)
        self.d = len(feature_ids)

        self._sync()
        t3 = time.perf_counter()

        # feature_strength_list = []
        # for global_id in feature_ids:
        #     layer_idx, node_idx = flat_index[global_id]
        #     strength = self.scm.compute_node_influence(
        #         all_latents=all_latents,
        #         layer_idx=layer_idx,
        #         node_idx=node_idx,
        #         target_node_idx=0,
        #     )
        #     feature_strength_list.append(strength)
        # feature_strength = torch.stack(feature_strength_list)
        selected_node_indices = [flat_index[global_id] for global_id in feature_ids]
        feature_strength = self.scm.compute_node_influence(all_latents=all_latents, 
                                                           node_indices=selected_node_indices,
                                                           target_node_idx=0
                                                           )

        self._sync()
        t4 = time.perf_counter()

        # feature_ids_tensor = torch.tensor(feature_ids, device=self.device, dtype=torch.long)
        # feature_strength = flat_influence[feature_ids_tensor]
        importance_ratio = feature_strength / feature_strength.sum().clamp_min(1e-12)

        (X_clean,feature_type,cardinality,type_ids,
         type_names,quality,prototypes,thresholds,heads) = self._observe_features(flat_latents, feature_ids)

        self._sync()
        t5 = time.perf_counter()

        print(
            f"init={t1-t0:.4f}s | "
            f"forward={t2-t1:.4f}s | "
            f"sampling={t3-t2:.4f}s | "
            f"grad={t4-t3:.4f}s | "
            f"observe={t5-t4:.4f}s"
        )
                
        target_global_id = sum(self.scm.widths[:-1])
        target_head = ScalarTargetObservationHead(
            device=self.device,
            observation_noise_scale=self.observation_noise_scale
        )
        target_score = target_head.score(flat_latents[target_global_id], self.g_aleatoric)
        if self.num_classes is None:
            y = target_score
            self.n_classes = None
        else:
            y = target_head.balanced_classes(target_score, self.num_classes, self.g_dag)
            self.n_classes = self.num_classes
        feature_ids_tensor = torch.tensor(feature_ids, device=self.device, dtype=torch.long)

        X_observed = X_clean.clone()
        missing_mask = _rand(*X_observed.shape, generator=self.g_x, device=self.device) < self.p_missing
        X_observed[missing_mask] = torch.nan

        if self.num_classes is not None:
            train_idx, test_idx = stratified_classification_split(
                y=y.long(),
                test_frac=self.test_frac,
                generator=self.g_x,
                device=self.device,   
            )
        else:
            n_test = min(max(1, round(self.n * self.test_frac)), self.n - 2)
            order = torch.randperm(self.n, generator=self.g_x, device=self.device)
            train_idx = order[:-n_test]
            test_idx = order[-n_test:]

        info = {
            "feature_type": feature_type,
            "cardinality": cardinality,
            "feature_observation_type_ids": type_ids,
            "feature_observation_type_names": type_names,
            "feature_observation_quality": quality,
            "feature_prototypes": prototypes,
            "feature_thresholds": thresholds,
            "feature_ids": feature_ids_tensor,
            "target_id": torch.tensor(target_global_id, device=self.device, dtype=torch.long),
            "feature_strength": feature_strength,
            "importance_ratio": importance_ratio,
            "is_active": (feature_strength > 0).float(),
            "all_node_influence": flat_influence,
            "layer_node_influence": layer_influence,
            "layer_widths": torch.tensor(self.scm.widths, device=self.device, dtype=torch.long),
            "connection_probs": torch.tensor(self.scm.connection_probs, device=self.device, dtype=torch.float32),
            "adjacency_matrices": [connection.adj for connection in self.scm.connections],
            "edge_weight_matrices": [connection.weights for connection in self.scm.connections],
            "latent_dim": torch.tensor(1, device=self.device, dtype=torch.long),
            "missing_mask_train": missing_mask[train_idx],
            "missing_mask_test": missing_mask[test_idx],
        }

        self.feature_type = feature_type
        self.cardinality = cardinality
        self.feature_observation_heads = heads
        self.target_observation_head = target_head
        self.n_features = self.d

        result = (
            X_observed[train_idx], y[train_idx],
            X_observed[test_idx], y[test_idx], info
        )

        result = detach_tree(result)

        return result

    def visualize(self):
        return None

    def forward(self, X):
        del X
        return None


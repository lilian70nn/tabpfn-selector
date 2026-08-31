# replace the dominant sampling method by ancestor/descendant penalty.
# retention add in the gradient imp for discrete faetures 

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.data.helper import make_gen, stratified_classification_split
from src.data.synthetic_task import GenerateTask
from src.data.helper import detach_tree


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



def _normalize_probs(values, device, expected_len=None, name="probabilities"):
    probs = torch.as_tensor(values, device=device, dtype=torch.float32)
    if expected_len is not None and probs.numel() != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values.")

    if (probs.numel() == 0 or bool((probs < 0).any()) or probs.sum() <= 0):
        raise ValueError(f"Invalid {name}.")
    return probs / probs.sum()


@dataclass(frozen=True)
class NodeSpec:
    kind: str
    cardinality: int = 0

@dataclass(frozen=True)
class SourceSpec:
    prior_id: int = -1
    skew_strength: float = 0.0
    categorical_probs: torch.Tensor = None


def _sample_source_spec(spec, prior_probs, g_dag, device):
    if spec.kind == "cont":
        prior_id = int(torch.multinomial(prior_probs, 1, generator=g_dag).item())
        skew_strength = 0.0
        if prior_id == 3:
            skew_strength = float((0.4 + 0.6 * _rand((), generator=g_dag, device=device)).item())
        return SourceSpec(prior_id=prior_id, skew_strength=skew_strength)

    probs = _rand(spec.cardinality, generator=g_dag, device=device)
    probs = probs / probs.sum().clamp_min(1e-12)
    return SourceSpec(categorical_probs=probs)


def _sample_node_spec(generator, device, p_cat, cardinalities, cardinality_probs):
    if float(_rand((), generator=generator, device=device).item()) >= float(p_cat):
        return NodeSpec("cont", 0)

    probs = _normalize_probs(
        cardinality_probs,
        device,
        expected_len=len(cardinalities),
        name="categorical_cardinality_probs",
    )
    idx = int(torch.multinomial(probs, 1, generator=generator).item())
    return NodeSpec("cat", int(cardinalities[idx]))


def _sample_latent(n, source_spec, g_x, device):
    prior_id = source_spec.prior_id

    if prior_id == 0:
        z = _randn(n, 1, generator=g_x, device=device)

    elif prior_id == 1:
        bound = 3.0 ** 0.5
        z = 2.0 * bound * _rand(n, 1, generator=g_x, device=device) - bound

    elif prior_id == 2:
        df = 4.0
        numerator = _randn(n, 1, generator=g_x, device=device)
        concentration = torch.full((n, 1), df / 2.0, device=device, dtype=numerator.dtype)
        chi2 = 2.0 * torch._standard_gamma(concentration, generator=g_x)
        z = numerator / torch.sqrt(chi2 / df).clamp_min(1e-4)

    elif prior_id == 3:
        normal = _randn(n, 1, generator=g_x, device=device)
        z = torch.exp(normal * source_spec.skew_strength)

    else:
        raise RuntimeError(f"Invalid continuous source prior_id: {prior_id}")

    return z.float()


def _sample_source_node(n, spec, source_spec, g_x, device):
    if spec.kind == "cont":
        return _sample_latent(n, source_spec, g_x, device)

    values = torch.multinomial(
        source_spec.categorical_probs,
        n,
        replacement=True,
        generator=g_x,
    )
    return values.long().reshape(-1, 1)



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
        self.use_residual = bool(_rand((), generator=generator, device=device) < 0.5)
        
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
            nonlinear = self._activation(value)
        elif self.edge_type == self.MLP:
            hidden = torch.tanh(x @ self.mlp_W1.T + self.mlp_b1)
            nonlinear = hidden @ self.mlp_W2.T + self.mlp_b2
        else:
            nonlinear = self._soft_tree(x)
        if self.use_residual:
            return nonlinear + x
        return nonlinear

class CatToContEdge:
    def __init__(self, parent_cardinality, generator, device):
        self.parent_cardinality = int(parent_cardinality)
        self.values = _randn(self.parent_cardinality, 1, generator=generator, device=device)

    def __call__(self, parent_value):
        ids = parent_value.long().reshape(-1)
        return self.values[ids]


class ContToCatEdge:
    def __init__(self, child_cardinality, generator, device, hidden_dim=8):
        self.child_cardinality = int(child_cardinality)
        hidden_dim = int(hidden_dim)

        self.W1 = _randn(hidden_dim, 1, generator=generator, device=device)
        self.b1 = _randn(hidden_dim, generator=generator, device=device)
        self.W2 = hidden_dim ** -0.5 * _randn(self.child_cardinality, hidden_dim, generator=generator, device=device)
        self.b2 = 0.3 * _randn(self.child_cardinality, generator=generator, device=device)

    def __call__(self, parent_value):
        x = parent_value.float()
        hidden = torch.tanh(x @ self.W1.T + self.b1)
        return hidden @ self.W2.T + self.b2


class CatToCatEdge:
    def __init__(self, parent_cardinality, child_cardinality, generator, device):
        self.parent_cardinality = int(parent_cardinality)
        self.child_cardinality = int(child_cardinality)
        self.table = _randn(self.parent_cardinality, self.child_cardinality, generator=generator, device=device)

    def __call__(self, parent_value):
        ids = parent_value.long().reshape(-1)
        return self.table[ids]


def make_typed_edge(parent_spec, child_spec, generator, device, **edge_kwargs):
    if parent_spec.kind == "cont" and child_spec.kind == "cont":
        return ScalarLatentEdge(generator=generator, device=device, **edge_kwargs)

    if parent_spec.kind == "cat" and child_spec.kind == "cont":
        return CatToContEdge(parent_spec.cardinality, generator, device)

    if parent_spec.kind == "cont" and child_spec.kind == "cat":
        return ContToCatEdge(child_spec.cardinality, generator, device)

    if parent_spec.kind == "cat" and child_spec.kind == "cat":
        return CatToCatEdge(parent_spec.cardinality, child_spec.cardinality, generator, device)

    raise RuntimeError(f"Unsupported typed edge: {parent_spec} -> {child_spec}")


def _sample_categorical_logits(logits, generator, temperature=1):
    probs = torch.softmax(logits / float(temperature), dim=1)
    return torch.multinomial(probs, 1, replacement=True, generator=generator).long()

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
        parent_specs,
        child_specs,
        child_source_specs,
        connection_prob,
        edge_weight_concentration,
        g_dag,
        g_x,
        device,
        edgewise_prob = 0.50,
        post_aggregate_prob = 0.25,
        joint_mlp_prob = 0.25,
        joint_mlp_hidden_dim = 8,
        **edge_kwargs,
    ):
        self.in_width = int(in_width)
        self.out_width = int(out_width)
        self.parent_specs = list(parent_specs)
        self.child_specs = list(child_specs)
        self.child_source_specs = list(child_source_specs)
        if len(self.parent_specs) != self.in_width:
            raise ValueError("parent_specs length must equal in_width.")
        if len(self.child_specs) != self.out_width:
            raise ValueError("child_specs length must equal out_width.")
        if len(self.child_source_specs) != self.out_width:
            raise ValueError("child_source_specs length must equal out_width.")
        
        self.g_dag = g_dag
        self.g_x = g_x
        self.device = device

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
        self.parent_scalar_edges = [[None for _ in range(self.out_width)] for _ in range(self.in_width)]

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
            raw_magnitudes = torch._standard_gamma(concentration, generator=self.g_dag).clamp_min(1e-8)
            signs = torch.where(
                _rand(parents.numel(), generator=self.g_dag, device=device) < 0.5,
                -torch.ones(parents.numel(), device=device, dtype=torch.float32),
                torch.ones(parents.numel(), device=device, dtype=torch.float32)
            )
            signed_weights = raw_magnitudes * signs
            normalized_weights = signed_weights / signed_weights.abs().sum().clamp_min(1e-12)
            self.weights[parents, child] = normalized_weights
            method = int(torch.multinomial(method_probs, 1, generator=self.g_dag).item())
            self.child_methods[child] = method

            child_spec = self.child_specs[child]

            if method == 0:
                for parent in parents.tolist():
                    self.edges[parent][child] = make_typed_edge(self.parent_specs[parent], child_spec, self.g_dag, device, **edge_kwargs)

            elif method == 1:
                for parent in parents.tolist():
                    if self.parent_specs[parent].kind == "cat":
                        self.parent_scalar_edges[parent][child] = CatToContEdge(self.parent_specs[parent].cardinality, self.g_dag, device)

                if child_spec.kind == "cont":
                    self.child_scalar_edges[child] = ScalarLatentEdge(generator=self.g_dag, device=device, **edge_kwargs)
                else:
                    self.child_scalar_edges[child] = ContToCatEdge(child_spec.cardinality, self.g_dag, device)

            else:
                for parent in parents.tolist():
                    if self.parent_specs[parent].kind == "cat":
                        self.parent_scalar_edges[parent][child] = CatToContEdge(self.parent_specs[parent].cardinality, self.g_dag, device)

                num_parents = int(parents.numel())
                hidden = int(joint_mlp_hidden_dim)
                output_dim = 1 if child_spec.kind == "cont" else child_spec.cardinality
                activation_probs = torch.tensor([0.35, 0.25, 0.25, 0.15], device=device,  dtype=torch.float32)
                activation_id = int(torch.multinomial(activation_probs, 1, generator=self.g_dag).item())
                self.child_joint_mlps[child] = {
                    "W1": num_parents**-0.5 * _randn(hidden, num_parents, generator=self.g_dag, device=device),
                    "b1": _randn(hidden, generator=self.g_dag, device=device),
                    "W2": hidden**-0.5 * _randn(output_dim, hidden, generator=self.g_dag, device=device),
                    "b2": _randn(output_dim, generator=self.g_dag, device=device),
                    "activation_id": activation_id
                }

    def _random_mlp_activation(self, x, activation_id):

        if activation_id == 0:
            return torch.tanh(x)
        if activation_id == 1:
            return torch.relu(x)
        if activation_id == 2:
            return F.softplus(x)
        return torch.sin(x)

    def _parent_as_scalar(self, parent_value, parent, child):
        if self.parent_specs[parent].kind == "cont":
            return parent_value.float().reshape(-1, 1)

        encoder = self.parent_scalar_edges[parent][child]
        if encoder is None:
            raise RuntimeError(f"Missing categorical parent encoder for parent={parent}, child={child}.")
        return encoder(parent_value)


    def __call__(self, parent_latents, generator, latent_noise_scale = 0.0):
        children = []

        for child in range(self.out_width):
            parents = torch.where(self.adj[:, child])[0]

            if parents.numel() == 0:
                child_spec = self.child_specs[child]
                source_spec = self.child_source_specs[child]
                value = _sample_source_node(parent_latents[0].shape[0], child_spec, source_spec, self.g_x, self.device)
            else:
                method = int(self.child_methods[child].item())
                if method == 0:
                    value = None
                    for parent in parents.tolist():
                        edge = self.edges[parent][child]
                        if edge is None:
                            raise RuntimeError(f"Missing edge function for parent={parent}, child={child}.")
                        contribution = self.weights[parent, child] * edge(parent_latents[parent])
                        value = contribution if value is None else value + contribution

                    if self.child_specs[child].kind == "cat":
                        value = _sample_categorical_logits(value, generator)

                elif method == 1:
                    aggregate = None

                    for parent in parents.tolist():
                        parent_scalar = self._parent_as_scalar(parent_latents[parent], parent, child)
                        contribution = self.weights[parent, child] * parent_scalar
                        aggregate = contribution if aggregate is None else aggregate + contribution

                    child_function = self.child_scalar_edges[child]
                    if child_function is None:
                        raise RuntimeError(f"Missing child function for child={child}.")

                    value = child_function(aggregate)

                    if self.child_specs[child].kind == "cat":
                        value = _sample_categorical_logits(value, generator)

                else:
                    parameters = self.child_joint_mlps[child]
                    if parameters is None:
                        raise RuntimeError(f"Missing joint MLP parameters for child={child}.")

                    weighted_inputs = []
                    for parent in parents.tolist():
                        parent_scalar = self._parent_as_scalar(parent_latents[parent], parent, child)
                        weighted_inputs.append(self.weights[parent, child] * parent_scalar)

                    parent_matrix = torch.cat(weighted_inputs, dim=1)
                    hidden = self._random_mlp_activation(parent_matrix @ parameters["W1"].T + parameters["b1"], parameters["activation_id"])
                    value = hidden @ parameters["W2"].T + parameters["b2"]

                    if self.child_specs[child].kind == "cat":
                        value = _sample_categorical_logits(value, generator)

            if value is None:
                raise RuntimeError(
                    f"Child {child} produced no value."
                )
            
            if latent_noise_scale > 0 and self.child_specs[child].kind == "cont":
                noise = torch.randn(value.shape, generator=generator, device=self.device, dtype=value.dtype)
                value = value + float(latent_noise_scale) * noise
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
        target_num_classes=None,
        num_roots=4,
        num_layers=5,
        hidden_width_min=8,
        hidden_width_max=12,
        final_width=1,
        p_cat=0.30,
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
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
        self.target_num_classes = None if target_num_classes is None else int(target_num_classes)
        self.num_roots = int(num_roots)
        self.num_layers = int(num_layers)
        self.hidden_width_min = int(hidden_width_min)
        self.hidden_width_max = int(hidden_width_max)
        self.final_width = int(final_width)
        self.p_cat = float(p_cat)
        self.categorical_cardinalities = tuple(int(k) for k in categorical_cardinalities)
        self.categorical_cardinality_probs = tuple(float(p) for p in categorical_cardinality_probs)
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

        self.layers = []
        for layer_idx, width in enumerate(self.widths):
            if layer_idx == self.num_layers - 1:
                target_spec = NodeSpec("cont", 0) if self.target_num_classes is None else NodeSpec("cat", self.target_num_classes)
                specs = [target_spec for _ in range(width)]
            else:
                specs = [
                    _sample_node_spec(
                        generator=self.g_dag,
                        device=self.device,
                        p_cat=self.p_cat,
                        cardinalities=self.categorical_cardinalities,
                        cardinality_probs=self.categorical_cardinality_probs,
                    )
                    for _ in range(width)
                ]
            self.layers.append(specs)

        self.source_specs = []
        for layer_specs in self.layers:
            layer_source_specs = [
                _sample_source_spec(
                    spec=spec,
                    prior_probs=self.source_prior_probs,
                    g_dag=self.g_dag,
                    device=self.device,
                )
                for spec in layer_specs
            ]
            self.source_specs.append(layer_source_specs)

        self.connections = []
        for layer in range(self.num_layers - 1):
            connection = WeightedScalarLayerConnection(
                in_width=self.widths[layer],
                out_width=self.widths[layer + 1],
                parent_specs=self.layers[layer],
                child_specs=self.layers[layer + 1],
                child_source_specs=self.source_specs[layer + 1],
                connection_prob=self.connection_probs[layer],
                edge_weight_concentration=edge_weight_concentration,
                g_dag=self.g_dag,
                g_x=self.g_x,
                device=self.device,
                **edge_kwargs
            )
            self.connections.append(connection)


    def forward(self, n_samples, latent_noise_scale=None):
        current = [
            _sample_source_node(n_samples, spec, source_spec, self.g_x, self.device)
            for spec, source_spec in zip(self.layers[0], self.source_specs[0])
        ]
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
            influence[layer] = decay * self.connections[layer].weights.abs() @ influence[layer + 1]
        return influence


    def _forward_intervention(self, all_latents, layer_idx, node_idx, value):

        current = list(all_latents[layer_idx])
        x = current[node_idx]
        x_spec = self.layers[layer_idx][node_idx]

        if x_spec.kind == "cont":
            current[node_idx] = torch.full_like(x, float(value))
        else:
            current[node_idx] = torch.full_like(x, int(value))

        for l in range(layer_idx, self.num_layers - 1):
            current = self.connections[l](
                current,
                generator=self.g_aleatoric,
                latent_noise_scale=self.latent_noise_scale,
            )

            connection = self.connections[l]
            for child_idx in range(connection.out_width):
                if not bool(connection.adj[:, child_idx].any()):
                    current[child_idx] = all_latents[l + 1][child_idx]

        return current


    def compute_intervention_influence(self, all_latents, layer_idx, node_idx, target_node_idx=0):
        x = all_latents[layer_idx][node_idx]
        x_spec = self.layers[layer_idx][node_idx]
        y = all_latents[-1][target_node_idx]
        y_spec = self.layers[-1][target_node_idx]

        base_dag_state = self.g_dag.get_state()
        base_x_state = self.g_x.get_state()
        base_aleatoric_state = self.g_aleatoric.get_state()

        def run_do(value):
            self.g_dag.set_state(base_dag_state)
            self.g_x.set_state(base_x_state)
            self.g_aleatoric.set_state(base_aleatoric_state)
            result = self._forward_intervention(all_latents, layer_idx, node_idx, value)
            return result[target_node_idx]

        if x_spec.kind == "cont":
            x_flat = x.detach().float().reshape(-1)
            quantiles = torch.tensor([0.25, 0.75], device=self.device)
            low, high = torch.quantile(x_flat, quantiles)
            delta_x = (high - low).abs()

            if delta_x <= 1e-8:
                self.g_dag.set_state(base_dag_state)
                self.g_x.set_state(base_x_state)
                self.g_aleatoric.set_state(base_aleatoric_state)
                return torch.zeros((), device=self.device)

            y_low = run_do(low)
            y_high = run_do(high)

            if y_spec.kind == "cont":
                y_std = y.detach().float().std(unbiased=False)
                x_std = x_flat.std(unbiased=False)
                if y_std <= 1e-8 or x_std <= 1e-8:
                    score = torch.zeros((), device=self.device)
                else:
                    delta_y = (y_high.float().mean() - y_low.float().mean()).abs()
                    score = delta_y * x_std / (delta_x * y_std)
            else:
                p_low = torch.bincount(y_low.long().reshape(-1), minlength=y_spec.cardinality).float()
                p_high = torch.bincount(y_high.long().reshape(-1), minlength=y_spec.cardinality).float()
                p_low = p_low / p_low.sum().clamp_min(1.0)
                p_high = p_high / p_high.sum().clamp_min(1.0)
                score = 0.5 * (p_high - p_low).abs().sum()

        else:
            x_flat = x.detach().long().reshape(-1)
            counts = torch.bincount(x_flat, minlength=x_spec.cardinality).float()
            category_weights = counts / counts.sum().clamp_min(1.0)
            if y_spec.kind == "cont":
                means = []
                for category in range(x_spec.cardinality):
                    y_do = run_do(category)
                    means.append(y_do.float().mean())
                means = torch.stack(means)
                weighted_mean = (category_weights * means).sum()
                weighted_var = (category_weights * (means - weighted_mean).square()).sum()
                y_std = y.detach().float().std(unbiased=False)
                if y_std <= 1e-8:
                    score = torch.zeros((), device=self.device)
                else:
                    score = torch.sqrt(weighted_var.clamp_min(0.0)) / y_std

            else:
                distributions = []
                for category in range(x_spec.cardinality):
                    y_do = run_do(category)
                    p = torch.bincount(y_do.long().reshape(-1), minlength=y_spec.cardinality).float()
                    p = p / p.sum().clamp_min(1.0)
                    distributions.append(p)
                distributions = torch.stack(distributions)
                mean_distribution = (category_weights[:, None] * distributions).sum(dim=0)
                tv = 0.5 * (distributions - mean_distribution[None, :]).abs().sum(dim=1)
                score = (category_weights * tv).sum()

        self.g_dag.set_state(base_dag_state)
        self.g_x.set_state(base_x_state)
        self.g_aleatoric.set_state(base_aleatoric_state)

        return score.float()


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

    def __init__(
        self,
        num_classes=None,
        n_min=400,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_cat=0.30,
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
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
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
            target_num_classes=self.num_classes,
            num_roots=num_roots,
            num_layers=num_layers,
            hidden_width_min=hidden_width_min,
            hidden_width_max=hidden_width_max,
            final_width=final_width,
            p_cat=p_cat,
            categorical_cardinalities=categorical_cardinalities,
            categorical_cardinality_probs=categorical_cardinality_probs,
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

    @staticmethod
    def _flatten_specs(layers):
        specs = []
        for layer in layers:
            specs.extend(layer)
        return specs

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


    def _observe_features(self, flat_latents, flat_specs, feature_ids):
        n = flat_latents[0].shape[0]
        d = len(feature_ids)

        X = torch.empty(n, d, device=self.device, dtype=torch.float32)
        feature_type = torch.empty(d, device=self.device, dtype=torch.long)
        cardinality = torch.zeros(d, device=self.device, dtype=torch.long)
        quality = torch.zeros(d, device=self.device, dtype=torch.float32)

        type_names = []

        for column, node_id in enumerate(feature_ids):
            value = flat_latents[node_id]
            spec = flat_specs[node_id]

            if spec.kind == "cont":
                observed = value.float().reshape(-1)

                if self.observation_noise_scale > 0:
                    noise = torch.randn(
                        observed.shape,
                        generator=self.g_aleatoric,
                        device=self.device,
                        dtype=observed.dtype,
                    )
                    observed = observed + self.observation_noise_scale * noise

                X[:, column] = observed
                feature_type[column] = self.CONTINUOUS
                cardinality[column] = 0
                quality[column] = 1.0
                type_names.append("continuous_node")

            else:
                observed = value.long().reshape(-1)
                X[:, column] = observed.float()
                feature_type[column] = self.CATEGORICAL
                cardinality[column] = spec.cardinality

                counts = torch.bincount(observed, minlength=spec.cardinality).float()
                quality[column] = counts.min() / counts.sum().clamp_min(1.0)
                type_names.append("categorical_node")

        return X, feature_type, cardinality, type_names, quality

    def _generate(self):
             
        self.scm = WeightedLayeredScalarSCM(
            self.g_dag,
            self.g_x,
            self.g_aleatoric,
            **self.scm_kwargs,
        )

        all_latents = self.scm.forward(self.n, latent_noise_scale=self.latent_noise_scale)

        flat_latents, flat_index = self._flatten(all_latents)
        flat_specs = self._flatten_specs(self.scm.layers)
        layer_influence = self.scm.compute_sampling_influence(target_node_idx=0)
        flat_influence = torch.cat(layer_influence)
        flat_influence = flat_influence / flat_influence.sum().clamp_min(1e-12)

        feature_ids = self._sample_feature_ids(flat_index, flat_influence, penalty=self.sampling_penalty)
        self.d = len(feature_ids)

        feature_strength = []
        for global_id in feature_ids:
            layer_idx, node_idx = flat_index[global_id]
            strength = self.scm.compute_intervention_influence(
                all_latents=all_latents,
                layer_idx=layer_idx,
                node_idx=node_idx,
                target_node_idx=0,
            )
            feature_strength.append(strength)
        feature_strength = torch.stack(feature_strength)

        X_clean, feature_type, cardinality, type_names, quality = self._observe_features(flat_latents, flat_specs, feature_ids)
        feature_importance = feature_strength / feature_strength.sum().clamp_min(1e-12)

        target_global_id = sum(self.scm.widths[:-1])
        target_value = flat_latents[target_global_id]

        if self.num_classes is None:
            y = target_value.float().reshape(-1)
            if self.observation_noise_scale > 0:
                noise = torch.randn(y.shape, generator=self.g_aleatoric, device=self.device, dtype=y.dtype)
                y = y + self.observation_noise_scale * noise
            self.n_classes = None
        else:
            y = target_value.long().reshape(-1)
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
            "feature_observation_type_names": type_names,
            "feature_observation_quality": quality,
            "feature_ids": feature_ids_tensor,
            "target_id": torch.tensor(target_global_id, device=self.device, dtype=torch.long),
            "feature_importance": feature_importance,
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


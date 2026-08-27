import torch
import torch.nn.functional as F
from .utils import (
    rand,
    randint,
    randn,
    standardize,
    normalize_probs,
)


def sample_latent(n, prior_probs, g_x, device):

    SOURCE_PRIORS = ("gaussian", "uniform", "heavy_tailed", "skewed")
    prior_id = int(torch.multinomial(prior_probs, 1, generator=g_x).item())
    name = SOURCE_PRIORS[prior_id]

    if name == "gaussian":
        z = randn(n, 1, generator=g_x, device=device)

    elif name == "uniform":
        bound = 3.0**0.5
        z = 2.0 * bound * rand(n, 1, generator=g_x, device=device) - bound

    elif name == "heavy_tailed":
        df = 4.0
        numerator = randn(n, 1, generator=g_x, device=device)
        concentration = torch.full((n, 1), df / 2.0, device=device, dtype=numerator.dtype)
        chi2 = 2.0 * torch._standard_gamma(concentration, generator=g_x)
        z = numerator / torch.sqrt(chi2 / df).clamp_min(1e-4)

    elif name == "skewed":
        normal = randn(n, 1, generator=g_x, device=device)
        strength = 0.4 + 0.6 * rand((), generator=g_x, device=device)
        z = torch.exp(normal * strength)

    z = z.float()
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

    ACTIVATIONS = ("identity", "tanh", "relu", "sigmoid", "sin", "square", "softplus")

    def __init__(
        self, generator, device,
        edge_family_probs=(0.50, 0.25, 0.25),
        small_mlp_hidden_dim = None,
        soft_tree_depth = 2,
        soft_tree_temperature = 0.5,
    ):
        self.device = device
        self.soft_tree_depth = int(soft_tree_depth)
        self.soft_tree_temperature = float(soft_tree_temperature)
        probs = normalize_probs(
            edge_family_probs,
            device,
            expected_len=3,
            name="edge-family probabilities",
        )
        self.edge_type = int(torch.multinomial(probs, 1, generator=generator).item())
        self.use_residual = bool(rand((), generator=generator, device=device) < 0.5)
        
        # Scalar -> scalar linear + activation.
        self.linear_w = randn((), generator=generator, device=device)
        self.linear_b = randn((), generator=generator, device=device)
        self.activation_name = self.ACTIVATIONS[
            int(randint(0, len(self.ACTIVATIONS), (), generator, device).item())
        ]

        # Scalar -> hidden -> scalar MLP.
        hidden = (int(small_mlp_hidden_dim) if small_mlp_hidden_dim is not None else 8)
        self.mlp_W1 = randn(hidden, 1, generator=generator, device=device)
        self.mlp_b1 = randn(hidden, generator=generator, device=device)
        self.mlp_W2 = hidden**-0.5 * randn(1, hidden, generator=generator, device=device)
        self.mlp_b2 = randn(1, generator=generator, device=device)

        # Soft tree on scalar input.
        n_internal = 2**self.soft_tree_depth - 1
        n_leaves = 2**self.soft_tree_depth
        self.tree_gate_W = randn(n_internal, 1, generator=generator, device=device)
        self.tree_gate_b = randn(n_internal, generator=generator, device=device)
        self.tree_leaf_values = randn(n_leaves, 1, generator=generator, device=device)

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
            return torch.clamp(x, -6.0, 6.0).square()
        if self.activation_name == "softplus":
            return F.softplus(x)
        raise RuntimeError(f"Unknown activation: {self.activation_name}")

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
        self.source_prior_probs = normalize_probs(source_prior_probs, device, expected_len=4, name="source_prior_probs")

        method_probs = normalize_probs(
            (edgewise_prob, post_aggregate_prob,joint_mlp_prob),
            device=device,
            expected_len=3,
            name="child-method probabilities",
        )

        self.child_methods = torch.empty(self.out_width, device=device, dtype=torch.long)
        self.adj = rand(self.in_width, self.out_width, generator=self.g_dag, device=device) < connection_prob


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
            raw_magnitudes = torch._standard_gamma(concentration, generator=self.g_dag).clamp_min(1e-8)
            signs = torch.where(
                rand(parents.numel(), generator=self.g_dag, device=device) < 0.5,
                -torch.ones(parents.numel(), device=device, dtype=torch.float32),
                torch.ones(parents.numel(), device=device, dtype=torch.float32)
            )
            signed_weights = raw_magnitudes * signs
            normalized_weights = signed_weights / signed_weights.abs().sum().clamp_min(1e-12)
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
                    "W1": num_parents**-0.5 * randn(hidden, num_parents, generator=self.g_dag, device=device),
                    "b1": randn(hidden, generator=self.g_dag, device=device),
                    "W2": hidden**-0.5 * randn(1, hidden, generator=self.g_dag, device=device),
                    "b2": randn(1, generator=self.g_dag, device=device),
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
                value = sample_latent(parent_latents[0].shape[0], self.source_prior_probs, self.g_x, self.device)
            else:
                method = int(self.child_methods[child].item())
                if method == 0:
                    value = None
                    for parent in parents.tolist():
                        edge = self.edges[parent][child]
                        if edge is None:
                            raise RuntimeError(f"Missing edge function for parent={parent}, child={child}.")
                        contribution = (self.weights[parent, child] * edge(parent_latents[parent]))
                        value = (contribution if value is None else value + contribution)

                elif method == 1:
                    aggregate = None
                    for parent in parents.tolist():
                        contribution = (self.weights[parent, child] * parent_latents[parent])
                        aggregate = ( contribution if aggregate is None else aggregate + contribution)
                    child_function = self.child_scalar_edges[child]
                    if child_function is None:
                        raise RuntimeError(f"Missing child scalar function for child={child}.")
                    value = child_function(aggregate)

                else:
                    parameters = self.child_joint_mlps[child]
                    if parameters is None:
                        raise RuntimeError(f"Missing joint MLP parameters for child={child}.")
                    weighted_inputs = [self.weights[parent, child] * parent_latents[parent] for parent in parents.tolist()]
                    parent_matrix = torch.cat(weighted_inputs, dim=1)
                    hidden = self._random_mlp_activation(
                        parent_matrix @ parameters["W1"].T + parameters["b1"],
                        generator=generator,
                        device=self.device
                    )
                    value = (hidden @ parameters["W2"].T + parameters["b2"])

            if value is None:
                raise RuntimeError(f"Child {child} produced no value.")
            
            if latent_noise_scale > 0:
                noise = torch.randn(value.shape, generator=generator, device=self.device, dtype=value.dtype)
                value = (value + float(latent_noise_scale) * noise)
            value = standardize(value, dim=0)
            children.append(value)
        return children


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

        self.source_prior_probs = normalize_probs(source_prior_probs, self.device, expected_len=4, name="source_prior_probs")
        self.widths = [self.num_roots]
        for _ in range(self.num_layers - 2):
            width = int(randint(self.hidden_width_min, self.hidden_width_max + 1, (), self.g_dag, self.device).item())
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
        current = [sample_latent(n_samples, self.source_prior_probs, self.g_x, self.device) for _ in range(self.num_roots)]
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
        

    def compute_node_influence(self, all_latents, node_indices, target_node_idx=0):
        """
        Scale-normalized local functional influence.
        strength_j = E[|d target / d node_j|] * std(node_j) / std(target)
        The scale correction makes influence comparable across nodes,
        including root nodes that are not internally standardized.
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
        target_std = target.detach().float().std(unbiased=False)
        if not torch.isfinite(target_std) or target_std <= 1e-8:
            return torch.zeros(len(nodes), device=self.device, dtype=torch.float32)
        strengths = []
        for node, grad in zip(nodes, grads):
            if grad is None:
                strength = torch.zeros((), device=self.device, dtype=torch.float32)
            else:
                node_std =  node.detach().float().std(unbiased=False)
                if not torch.isfinite(node_std) or node_std <= 1e-8:
                    strength = torch.zeros((), device=self.device, dtype=torch.float32)
                else:
                    grad_mag = grad.detach().abs().mean().float()
                    strength = grad_mag * node_std / target_std
            strengths.append(strength)
        return torch.stack(strengths)
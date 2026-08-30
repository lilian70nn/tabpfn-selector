import torch
import torch.nn.functional as F
from .utils import rand, randint, randn, standardize, normalize_probs


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

    else:
        raise RuntimeError(f"Unknown source prior: {name}")

    z = z.float()
    z.requires_grad_(True)
    return z


class RandomMultivariateFunction:
    """
    Random multivariate function built by round-based stochastic pool reduction.

    Start:
        current_pool = [x1, x2, ..., xK]

    Within each round:
        - sample unary / binary / ternary according to the number of remaining expressions
        - sample the required expressions only from current_pool
        - remove the selected expressions from current_pool
        - apply the sampled operation
        - place the result into next_pool

    Results produced during a round cannot be selected again until the next round.

    At the end of a round:
        current_pool = next_pool

    Repeat until exactly one expression remains. That expression defines the child mechanism.

    The whole symbolic program is sampled once at SCM construction time and remains fixed
    for the lifetime of the SCM.
    """

    UNARY_OPS = ("identity", "scale", "tanh", "sin", "square", "abs", "softplus")
    BINARY_OPS = ("add", "sub", "mul", "safe_div")
    TERNARY_OPS = ("sum3", "mul_add", "mul_sub", "gated_mix")

    UNARY = 0
    BINARY = 1
    TERNARY = 2

    def __init__(
            self,
            num_inputs, 
            generator, 
            device, 
            arity_probs=(0.25, 0.45, 0.30), 
            unary_op_probs=(0.05, 0.15, 0.20, 0.20, 0.15, 0.10, 0.15), 
            binary_op_probs=(0.25, 0.20, 0.35, 0.20), 
            ternary_op_probs=(0.20, 0.30, 0.20, 0.30), 
            scale_min=0.25, scale_max=4.0
    ):

        self.num_inputs = int(num_inputs)
        self.generator = generator
        self.device = device

        if self.num_inputs < 1:
            raise ValueError("num_inputs must be >= 1.")

        self.arity_probs = normalize_probs(arity_probs, device, expected_len=3, name="arity_probs")
        self.unary_op_probs = normalize_probs(unary_op_probs, device, expected_len=len(self.UNARY_OPS), name="unary_op_probs")
        self.binary_op_probs = normalize_probs(binary_op_probs, device, expected_len=len(self.BINARY_OPS), name="binary_op_probs")
        self.ternary_op_probs = normalize_probs(ternary_op_probs, device, expected_len=len(self.TERNARY_OPS), name="ternary_op_probs")

        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        if self.scale_min <= 0:
            raise ValueError("scale_min must be > 0.")
        if self.scale_max < self.scale_min:
            raise ValueError("scale_max must be >= scale_min.")
        if self.arity_probs[self.BINARY] <= 0:
            raise ValueError("binary probability must be positive.")

        self.program = self._sample_program()

    def _sample_categorical(self, probs):
        return int(torch.multinomial(probs, 1, generator=self.generator).item())

    def _sample_indices(self, pool_size, count):
        return torch.randperm(pool_size, generator=self.generator, device=self.device)[:count].tolist()

    def _sample_scale(self):
        u = rand((), generator=self.generator, device=self.device)
        log_min = torch.log(torch.tensor(self.scale_min, device=self.device, dtype=torch.float32))
        log_max = torch.log(torch.tensor(self.scale_max, device=self.device, dtype=torch.float32))
        magnitude = torch.exp(log_min + u * (log_max - log_min))
        sign = -1.0 if bool(rand((), generator=self.generator, device=self.device) < 0.5) else 1.0
        return sign * float(magnitude.item())

    def _sample_arity(self, pool_size):
        if pool_size < 2:
            raise ValueError(f"_sample_arity requires pool_size >= 2, got {pool_size}.")

        probs = self.arity_probs.clone()

        if pool_size >= 6:
            probs[self.UNARY] *= 0.25
            probs[self.TERNARY] *= 1.50
        elif pool_size >= 4:
            probs[self.UNARY] *= 0.50
            probs[self.TERNARY] *= 1.25
        elif pool_size == 3:
            probs[self.UNARY] *= 0.75
        elif pool_size == 2:
            probs[self.TERNARY] = 0.0

        probs = probs / probs.sum().clamp_min(1e-12)
        return self._sample_categorical(probs)

    def _sample_program(self):
        pool = [("input", input_idx) for input_idx in range(self.num_inputs)]

        while len(pool) > 1:
            current_pool = list(pool)
            next_pool = []

            while current_pool:
                pool_size = len(current_pool)

                if pool_size == 1:
                    next_pool.append(current_pool.pop())
                    continue

                arity = self._sample_arity(pool_size)

                if arity == self.UNARY:
                    idx = self._sample_indices(pool_size, 1)[0]
                    child = current_pool.pop(idx)             
                    op = self.UNARY_OPS[self._sample_categorical(self.unary_op_probs)]
                    parameter = self._sample_scale() if op == "scale" else None
                    next_pool.append(("unary", op, parameter, child))

                elif arity == self.BINARY:
                    i, j = self._sample_indices(pool_size, 2)
                    x1, x2 = current_pool[i], current_pool[j]
                    for idx in sorted((i, j), reverse=True):
                        current_pool.pop(idx)

                    op = self.BINARY_OPS[self._sample_categorical(self.binary_op_probs)]

                    if op == "mul":
                        self.binary_op_probs[2] = 0.0
                        self.ternary_op_probs[1] = 0.0
                        self.ternary_op_probs[2] = 0.0
                        self.binary_op_probs = self.binary_op_probs / self.binary_op_probs.sum().clamp_min(1e-12)
                        self.ternary_op_probs = self.ternary_op_probs / self.ternary_op_probs.sum().clamp_min(1e-12)

                    next_pool.append(("binary", op, x1, x2))

                elif arity == self.TERNARY:
                    i, j, k = self._sample_indices(pool_size, 3)
                    x1, x2, x3 = current_pool[i], current_pool[j], current_pool[k]
                    for idx in sorted((i, j, k), reverse=True):
                        current_pool.pop(idx)

                    op = self.TERNARY_OPS[self._sample_categorical(self.ternary_op_probs)]

                    if op in ("mul_add", "mul_sub"):
                        self.binary_op_probs[2] = 0.0
                        self.ternary_op_probs[1] = 0.0
                        self.ternary_op_probs[2] = 0.0
                        self.binary_op_probs = self.binary_op_probs / self.binary_op_probs.sum().clamp_min(1e-12)
                        self.ternary_op_probs = self.ternary_op_probs / self.ternary_op_probs.sum().clamp_min(1e-12)

                    next_pool.append(("ternary", op, x1, x2, x3))

                else:
                    raise RuntimeError(f"Unknown arity: {arity}")

            pool = next_pool

        return pool[0]

    def _apply_unary(self, op, x, parameter):
        if op == "identity":
            return x
        if op == "scale":
            return parameter * x
        if op == "tanh":
            return torch.tanh(x)
        if op == "sin":
            return torch.sin(x)
        if op == "square":
            return torch.clamp(x, -4.0, 4.0).square()
        if op == "abs":
            return torch.abs(x)
        if op == "softplus":
            return F.softplus(torch.clamp(x, -10.0, 10.0))
        raise RuntimeError(f"Unknown unary op: {op}")

    def _apply_binary(self, op, x1, x2):
        if op == "add":
            return x1 + x2
        if op == "sub":
            return x1 - x2
        if op == "mul":
            return torch.clamp(x1, -6.0, 6.0) * torch.clamp(x2, -6.0, 6.0)
        if op == "safe_div":
            return x1 / (1.0 + torch.abs(x2))
        raise RuntimeError(f"Unknown binary op: {op}")

    def _apply_ternary(self, op, x1, x2, x3):
        if op == "sum3":
            return x1 + x2 + x3
        if op == "mul_add":
            return torch.clamp(x1, -6.0, 6.0) * torch.clamp(x2, -6.0, 6.0) + x3
        if op == "mul_sub":
            return torch.clamp(x1, -6.0, 6.0) * torch.clamp(x2, -6.0, 6.0) - x3
        if op == "gated_mix":
            gate = torch.sigmoid(torch.clamp(x3, -10.0, 10.0))
            return gate * x1 + (1.0 - gate) * x2
        raise RuntimeError(f"Unknown ternary op: {op}")

    def _evaluate(self, node, x):
        kind = node[0]

        if kind == "input":
            input_idx = node[1]
            return x[:, input_idx:input_idx + 1]
        if kind == "unary":
            _, op, parameter, child = node
            return self._apply_unary(op, self._evaluate(child, x), parameter)
        if kind == "binary":
            _, op, left, right = node
            return self._apply_binary(op, self._evaluate(left, x), self._evaluate(right, x))
        if kind == "ternary":
            _, op, first, second, third = node
            return self._apply_ternary(op, self._evaluate(first, x), self._evaluate(second, x), self._evaluate(third, x))

        raise RuntimeError(f"Unknown node kind: {kind}")

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.num_inputs:
            raise ValueError(f"Expected [N, {self.num_inputs}], received {tuple(x.shape)}.")
        return self._evaluate(self.program, x.float())


class ScalarLayerConnection:
    """
    For each child:
        1. adjacency determines connected parents
        2. connected parents initialize the expression pool
        3. one fixed RandomMultivariateFunction reduces them to one scalar

    No parent-level edge weights.
    """

    def __init__(
            self,
            in_width,
            out_width,
            connection_prob,
            g_dag,
            g_x,
            device,
            source_prior_probs=(0.45, 0.20, 0.15, 0.05),
            arity_probs=(0.25, 0.45, 0.30),
            unary_op_probs=(0.05, 0.15, 0.20, 0.20, 0.15, 0.10, 0.15),
            binary_op_probs=(0.25, 0.20, 0.35, 0.20),
            ternary_op_probs=(0.20, 0.30, 0.20, 0.30),
            scale_min=0.25,
            scale_max=4.0
    ):

        self.in_width = int(in_width)
        self.out_width = int(out_width)
        self.g_dag = g_dag
        self.g_x = g_x
        self.device = device

        self.source_prior_probs = normalize_probs(source_prior_probs, device, expected_len=4, name="source_prior_probs")
        self.adj = rand(self.in_width, self.out_width, generator=self.g_dag, device=device) < connection_prob
        self.child_functions = [None for _ in range(self.out_width)]

        for child in range(self.out_width):
            parents = torch.where(self.adj[:, child])[0]

            if parents.numel() == 0:
                continue

            self.child_functions[child] = RandomMultivariateFunction(
                num_inputs=int(parents.numel()),
                generator=self.g_dag,
                device=device,
                arity_probs=arity_probs,
                unary_op_probs=unary_op_probs,
                binary_op_probs=binary_op_probs,
                ternary_op_probs=ternary_op_probs,
                scale_min=scale_min, scale_max=scale_max
            )

    def __call__(self, parent_latents, generator, latent_noise_scale=0.0):
        children = []
        n = parent_latents[0].shape[0]

        for child in range(self.out_width):
            parents = torch.where(self.adj[:, child])[0]

            if parents.numel() == 0:
                value = sample_latent(n, self.source_prior_probs, self.g_x, self.device)
            else:
                parent_matrix = torch.cat([parent_latents[parent] for parent in parents.tolist()], dim=1)
                child_function = self.child_functions[child]

                if child_function is None:
                    raise RuntimeError(f"Missing random function for child={child}.")

                value = child_function(parent_matrix)

            if value is None:
                raise RuntimeError(f"Child {child} produced no value.")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Child {child} produced non-finite values.")

            if latent_noise_scale > 0:
                noise = torch.randn(value.shape, generator=generator, device=self.device, dtype=value.dtype)
                value = value + float(latent_noise_scale) * noise

            value = standardize(value, dim=0)
            children.append(value)

        return children


class WeightedLayeredScalarSCM:
    """
    Every SCM node stores one continuous scalar per sample.
    Each node tensor has shape [N, 1].
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
            latent_noise_scale=0.03,
            source_prior_probs=(0.45, 0.20, 0.15, 0.05),
            arity_probs=(0.25, 0.45, 0.30),
            unary_op_probs=(0.05, 0.15, 0.20, 0.20, 0.15, 0.10, 0.15),
            binary_op_probs=(0.25, 0.20, 0.35, 0.20),
            ternary_op_probs=(0.20, 0.30, 0.20, 0.30),
            scale_min=0.25,
            scale_max=4.0,
            device=None
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
            connection = ScalarLayerConnection(
                in_width=self.widths[layer],
                out_width=self.widths[layer + 1],
                connection_prob=self.connection_probs[layer],
                g_dag=self.g_dag,
                g_x=self.g_x,
                device=self.device,
                source_prior_probs=self.source_prior_probs,
                arity_probs=arity_probs,
                unary_op_probs=unary_op_probs,
                binary_op_probs=binary_op_probs,
                ternary_op_probs=ternary_op_probs,
                scale_min=scale_min,
                scale_max=scale_max
            )
            self.connections.append(connection)

    def forward(self, n_samples, latent_noise_scale=None):
        current = [sample_latent(n_samples, self.source_prior_probs, self.g_x, self.device) for _ in range(self.num_roots)]
        all_latents = [current]
        noise_scale = self.latent_noise_scale if latent_noise_scale is None else float(latent_noise_scale)

        for connection in self.connections:
            current = connection(current, generator=self.g_aleatoric, latent_noise_scale=noise_scale)
            all_latents.append(current)

        return all_latents

    def compute_sampling_influence(self, target_node_idx=0, decay=1.0):
        """
        Structural influence based on adjacency path propagation.

        Since there are no parent-level edge weights anymore,
        influence is propagated through adjacency only.
        """
        if not (0 <= target_node_idx < self.widths[-1]):
            raise ValueError("Invalid target_node_idx.")

        influence = [torch.zeros(width, device=self.device, dtype=torch.float32) for width in self.widths]
        influence[-1][target_node_idx] = 1.0

        for layer in range(self.num_layers - 2, -1, -1):
            adjacency = self.connections[layer].adj.float()
            influence[layer] = decay * adjacency @ influence[layer + 1]

        return influence

    def compute_node_influence(self, all_latents, node_indices, target_node_idx=0):
        """
        Scale-normalized local functional influence.
        strength_j = E[|d target / d node_j|] * std(node_j) / std(target)
        """
        target = all_latents[-1][target_node_idx]
        nodes = [all_latents[layer_idx][node_idx] for layer_idx, node_idx in node_indices]

        grads = torch.autograd.grad(
            outputs=target,
            inputs=nodes,
            grad_outputs=torch.ones_like(target),
            retain_graph=False,
            allow_unused=True
        )

        target_std = target.detach().float().std(unbiased=False)

        if not torch.isfinite(target_std) or target_std <= 1e-8:
            return torch.zeros(len(nodes), device=self.device, dtype=torch.float32)

        strengths = []
        for node, grad in zip(nodes, grads):
            if grad is None:
                strength = torch.zeros((), device=self.device, dtype=torch.float32)
            else:
                node_std = node.detach().float().std(unbiased=False)
                if not torch.isfinite(node_std) or node_std <= 1e-8:
                    strength = torch.zeros((), device=self.device, dtype=torch.float32)
                else:
                    grad_mag = grad.detach().abs().mean().float()
                    strength = grad_mag * node_std / target_std
            strengths.append(strength)
        return torch.stack(strengths)
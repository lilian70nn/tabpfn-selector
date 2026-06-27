# scm_task.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import torch
from src.data.helper import make_gen, stratified_classification_split, discretize_latent_random_bins
from src.data.synthetic_task import GenerateTask


NodeKind = Literal["cont", "cat"]


@dataclass
class NodeSpec:
    kind: NodeKind
    K: Optional[int] = None  # only used when kind == "cat"


def _randn(*shape, generator: torch.Generator, device: torch.device):
    return torch.randn(*shape, generator=generator, device=device)


def _rand(*shape, generator: torch.Generator, device: torch.device):
    return torch.rand(*shape, generator=generator, device=device)


def _randint(
    low: int,
    high: int,
    shape,
    generator: torch.Generator,
    device: torch.device,
):
    return torch.randint(low, high, shape, generator=generator, device=device)



# Edge functions
class BaseEdge:
    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ContToContEdge(BaseEdge):
    EDGE_NAMES = {
        0: "linear",
        1: "tanh",
        2: "sin",
        3: "quadratic",
        4: "relu",
        5: "sigmoid",
        6: "small_mlp",
        7: "fourier",
        8: "depth2_tree",
    }

    def __init__(self, generator: torch.Generator, device: torch.device):
        self.edge_type = _randint(0, 9, (), generator=generator, device=device).item()
        self.edge_name = self.EDGE_NAMES[self.edge_type]

        self.a = _randn((), generator=generator, device=device)
        self.b = _randn((), generator=generator, device=device)
        self.c = _randn((), generator=generator, device=device)

        # edge_type == 6: small MLP, 1 -> 8 -> 1
        H = 8
        self.W1 = _randn(H, 1, generator=generator, device=device)
        self.b1 = _randn(H, generator=generator, device=device)
        self.W2 = _randn(1, H, generator=generator, device=device) / (H ** 0.5)
        self.b2 = _randn(1, generator=generator, device=device)

        # edge_type == 7: Fourier random features
        R = 8
        self.freq = 3.0 * _randn(R, generator=generator, device=device)
        self.phase = 6.28318530718 * _rand(R, generator=generator, device=device)
        self.fourier_w = _randn(R, generator=generator, device=device) / (R ** 0.5)

        # edge_type == 8: depth-2 1D decision tree
        self.tree_t0 = _randn((), generator=generator, device=device)
        self.tree_t1 = _randn((), generator=generator, device=device)
        self.tree_t2 = _randn((), generator=generator, device=device)
        self.tree_leaf = _randn(4, generator=generator, device=device)

    def name(self):
        return self.edge_name

    def _std(self, y: torch.Tensor) -> torch.Tensor:
        std = y.std(unbiased=False)
        if float(std.item()) < 1e-6:
            return y - y.mean()
        return (y - y.mean()) / std

    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        x = parent_value.float()
        x = self._std(x)

        if self.edge_type == 0:
            y = self.a * x + self.b

        elif self.edge_type == 1:
            y = torch.tanh(self.a * x + self.b)

        elif self.edge_type == 2:
            y = torch.sin(self.a * x + self.b)

        elif self.edge_type == 3:
            y = self.a * (x ** 2) + self.b * x + self.c

        elif self.edge_type == 4:
            y = torch.relu(self.a * x + self.b)

        elif self.edge_type == 5:
            y = torch.sigmoid(self.a * x + self.b) - 0.5

        elif self.edge_type == 6:
            h = torch.tanh(x[:, None] @ self.W1.T + self.b1)
            y = (h @ self.W2.T).squeeze(-1) + self.b2.squeeze(0)

        elif self.edge_type == 7:
            z = torch.sin(x[:, None] * self.freq[None, :] + self.phase[None, :])
            y = z @ self.fourier_w

        elif self.edge_type == 8:
            left_root = x <= self.tree_t0

            left_leaf = torch.where(
                x <= self.tree_t1,
                self.tree_leaf[0],
                self.tree_leaf[1],
            )

            right_leaf = torch.where(
                x <= self.tree_t2,
                self.tree_leaf[2],
                self.tree_leaf[3],
            )

            y = torch.where(left_root, left_leaf, right_leaf)

        else:
            raise RuntimeError(f"Unknown edge_type={self.edge_type}")

        return self._std(y)



class ContToCatEdge(BaseEdge):
    """
    continuous -> categorical logits

    edge_type:
        0 bucket_logits   
        1 prototype_logits
        2 small_mlp_logits
    """

    EDGE_NAMES = {
        0: "bucket_logits",
        1: "prototype_logits",
        2: "small_mlp_logits",
    }

    def __init__(
        self,
        child_cardinality: int,
        num_bins: int,
        generator: torch.Generator,
        device: torch.device,
    ):
        self.child_K = child_cardinality
        self.num_bins = num_bins
        self.edge_type = _randint(0, 3, (), generator=generator, device=device).item()
        self.edge_name = self.EDGE_NAMES[self.edge_type]

        # edge_type == 0: original bucket logits
        raw = 3.0 * _randn(num_bins - 1, generator=generator, device=device)
        self.thresholds = torch.sort(raw).values

        self.bin_logits = _randn(
            num_bins,
            child_cardinality,
            generator=generator,
            device=device,
        )

        # edge_type == 1: prototype logits
        self.prototypes = 2.0 * _randn(
            child_cardinality,
            generator=generator,
            device=device,
        )
        self.prototype_scale = torch.rand((), generator=generator, device=device) + 0.5

        # edge_type == 2: small MLP logits, 1 -> 8 -> child_K
        H = 8
        self.W1 = _randn(H, 1, generator=generator, device=device)
        self.b1 = _randn(H, generator=generator, device=device)
        self.W2 = _randn(child_cardinality, H, generator=generator, device=device) / (H ** 0.5)
        self.b2 = _randn(child_cardinality, generator=generator, device=device)

    def name(self):
        return self.edge_name

    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        x = parent_value.float()
        x = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)

        if self.edge_type == 0:
            # original: discretize continuous x into bins, then lookup logits
            b = torch.bucketize(x, self.thresholds)
            logits = self.bin_logits[b]

        elif self.edge_type == 1:
            # category k is likely if x is close to prototype_k
            dist = (x[:, None] - self.prototypes[None, :]) ** 2
            logits = -self.prototype_scale * dist

        elif self.edge_type == 2:
            # small neural network maps scalar x to child_K logits
            h = torch.tanh(x[:, None] @ self.W1.T + self.b1)
            logits = h @ self.W2.T + self.b2

        else:
            raise RuntimeError(f"Unknown edge_type={self.edge_type}")

        return logits
    

    
class CatToContEdge(BaseEdge):
    """
    categorical -> continuous

    edge_type:
        0 lookup          category -> scalar table
        1 embedding_mlp   category embedding -> small MLP -> scalar
    """

    EDGE_NAMES = {
        0: "lookup",
        1: "embedding_mlp",
    }

    def __init__(
        self,
        parent_cardinality: int,
        generator: torch.Generator,
        device: torch.device,
    ):
        self.parent_K = parent_cardinality
        self.edge_type = _randint(0, 2, (), generator=generator, device=device).item()
        self.edge_name = self.EDGE_NAMES[self.edge_type]

        # edge_type == 0: original lookup
        self.values = _randn(parent_cardinality, generator=generator, device=device)

        # edge_type == 1: embedding MLP
        E = 4
        H = 8
        self.emb = _randn(parent_cardinality, E, generator=generator, device=device)
        self.W1 = _randn(H, E, generator=generator, device=device) / (E ** 0.5)
        self.b1 = _randn(H, generator=generator, device=device)
        self.W2 = _randn(1, H, generator=generator, device=device) / (H ** 0.5)
        self.b2 = _randn(1, generator=generator, device=device)

    def name(self):
        return self.edge_name
    
    def _std(self, y: torch.Tensor) -> torch.Tensor:
        std = y.std(unbiased=False)
        if float(std.item()) < 1e-6:
            return y - y.mean()
        return (y - y.mean()) / std

    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        c = parent_value.long()

        if self.edge_type == 0:
            y = self.values[c]

        elif self.edge_type == 1:
            e = self.emb[c]  # [batch, E]
            h = torch.tanh(e @ self.W1.T + self.b1)
            y = (h @ self.W2.T).squeeze(-1) + self.b2.squeeze(0)

        else:
            raise RuntimeError(f"Unknown edge_type={self.edge_type}")

        return self._std(y)



class CatToCatEdge(BaseEdge):
    """
    categorical -> categorical logits

    edge_type:
        0 transition_table
        1 embedding_mlp_logits

    Output shape:
        parent_value: [batch], integer category ids
        output:       [batch, child_K]
    """

    EDGE_NAMES = {
        0: "transition_table",
        1: "embedding_mlp_logits",
    }

    def __init__(
        self,
        parent_cardinality: int,
        child_cardinality: int,
        generator: torch.Generator,
        device: torch.device,
    ):
        self.parent_K = parent_cardinality
        self.child_K = child_cardinality
        self.edge_type = _randint(0, 2, (), generator=generator, device=device).item()
        self.edge_name = self.EDGE_NAMES[self.edge_type]
        # edge_type == 0: original transition table
        self.logits_table = _randn(
            parent_cardinality,
            child_cardinality,
            generator=generator,
            device=device,
        )

        # edge_type == 1: category embedding -> small MLP -> child logits
        E = 4
        H = 8
        self.emb = _randn(parent_cardinality, E, generator=generator, device=device)
        self.W1 = _randn(H, E, generator=generator, device=device) / (E ** 0.5)
        self.b1 = _randn(H, generator=generator, device=device)
        self.W2 = _randn(child_cardinality, H, generator=generator, device=device) / (H ** 0.5)
        self.b2 = _randn(child_cardinality, generator=generator, device=device)


    def name(self):
        return self.edge_name
    

    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        c = parent_value.long()

        if self.edge_type == 0:
            logits = self.logits_table[c]

        elif self.edge_type == 1:
            e = self.emb[c]  # [batch, E]
            h = torch.tanh(e @ self.W1.T + self.b1)
            logits = h @ self.W2.T + self.b2

        else:
            raise RuntimeError(f"Unknown edge_type={self.edge_type}")

        return logits


def sample_edge(
    parent_spec: NodeSpec,
    child_spec: NodeSpec,
    num_bins: int,
    generator: torch.Generator,
    device: torch.device,
) -> BaseEdge:
    """
    Decide edge-function family by parent node type and child node type.
    """

    if parent_spec.kind == "cont" and child_spec.kind == "cont":
        return ContToContEdge(generator=generator, device=device)

    if parent_spec.kind == "cont" and child_spec.kind == "cat":
        assert child_spec.K is not None
        return ContToCatEdge(
            child_cardinality=child_spec.K,
            num_bins=num_bins,
            generator=generator,
            device=device,
        )

    if parent_spec.kind == "cat" and child_spec.kind == "cont":
        assert parent_spec.K is not None
        return CatToContEdge(
            parent_cardinality=parent_spec.K,
            generator=generator,
            device=device,
        )

    if parent_spec.kind == "cat" and child_spec.kind == "cat":
        assert parent_spec.K is not None
        assert child_spec.K is not None
        return CatToCatEdge(
            parent_cardinality=parent_spec.K,
            child_cardinality=child_spec.K,
            generator=generator,
            device=device,
        )

    raise ValueError(f"Unknown edge type: {parent_spec} -> {child_spec}")


# ============================================================
# One layer connection
# ============================================================

class LayerConnection:
    """
    Sparse random connection from layer l to layer l+1.

    adj[i, j] == True means:
        parent node i in previous layer connects to child node j in next layer.

    edges[i][j] is the edge function f if adj[i, j] == True.
    """

    def __init__(
        self,
        parent_specs: list[NodeSpec],
        child_specs: list[NodeSpec],
        edge_prob: float,
        min_parents_per_node: int,
        num_bins: int,
        generator: torch.Generator,
        device: torch.device,
    ):
        self.parent_specs = parent_specs
        self.child_specs = child_specs
        self.in_width = len(parent_specs)
        self.out_width = len(child_specs)
        self.device = device

        if self.in_width <= 0:
            raise ValueError("in_width must be positive.")
        if self.out_width <= 0:
            raise ValueError("out_width must be positive.")

        min_parents = min(min_parents_per_node, self.in_width)

        # Step 1: random sparse adjacency.
        self.adj = _rand(
            self.in_width,
            self.out_width,
            generator=generator,
            device=device,
        ) < edge_prob

        # Step 2: guarantee every child has at least min_parents parents.
        for j in range(self.out_width):
            current_parents = int(self.adj[:, j].sum().item())

            if current_parents < min_parents:
                missing = min_parents - current_parents

                candidates = torch.where(~self.adj[:, j])[0]
                perm = candidates[
                    torch.randperm(
                        len(candidates),
                        generator=generator,
                        device=device,
                    )
                ]

                chosen = perm[:missing]
                self.adj[chosen, j] = True

        # Step 3: create edge functions for every 1 position.
        self.edges: list[list[Optional[BaseEdge]]] = [
            [None for _ in range(self.out_width)]
            for _ in range(self.in_width)
        ]

        for i in range(self.in_width):
            for j in range(self.out_width):
                if self.adj[i, j]:
                    self.edges[i][j] = sample_edge(
                        parent_spec=parent_specs[i],
                        child_spec=child_specs[j],
                        num_bins=num_bins,
                        generator=generator,
                        device=device,
                    )

    def __call__(
        self,
        parent_values: list[torch.Tensor],
        generator: torch.Generator,
        sample_categorical: bool = True,
        noise_scale: float = 0.0,
    ) -> list[torch.Tensor]:
        """
        Compute next layer.

        For continuous child:
            child_value = sum scalar edge outputs

        For categorical child:
            child_logits = sum logits edge outputs
            then sample category or take argmax
        """

        if len(parent_values) != self.in_width:
            raise ValueError(
                f"Expected {self.in_width} parent values, got {len(parent_values)}."
            )

        child_values: list[torch.Tensor] = []

        for j, child_spec in enumerate(self.child_specs):
            incoming_outputs = []

            for i in range(self.in_width):
                edge = self.edges[i][j]
                if edge is not None:
                    incoming_outputs.append(edge(parent_values[i]))

            if len(incoming_outputs) == 0:
                raise RuntimeError(
                    "This should not happen because every child has parents."
                )

            combined = torch.stack(incoming_outputs, dim=0).sum(dim=0)

            if child_spec.kind == "cont":
                combined = combined.float()
                combined = (combined - combined.mean()) / combined.std(unbiased=False).clamp_min(1e-6)

                if noise_scale > 0:
                    combined = combined + noise_scale * torch.randn(
                        combined.shape,
                        generator=generator,
                        device=self.device,
                    )
                    combined = (combined - combined.mean()) / combined.std(unbiased=False).clamp_min(1e-6)

                child_values.append(combined)

            else:
                if sample_categorical:
                    probs = torch.softmax(combined, dim=-1)
                    sampled = torch.multinomial(
                        probs,
                        num_samples=1,
                        replacement=True,
                        generator=generator,
                    ).squeeze(-1)
                    child_values.append(sampled)
                else:
                    child_values.append(torch.argmax(combined, dim=-1))

        return child_values


# Full layered SCM task
class RandomLayeredSCM:
    """
    Random sparse layered SCM.

    Important:
        num_layers includes the root layer.

    Example:
        num_roots = 3
        num_layers = 4

    Then the graph has:

        layer 0: root layer, width = 3
        layer 1: random width
        layer 2: random width
        layer 3: random width

    Edges only go from layer l to layer l+1.
    Therefore the graph is automatically a DAG.
    """

    def __init__(
        self,
        g_dag: torch.Generator,
        g_x: torch.Generator,
        g_aleatoric: torch.Generator,
        num_roots: int = 3,
        num_layers: int = 4,
        max_nodes_per_layer: int = 5,
        edge_prob: float = 0.35,
        p_cat: float = 0.3,
        max_cardinality: int = 5,
        min_parents_per_node: int = 1,
        num_bins: int = 5,
        node_noise_scale: float = 0.05,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cpu")

        if num_roots <= 0:
            raise ValueError("num_roots must be positive.")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")
        if max_nodes_per_layer <= 0:
            raise ValueError("max_nodes_per_layer must be positive.")
        if not (0.0 <= edge_prob <= 1.0):
            raise ValueError("edge_prob must be in [0, 1].")
        if not (0.0 <= p_cat <= 1.0):
            raise ValueError("p_cat must be in [0, 1].")
        if max_cardinality < 2:
            raise ValueError("max_cardinality must be at least 2.")
        if min_parents_per_node < 1:
            raise ValueError("min_parents_per_node must be at least 1.")
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2.")

        self.num_roots = num_roots
        self.num_layers = num_layers
        self.max_nodes_per_layer = max_nodes_per_layer
        self.edge_prob = edge_prob
        self.p_cat = p_cat
        self.max_cardinality = max_cardinality
        self.min_parents_per_node = min_parents_per_node
        self.num_bins = num_bins
        self.node_noise_scale = node_noise_scale
        self.device = device

        self.g_dag = g_dag
        self.g_x = g_x
        self.g_aleatoric = g_aleatoric

        # 1. Generate widths.
        self.widths = self._sample_widths()

        # 2. Generate node specs.
        self.layers: list[list[NodeSpec]] = []
        for width in self.widths:
            specs = [
                self._sample_node_spec()
                for _ in range(width)
            ]
            self.layers.append(specs)

        # 3. Generate sparse connections and edge functions.
        self.connections: list[LayerConnection] = []

        for l in range(num_layers - 1):
            conn = LayerConnection(
                parent_specs=self.layers[l],
                child_specs=self.layers[l + 1],
                edge_prob=edge_prob,
                min_parents_per_node=min_parents_per_node,
                num_bins=num_bins,
                generator=self.g_dag,
                device=device,
            )
            self.connections.append(conn)

    def _sample_widths(self) -> list[int]:
        widths = [self.num_roots]

        for _ in range(self.num_layers - 1):
            width = _randint(
                1,
                self.max_nodes_per_layer + 1,
                (),
                generator=self.g_dag,
                device=self.device,
            ).item()
            widths.append(width)

        return widths

    def _sample_node_spec(self) -> NodeSpec:
        u = _rand((), generator=self.g_dag, device=self.device).item()

        if u < self.p_cat:
            K = _randint(
                2,
                self.max_cardinality + 1,
                (),
                generator=self.g_dag,
                device=self.device,
            ).item()
            return NodeSpec(kind="cat", K=K)

        return NodeSpec(kind="cont", K=None)

    def sample_roots(self, n_samples: int) -> list[torch.Tensor]:
        """
        Sample root node values.

        Continuous root:
            randn, shape [batch]

        Categorical root:
            randint(0, K), shape [batch]
        """

        root_values = []

        for spec in self.layers[0]:
            if spec.kind == "cont":
                value = _randn(
                    n_samples,
                    generator=self.g_x,
                    device=self.device,
                )
            else:
                assert spec.K is not None
                value = _randint(
                    0,
                    spec.K,
                    (n_samples,),
                    generator=self.g_x,
                    device=self.device,
                )

            root_values.append(value)

        return root_values

    def forward(
        self,
        root_values: Optional[list[torch.Tensor]] = None,
        n_samples: Optional[int] = None,
        sample_categorical: bool = True,
        noise_scale: Optional[float] = None,
    ) -> list[list[torch.Tensor]]:
        """
        Run SCM forward.

        Returns:
            all_values[l][j] is value of node j in layer l.

        If root_values is None, batch_size must be provided.
        """

        if root_values is None:
            if n_samples is None:
                raise ValueError("Either root_values or n_samples must be provided.")
            current_values = self.sample_roots(n_samples)
        else:
            current_values = root_values
        
        if noise_scale is None:
            noise_scale = self.node_noise_scale

        all_values = [current_values]

        for conn in self.connections:
            current_values = conn(
                current_values,
                generator=self.g_aleatoric,
                sample_categorical=sample_categorical,
                noise_scale=noise_scale,
            )
            all_values.append(current_values)

        return all_values

    

    def reforward_after_intervention(
        self,
        all_values,
        start_layer,
        sample_categorical=False,
    ):
        new_values = [list(layer) for layer in all_values]
        current_values = new_values[start_layer]

        for l in range(start_layer, self.num_layers - 1):
            current_values = self.connections[l](
                current_values,
                generator=self.g_aleatoric,
                sample_categorical=sample_categorical,
                noise_scale=0.0,
            )
            new_values[l + 1] = current_values

        return new_values
    

    def describe(self) -> None:
        """
        Print graph structure.
        """

        print("========== RandomLayeredSCMTask ==========")
        print(f"widths: {self.widths}")
        print()

        for l, specs in enumerate(self.layers):
            print(f"Layer {l}:")
            for j, spec in enumerate(specs):
                if spec.kind == "cont":
                    print(f"  node {j}: cont")
                else:
                    print(f"  node {j}: cat, K={spec.K}")
            print()

        for l, conn in enumerate(self.connections):
            print(f"Connection layer {l} -> layer {l + 1}:")
            print(conn.adj.long())
            print(f"num_edges = {int(conn.adj.sum().item())}")
            for i in range(conn.in_width):
                for j in range(conn.out_width):
                    edge = conn.edges[i][j]
                    if edge is not None:
                        edge_name = edge.name() if hasattr(edge, "name") else edge.__class__.__name__
                        print(f"  edge {i}->{j}: {edge.__class__.__name__}, {edge_name}")
            print()


class MixedSCMTask(GenerateTask):
    CONTINUOUS = 0
    CATEGORICAL = 1

    def __init__(
        self,
        num_classes=None,
        n_max=500,
        d_max=20,
        n_min=128,
        d_min=2,
        test_frac=0.15,
        p_missing=0.05,
        node_noise_scale=0.05,
        device=None,
        dag_seed=None,
        aleatoric_seed=None,
        x_seed=None,
        num_roots=3,
        num_layers=4,
        max_nodes_per_layer=8,
        edge_prob=0.35,
        p_cat=0.3,
        max_cardinality=10,
        min_parents_per_node=1,
        num_bins=5,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.n_max = int(n_max)
        self.d_max = int(d_max)
        self.n_min = int(n_min)
        self.d_min = int(d_min)
        self.test_frac = float(test_frac)
        self.p_missing = float(p_missing)
        self.node_noise_scale = float(node_noise_scale)

        self.num_roots = int(num_roots)
        self.num_layers = int(num_layers)
        self.max_nodes_per_layer = int(max_nodes_per_layer)
        self.edge_prob = float(edge_prob)
        self.p_cat = float(p_cat)
        self.max_cardinality = int(max_cardinality)
        self.min_parents_per_node = int(min_parents_per_node)
        self.num_bins = int(num_bins)

        self.g_dag, self.dag_seed = make_gen(self.device, dag_seed)
        self.g_aleatoric, self.aleatoric_seed = make_gen(self.device, aleatoric_seed)
        self.g_x, self.x_seed = make_gen(self.device, x_seed)

        self.d = torch.randint(
            self.d_min, self.d_max + 1, (1,),
            device=self.device, generator=self.g_dag
        ).item()

        self.n = torch.randint(
            self.n_min, self.n_max + 1, (1,),
            device=self.device, generator=self.g_dag
        ).item()

        super().__init__()

    def _flatten_values(self, scm, all_values):
        flat_values = []
        flat_specs = []
        flat_index = []

        for l, layer_values in enumerate(all_values):
            for j, value in enumerate(layer_values):
                flat_values.append(value)
                flat_specs.append(scm.layers[l][j])
                flat_index.append((l, j))

        return flat_values, flat_specs, flat_index
    
    def _sample_feature_and_target_sources(
        self,
        flat_specs,
        flat_index,
        d,
        allow_target_as_feature=False,
    ):
        all_ids = list(range(len(flat_specs)))
        cont_ids = [i for i, spec in enumerate(flat_specs) if spec.kind == "cont"]

        if len(cont_ids) == 0:
            raise RuntimeError("No continuous node available for target.")

        max_layer = max(l for l, _ in flat_index)

        target_pool = [
            i for i, spec in enumerate(flat_specs)
            if spec.kind == "cont" and flat_index[i][0] == max_layer
        ]

        if len(target_pool) == 0:
            target_pool = [
                i for i, spec in enumerate(flat_specs)
                if spec.kind == "cont"
            ]

        if len(target_pool) == 0:
            raise RuntimeError("No continuous node available for target.")

        target_pos = _randint(
            0,
            len(target_pool),
            (),
            generator=self.g_dag,
            device=self.device,
        ).item()
        target_id = target_pool[int(target_pos)]

        candidates = all_ids
        if not allow_target_as_feature:
            candidates = [i for i in candidates if i != target_id]

        if len(candidates) < self.d_min:
            raise ValueError(
                f"Not enough feature candidates: got {len(candidates)}, "
                f"but d_min={self.d_min}."
            )

        d = min(d, len(candidates))

        perm = torch.randperm(
            len(candidates),
            generator=self.g_dag,
            device=self.device,
        )
        feature_ids = [candidates[int(i)] for i in perm[:d].tolist()]

        return feature_ids, target_id
    

    def _extract_table_from_sources(
        self,
        flat_values,
        flat_specs,
        feature_ids,
        target_id,
    ):
        n = flat_values[0].shape[0]
        d = len(feature_ids)

        X = torch.empty(n, d, device=self.device, dtype=torch.float32)
        feature_type = torch.empty(d, device=self.device, dtype=torch.long)
        cardinality = torch.zeros(d, device=self.device, dtype=torch.long)

        for j, node_id in enumerate(feature_ids):
            spec = flat_specs[node_id]
            value = flat_values[node_id]

            if spec.kind == "cont":
                col = value.float()
                col = (col - col.mean()) / col.std(unbiased=False).clamp_min(1e-6)
                X[:, j] = col
                feature_type[j] = 0
                cardinality[j] = 0
            else:
                assert spec.K is not None
                X[:, j] = value.long().float()
                feature_type[j] = 1
                cardinality[j] = int(spec.K)

        latent_y = flat_values[target_id].float()
        latent_y = (latent_y - latent_y.mean()) / latent_y.std(unbiased=False).clamp_min(1e-6)

        return X, latent_y, feature_type, cardinality
    

    def _compute_intervention_importance(
        self,
        scm,
        all_values,
        flat_values,
        flat_specs,
        flat_index,
        feature_ids,
        target_id,
    ):
        target_layer, target_node = flat_index[target_id]
        latent_y_original = flat_values[target_id].float()

        strengths = []

        for feature_id in feature_ids:
            source_layer, source_node = flat_index[feature_id]

            # Layered DAG: later/same-layer source cannot affect earlier/same-layer target.
            if source_layer >= target_layer:
                strengths.append(torch.tensor(0.0, device=self.device))
                continue

            intervened = [list(layer) for layer in all_values]

            source_value = intervened[source_layer][source_node]
            perm = torch.randperm(
                source_value.shape[0],
                generator=self.g_x,
                device=self.device,
            )
            intervened[source_layer][source_node] = source_value[perm]

            intervened = scm.reforward_after_intervention(
                intervened,
                start_layer=source_layer,
                sample_categorical=False,
            )

            y_do = intervened[target_layer][target_node].float()
            strength = ((y_do - latent_y_original) ** 2).mean().sqrt()
            strengths.append(strength)

        feature_strength = torch.stack(strengths)

        total = feature_strength.sum()
        if float(total.item()) <= 1e-12:
            importance_ratio = torch.ones_like(feature_strength) / feature_strength.numel()
        else:
            importance_ratio = feature_strength / total.clamp_min(1e-12)

        return feature_strength, importance_ratio

    def _generate(self):
        device = self.device
        n, d = self.n, self.d

        scm = RandomLayeredSCM(
            num_roots=self.num_roots,
            num_layers=self.num_layers,
            max_nodes_per_layer=self.max_nodes_per_layer,
            edge_prob=self.edge_prob,
            p_cat=self.p_cat,
            max_cardinality=self.max_cardinality,
            min_parents_per_node=self.min_parents_per_node,
            num_bins=self.num_bins,
            g_dag=self.g_dag,
            g_x=self.g_x,
            g_aleatoric=self.g_aleatoric,
            node_noise_scale=self.node_noise_scale,
            device=device,
        )

        all_values = scm.forward(n_samples=n, sample_categorical=False, noise_scale=self.node_noise_scale)

        # flatten nodes
        flat_values, flat_specs, flat_index = self._flatten_values(scm, all_values)

        # sample feature nodes and target node
        feature_ids, target_id = self._sample_feature_and_target_sources(
            flat_specs=flat_specs,
            flat_index=flat_index,
            d=d,
            allow_target_as_feature=False,
        )

        d = len(feature_ids)
        self.d = d

        # get X_clean and latent_y
        X_clean, latent_y, feature_type, cardinality = self._extract_table_from_sources(
            flat_values=flat_values,
            flat_specs=flat_specs,
            feature_ids=feature_ids,
            target_id=target_id,
        )

        # importance from intervention
        feature_strength, importance_ratio = self._compute_intervention_importance(
            scm=scm,
            all_values=all_values,
            flat_values=flat_values,
            flat_specs=flat_specs,
            flat_index=flat_index,
            feature_ids=feature_ids,
            target_id=target_id,
        )

        # y
        if self.num_classes is None:
            y = latent_y
            self.n_classes = None
        else:
            C = int(self.num_classes)
            y = discretize_latent_random_bins(
                latent_y=latent_y,
                C=C,
                generator=self.g_aleatoric,
                min_per_class=2,
                alpha=5.0,
            )
            self.n_classes = C

        # missing
        X_obs = X_clean.clone()
        missing_mask = torch.rand(X_obs.shape, device=device, generator=self.g_x) < self.p_missing
        X_obs[missing_mask] = torch.nan

        # split
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
            perm = torch.randperm(n, device=device, generator=self.g_x)
            train_idx = perm[:-n_test]
            test_idx = perm[-n_test:]

        X_train = X_obs[train_idx]
        y_train = y[train_idx]
        X_test = X_obs[test_idx]
        y_test = y[test_idx]

        eps = 1e-8
        is_active = (feature_strength > eps).float()

        info = {
            "feature_type": feature_type,
            "cardinality": cardinality,
            "is_active": is_active,
            "importance_ratio": importance_ratio,
            "feature_strength": feature_strength,
            "sampled_active": is_active,
            "missing_mask_train": missing_mask[train_idx],
            "missing_mask_test": missing_mask[test_idx],
            "feature_ids": torch.tensor(feature_ids, device=device),
            "target_id": torch.tensor(target_id, device=device),
        }
        

        self.n_features = d
        self.feature_type = feature_type
        self.cardinality = cardinality
        self.scm = scm

        return X_train, y_train, X_test, y_test, info
    

    
    def visualize(self):
        return None

    def forward(self, X: torch.Tensor):
        return None


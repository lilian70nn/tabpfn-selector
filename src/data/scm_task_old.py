# scm_task.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Any

import torch


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


# ============================================================
# Edge functions
# ============================================================

class BaseEdge:
    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ContToContEdge(BaseEdge):
    """
    continuous -> continuous

    Output shape:
        parent_value: [batch]
        output:       [batch]
    """

    def __init__(self, generator: torch.Generator, device: torch.device):
        self.edge_type = _randint(0, 3, (), generator=generator, device=device).item()

        self.a = _randn((), generator=generator, device=device)
        self.b = _randn((), generator=generator, device=device)

    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        x = parent_value

        if self.edge_type == 0:
            return self.a * x + self.b

        if self.edge_type == 1:
            return torch.tanh(self.a * x + self.b)

        return torch.sin(self.a * x + self.b)


class ContToCatEdge(BaseEdge):
    """
    continuous -> categorical logits

    First bucketize continuous input into bins.
    Then each bin gives logits over child categories.

    Output shape:
        parent_value: [batch]
        output:       [batch, child_K]
    """

    def __init__(
        self,
        child_cardinality: int,
        num_bins: int,
        generator: torch.Generator,
        device: torch.device,
    ):
        self.child_K = child_cardinality
        self.num_bins = num_bins

        raw = 3.0 * _randn(num_bins - 1, generator=generator, device=device)
        self.thresholds = torch.sort(raw).values

        self.bin_logits = _randn(
            num_bins,
            child_cardinality,
            generator=generator,
            device=device,
        )

    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        x = parent_value
        b = torch.bucketize(x, self.thresholds)
        return self.bin_logits[b]


class CatToContEdge(BaseEdge):
    """
    categorical -> continuous

    Each parent category maps to one scalar value.

    Output shape:
        parent_value: [batch], integer category ids
        output:       [batch]
    """

    def __init__(
        self,
        parent_cardinality: int,
        generator: torch.Generator,
        device: torch.device,
    ):
        self.parent_K = parent_cardinality
        self.values = _randn(parent_cardinality, generator=generator, device=device)

    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        c = parent_value.long()
        return self.values[c]


class CatToCatEdge(BaseEdge):
    """
    categorical -> categorical logits

    Each parent category gives logits over child categories.

    Output shape:
        parent_value: [batch], integer category ids
        output:       [batch, child_K]
    """

    def __init__(
        self,
        parent_cardinality: int,
        child_cardinality: int,
        generator: torch.Generator,
        device: torch.device,
    ):
        self.parent_K = parent_cardinality
        self.child_K = child_cardinality

        self.logits_table = _randn(
            parent_cardinality,
            child_cardinality,
            generator=generator,
            device=device,
        )

    def __call__(self, parent_value: torch.Tensor) -> torch.Tensor:
        c = parent_value.long()
        return self.logits_table[c]


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

        batch_size = parent_values[0].shape[0]
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
                # combined shape: [batch]
                child_values.append(combined)

            else:
                # combined shape: [batch, K]
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


# ============================================================
# Full layered SCM task
# ============================================================

class RandomLayeredSCMTask:
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
        num_roots: int,
        num_layers: int,
        max_nodes_per_layer: int,
        edge_prob: float,
        p_cat: float = 0.3,
        max_cardinality: int = 5,
        min_parents_per_node: int = 1,
        num_bins: int = 5,
        seed: int = 0,
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
        self.device = device

        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

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
                generator=self.generator,
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
                generator=self.generator,
                device=self.device,
            ).item()
            widths.append(width)

        return widths

    def _sample_node_spec(self) -> NodeSpec:
        u = _rand((), generator=self.generator, device=self.device).item()

        if u < self.p_cat:
            K = _randint(
                2,
                self.max_cardinality + 1,
                (),
                generator=self.generator,
                device=self.device,
            ).item()
            return NodeSpec(kind="cat", K=K)

        return NodeSpec(kind="cont", K=None)

    def sample_roots(self, batch_size: int) -> list[torch.Tensor]:
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
                    batch_size,
                    generator=self.generator,
                    device=self.device,
                )
            else:
                assert spec.K is not None
                value = _randint(
                    0,
                    spec.K,
                    (batch_size,),
                    generator=self.generator,
                    device=self.device,
                )

            root_values.append(value)

        return root_values

    def forward(
        self,
        root_values: Optional[list[torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        sample_categorical: bool = True,
    ) -> list[list[torch.Tensor]]:
        """
        Run SCM forward.

        Returns:
            all_values[l][j] is value of node j in layer l.

        If root_values is None, batch_size must be provided.
        """

        if root_values is None:
            if batch_size is None:
                raise ValueError("Either root_values or batch_size must be provided.")
            current_values = self.sample_roots(batch_size)
        else:
            current_values = root_values

        all_values = [current_values]

        for conn in self.connections:
            current_values = conn(
                current_values,
                generator=self.generator,
                sample_categorical=sample_categorical,
            )
            all_values.append(current_values)

        return all_values

    def __call__(
        self,
        root_values: Optional[list[torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        sample_categorical: bool = True,
    ) -> list[list[torch.Tensor]]:
        return self.forward(
            root_values=root_values,
            batch_size=batch_size,
            sample_categorical=sample_categorical,
        )

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
            print()


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    device = torch.device("cpu")

    scm_task = RandomLayeredSCMTask(
        num_roots=3,
        num_layers=4,
        max_nodes_per_layer=5,
        edge_prob=0.35,
        p_cat=0.3,
        max_cardinality=4,
        min_parents_per_node=1,
        num_bins=5,
        seed=123,
        device=device,
    )

    scm_task.describe()

    all_values = scm_task(batch_size=8, sample_categorical=True)

    print("========== Values ==========")
    for l, layer_values in enumerate(all_values):
        print(f"Layer {l}:")
        for j, value in enumerate(layer_values):
            print(f"  node {j}: shape={tuple(value.shape)}, value={value}")
        print()
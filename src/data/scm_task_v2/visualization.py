import torch
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.data.scm_task_v2.task import SCMTask


def build_node_id_maps(scm):
    pair_to_gid = {}
    gid_to_pair = {}

    gid = 0
    for layer_idx, width in enumerate(scm.widths):
        for node_idx in range(width):
            pair_to_gid[(layer_idx, node_idx)] = gid
            gid_to_pair[gid] = (layer_idx, node_idx)
            gid += 1

    return pair_to_gid, gid_to_pair


def plot_scm_graph(task, save_path="scm_graph.png"):
    """
    Visualize the sampled SCM.

    Style:
    - ordinary latent node: white circle
    - selected continuous feature: gray circle
    - selected categorical feature: hatched circle
    - target: gray star
    - edges: structural parent -> child connections
    - selected feature annotation: normalized feature importance
    """

    scm = task.scm
    info = task.info

    feature_ids = info["feature_ids"].detach().cpu().tolist()
    feature_type = info["feature_type"].detach().cpu().tolist()
    feature_importance = info["feature_importance"].detach().cpu().tolist()
    target_id = int(info["target_id"].detach().cpu().item())

    pair_to_gid, gid_to_pair = build_node_id_maps(scm)

    feature_gid_to_column = {
        int(gid): column
        for column, gid in enumerate(feature_ids)
    }

    graph = nx.DiGraph()

    for gid, (layer_idx, node_idx) in gid_to_pair.items():
        graph.add_node(gid, layer=layer_idx, node=node_idx)

    for layer_idx, connection in enumerate(scm.connections):
        adjacency = connection.adj.detach().cpu()

        for parent_idx in range(connection.in_width):
            for child_idx in range(connection.out_width):
                if not bool(adjacency[parent_idx, child_idx]):
                    continue

                parent_gid = pair_to_gid[(layer_idx, parent_idx)]
                child_gid = pair_to_gid[(layer_idx + 1, child_idx)]
                graph.add_edge(parent_gid, child_gid)

    positions = {}

    x_gap = 4.0
    y_gap = 1.5

    for layer_idx, width in enumerate(scm.widths):
        for node_idx in range(width):
            gid = pair_to_gid[(layer_idx, node_idx)]
            x = layer_idx * x_gap
            y = ((width - 1) / 2 - node_idx) * y_gap
            positions[gid] = (x, y)

    ordinary_nodes = [
        gid
        for gid in graph.nodes
        if gid not in feature_gid_to_column and gid != target_id
    ]

    continuous_nodes = [
        int(gid)
        for gid in feature_ids
        if int(feature_type[feature_gid_to_column[int(gid)]]) == SCMTask.CONTINUOUS
    ]

    categorical_nodes = [
        int(gid)
        for gid in feature_ids
        if int(feature_type[feature_gid_to_column[int(gid)]]) == SCMTask.CATEGORICAL
    ]

    fig, ax = plt.subplots(figsize=(18, 10))

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        width=1.0,
        alpha=0.45,
        node_size=900,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=ordinary_nodes,
        node_size=900,
        node_shape="o",
        node_color="white",
        edgecolors="black",
        linewidths=1.3,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=continuous_nodes,
        node_size=1050,
        node_shape="o",
        node_color="0.55",
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    categorical_collection = nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=categorical_nodes,
        node_size=1050,
        node_shape="o",
        node_color="white",
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    if categorical_collection is not None:
        categorical_collection.set_hatch("///")

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[target_id],
        node_size=1700,
        node_shape="*",
        node_color="0.75",
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    node_labels = {}

    for gid in graph.nodes:
        if gid == target_id:
            node_labels[gid] = f"{gid}\nTARGET"
        elif gid in feature_gid_to_column:
            column = feature_gid_to_column[gid]
            node_labels[gid] = f"{gid}\nF{column}"
        else:
            node_labels[gid] = str(gid)

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=node_labels,
        font_size=8,
        font_weight="bold",
        ax=ax,
    )

    for layer_idx, width in enumerate(scm.widths):
        layer_y = [
            positions[pair_to_gid[(layer_idx, node_idx)]][1]
            for node_idx in range(width)
        ]

        top_y = max(layer_y) + 1.5

        if layer_idx == len(scm.widths) - 1:
            layer_name = "Target"
        elif layer_idx == 0:
            layer_name = "Root"
        else:
            layer_name = f"Layer {layer_idx}"

        ax.text(
            layer_idx * x_gap,
            top_y,
            layer_name,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    for gid in feature_ids:
        gid = int(gid)
        column = feature_gid_to_column[gid]
        importance = float(feature_importance[column])
        x, y = positions[gid]

        ax.text(
            x,
            y - 0.75,
            f"I={importance:.3f}",
            ha="center",
            va="top",
            fontsize=7,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=11,
            label="Latent node",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="0.55",
            markeredgecolor="black",
            markersize=11,
            label="Selected continuous",
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label="Selected categorical",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor="0.75",
            markeredgecolor="black",
            markersize=16,
            label="Target",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
    )

    ax.set_title(
        "Scalar Structural Causal Model",
        fontsize=15,
        fontweight="bold",
    )

    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved graph to {save_path}")


def print_selected_features(task):
    info = task.info

    feature_ids = info["feature_ids"].detach().cpu().tolist()
    feature_type = info["feature_type"].detach().cpu().tolist()
    cardinality = info["cardinality"].detach().cpu().tolist()
    importance = info["feature_importance"].detach().cpu().tolist()
    retention = info["feature_retention"].detach().cpu().tolist()
    observation_names = info["feature_observation_type_names"]

    print("\nSELECTED FEATURES")
    print("=" * 100)

    print(
        f"{'feature':>8} "
        f"{'node':>8} "
        f"{'type':>15} "
        f"{'observation':>28} "
        f"{'K':>6} "
        f"{'retention':>12} "
        f"{'importance':>12}"
    )

    for column, gid in enumerate(feature_ids):
        if int(feature_type[column]) == SCMTask.CONTINUOUS:
            kind = "continuous"
        else:
            kind = "categorical"

        if int(cardinality[column]) == 0:
            k = "-"
        else:
            k = str(int(cardinality[column]))

        print(
            f"{'F' + str(column):>8} "
            f"{int(gid):>8} "
            f"{kind:>15} "
            f"{observation_names[column]:>28} "
            f"{k:>6} "
            f"{float(retention[column]):>12.4f} "
            f"{float(importance[column]):>12.4f}"
        )


if __name__ == "__main__":
    task = SCMTask(
        num_classes=3,
        n_min=400,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_missing=0.05,
        num_roots=5,
        num_layers=3,
        final_width=1,

        connection_probs=(
            (0.25, 0.40),
            (0.55, 0.75),
        ),

        source_prior_probs=(0.55, 0.20, 0.15, 0.10),
        arity_probs=(2.5, 3.0, 3.0,),
        unary_op_probs=(1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.5, 0.75),
        binary_op_probs=(2.0, 2.0, 2.0, 1.5, 1.5),
        ternary_op_probs=(3.0, 1.0, 1.0, 3.0, 1.5),
        observation_type_probs=(7.0, 1.5, 1.5),
        latent_noise_scale=(0.0, 0.0,),
        scale_min=0.25,
        scale_max=4.0,
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04,),
        min_samples_per_category=8,
        min_component_weight=0.05,
        observation_noise_scale=0.03,
        device=torch.device("cpu"),
        dag_seed=34,
        x_seed=57,
        aleatoric_seed=67,
    )

    print_selected_features(task)

    plot_scm_graph(
        task,
        save_path="scm_graph.png",
    )
import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from src.data.scm_task_v10 import WeightedMixedScalarSCMTask


def build_node_id_maps(scm):
    pair_to_gid = {}
    gid_to_pair = {}
    gid = 0

    for layer, width in enumerate(scm.widths):
        for node in range(width):
            pair_to_gid[(layer, node)] = gid
            gid_to_pair[gid] = (layer, node)
            gid += 1

    return pair_to_gid, gid_to_pair


def plot_scm_graph(task, save_path="scm_v10_graph.png"):
    scm = task.scm
    info = task.info

    feature_ids = info["feature_ids"].detach().cpu().tolist()
    feature_type = info["feature_type"].detach().cpu().tolist()
    feature_importance = info["feature_importance"].detach().cpu().tolist()
    target_id = int(info["target_id"].detach().cpu().item())

    pair_to_gid, gid_to_pair = build_node_id_maps(scm)
    feature_gid_to_column = {int(gid): j for j, gid in enumerate(feature_ids)}

    G = nx.DiGraph()

    for gid, (layer, node) in gid_to_pair.items():
        G.add_node(gid, layer=layer, node=node)

    edge_labels = {}

    for layer, connection in enumerate(scm.connections):
        adjacency = connection.adj.detach().cpu()
        weights = connection.weights.detach().cpu()

        for parent in range(connection.in_width):
            for child in range(connection.out_width):
                if not bool(adjacency[parent, child]):
                    continue

                parent_gid = pair_to_gid[(layer, parent)]
                child_gid = pair_to_gid[(layer + 1, child)]
                weight = float(weights[parent, child].item())

                G.add_edge(parent_gid, child_gid)
                edge_labels[(parent_gid, child_gid)] = f"{weight:.2f}"

    pos = {}
    x_gap = 4.0
    y_gap = 1.5

    for layer, width in enumerate(scm.widths):
        for node in range(width):
            gid = pair_to_gid[(layer, node)]
            x = layer * x_gap
            y = ((width - 1) / 2 - node) * y_gap
            pos[gid] = (x, y)

    fig, ax = plt.subplots(figsize=(18, 10))

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=15, width=1.0, alpha=0.45, node_size=900)

    ordinary_nodes = [gid for gid in G.nodes if gid not in feature_gid_to_column and gid != target_id]
    continuous_nodes = [gid for gid in feature_ids if int(feature_type[feature_gid_to_column[int(gid)]]) == 0]
    categorical_nodes = [gid for gid in feature_ids if int(feature_type[feature_gid_to_column[int(gid)]]) == 1]

    nx.draw_networkx_nodes(G, pos, nodelist=ordinary_nodes, node_size=900, node_shape="o", node_color="white", edgecolors="black", linewidths=1.3, ax=ax)

    nx.draw_networkx_nodes(G, pos, nodelist=continuous_nodes, node_size=1050, node_shape="o", node_color="0.55", edgecolors="black", linewidths=1.5, ax=ax)

    categorical_collection = nx.draw_networkx_nodes(G, pos, nodelist=categorical_nodes, node_size=1050, node_shape="o", node_color="white", edgecolors="black", linewidths=1.5, ax=ax)
    if categorical_collection is not None:
        categorical_collection.set_hatch("///")

    nx.draw_networkx_nodes(G, pos, nodelist=[target_id], node_size=1700, node_shape="*", node_color="0.75", edgecolors="black", linewidths=1.5, ax=ax)

    labels = {}

    for gid in G.nodes:
        if gid == target_id:
            labels[gid] = f"{gid}\nTARGET"
        elif gid in feature_gid_to_column:
            j = feature_gid_to_column[gid]
            labels[gid] = f"{gid}\nF{j}"
        else:
            labels[gid] = str(gid)

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold", ax=ax)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, rotate=False, label_pos=0.5, ax=ax)

    for layer, width in enumerate(scm.widths):
        ys = [pos[pair_to_gid[(layer, node)]][1] for node in range(width)]
        top_y = max(ys) + 1.5
        name = "Target" if layer == len(scm.widths) - 1 else f"Layer {layer}"
        ax.text(layer * x_gap, top_y, name, ha="center", va="bottom", fontsize=12, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white", markeredgecolor="black", markersize=11, label="Latent node"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="0.55", markeredgecolor="black", markersize=11, label="Selected continuous"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Selected categorical"),
        Line2D([0], [0], marker="*", linestyle="None", markerfacecolor="0.75", markeredgecolor="black", markersize=16, label="Target"),
    ]

    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    ax.set_title("Weighted Mixed Scalar SCM", fontsize=15, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved to {save_path}")


def print_selected_features(task):
    info = task.info
    feature_ids = info["feature_ids"].detach().cpu().tolist()
    feature_type = info["feature_type"].detach().cpu().tolist()
    cardinality = info["cardinality"].detach().cpu().tolist()
    importance = info["feature_importance"].detach().cpu().tolist()

    print("\nSELECTED FEATURES")
    print("=" * 70)
    print(f"{'feature':>8} {'node':>8} {'type':>15} {'K':>6} {'importance':>12}")

    for j, gid in enumerate(feature_ids):
        kind = "continuous" if int(feature_type[j]) == 0 else "categorical"
        K = "-" if int(cardinality[j]) == 0 else str(int(cardinality[j]))
        print(f"{'F' + str(j):>8} {int(gid):>8} {kind:>15} {K:>6} {float(importance[j]):>12.4f}")


if __name__ == "__main__":
    task = WeightedMixedScalarSCMTask(
        num_classes=3,
        n_min=400,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_missing=0.05,
        num_roots=8,
        num_layers=5,
        hidden_width_min=6,
        hidden_width_max=10,
        final_width=1,
        connection_probs=(0.20, 0.20, 0.30, 0.85),
        edge_weight_concentration=0.30,
        latent_noise_scale=0.0,
        sampling_penalty=0.25,
        observation_noise_scale=0.03,
        observation_type_probs=(0.70, 0.15, 0.15),
        categorical_cardinalities=(2, 3, 4, 5, 6),
        categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
        min_samples_per_category=8,
        min_component_weight=0.05,
        source_prior_probs=(0.45, 0.20, 0.15, 0.05),
        linear_activation_prob=0.60,
        small_mlp_prob=0.25,
        soft_tree_prob=0.15,
        soft_tree_depth=2,
        soft_tree_temperature=0.5,
        device=torch.device("cpu"),
        dag_seed=7,
        x_seed=8,
        aleatoric_seed=9,
    )

    print_selected_features(task)
    plot_scm_graph(task, save_path="scm_v10_graph.png")
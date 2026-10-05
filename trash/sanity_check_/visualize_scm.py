# sanity_check/visualize_scm_v9.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from Trash.scm_task_v9 import WeightedMixedScalarSCMTask


BASE_SEED = 0
TABLE_ID = 0
NUM_CLASSES = 2
DEVICE = torch.device("cpu")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_KWARGS = dict(
    n_min=400,
    n_max=512,
    d_min=8,
    d_max=16,
    test_frac=0.15,
    p_missing=0.05,
    num_roots=8,
    num_layers=4,
    hidden_width_min=6,
    hidden_width_max=10,
    final_width=1,
    connection_probs=(0.20, 0.20, 0.30, 0.85),
    # min_parents_per_node=2,
    edge_weight_concentration=0.45,
    latent_noise_scale=0.0,
    observation_noise_scale=0.03,
    dominant_mass_threshold=0.70,
    dominant_feature_fraction=0.60,
    observation_type_probs=(0.70, 0.15, 0.15),
    categorical_cardinalities=(2, 3, 4, 5, 6),
    categorical_cardinality_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    min_samples_per_category=8,
    min_component_weight=0.05,
    prototype_max_attempts=8,
    prototype_min_separation=1.0,
    binning_jitter=0.20,
    source_prior_probs=(0.45, 0.20, 0.15, 0.05),
    # root_mixture_component_probs=(0.40, 0.30, 0.18, 0.08, 0.04),
    # root_mixture_separation_min=1.5,
    # root_mixture_separation_max=3.0,
    # root_mixture_scale_min=0.40,
    # root_mixture_scale_max=0.90,
    linear_activation_prob=0.60,
    small_mlp_prob=0.25,
    soft_tree_prob=0.15,
    small_mlp_hidden_dim=None,
    soft_tree_depth=2,
    soft_tree_temperature=0.5,
    device=DEVICE,
)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def global_node_id_to_layer(global_id, widths):
    start = 0
    for layer_idx, width in enumerate(widths):
        end = start + int(width)
        if start <= global_id < end:
            return layer_idx, global_id - start
        start = end
    raise ValueError(f"global_id={global_id} is outside widths={widths}")


def make_layer_positions(widths):
    positions = {}

    for layer_idx, width in enumerate(widths):
        width = int(width)

        if width == 1:
            y_values = np.array([0.5])
        else:
            y_values = np.linspace(0.08, 0.92, width)

        for node_idx in range(width):
            positions[(layer_idx, node_idx)] = (float(layer_idx), float(y_values[node_idx]))

    return positions


def get_selected_feature_lookup(feature_ids, widths):
    selected = {}

    for feature_column, global_id in enumerate(feature_ids):
        layer_idx, node_idx = global_node_id_to_layer(int(global_id), widths)
        selected[(layer_idx, node_idx)] = feature_column

    return selected


def get_feature_importance_lookup(feature_ids, feature_importance, widths):
    importance_lookup = {}

    for global_id, importance in zip(feature_ids, feature_importance):
        layer_idx, node_idx = global_node_id_to_layer(int(global_id), widths)
        importance_lookup[(layer_idx, node_idx)] = float(importance)

    return importance_lookup


def extract_edges(scm):
    edges = []

    for layer_idx, connection in enumerate(scm.connections):
        weights = to_numpy(connection.weights).astype(float)
        source_width, target_width = weights.shape

        for source_idx in range(source_width):
            for target_idx in range(target_width):
                weight = float(weights[source_idx, target_idx])

                if weight <= 0:
                    continue

                edges.append(
                    (
                        layer_idx,
                        source_idx,
                        layer_idx + 1,
                        target_idx,
                        weight,
                    )
                )

    return edges


def normalize_edge_widths(edges, min_width=0.25, max_width=4.0):
    if not edges:
        return []

    weights = np.asarray([edge[4] for edge in edges], dtype=float)
    maximum = weights.max()

    if maximum <= 0:
        return [min_width for _ in edges]

    scaled = weights / maximum
    return (min_width + scaled * (max_width - min_width)).tolist()


def draw_edges(ax, edges, positions, show_edge_weights=False, minimum_visible_weight=0.0):
    filtered_edges = [edge for edge in edges if edge[4] >= minimum_visible_weight]
    line_widths = normalize_edge_widths(filtered_edges)

    for edge, line_width in zip(filtered_edges, line_widths):
        source_layer, source_idx, target_layer, target_idx, weight = edge

        x1, y1 = positions[(source_layer, source_idx)]
        x2, y2 = positions[(target_layer, target_idx)]

        ax.annotate(
            "",
            xy=(x2 - 0.05, y2),
            xytext=(x1 + 0.05, y1),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=line_width,
                alpha=0.30,
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=1,
        )

        if show_edge_weights and weight >= 0.10:
            middle_x = (x1 + x2) / 2
            middle_y = (y1 + y2) / 2

            ax.text(
                middle_x,
                middle_y,
                f"{weight:.2f}",
                fontsize=7,
                ha="center",
                va="center",
                alpha=0.75,
                zorder=4,
            )


def draw_nodes(
    ax,
    widths,
    positions,
    selected_lookup,
    importance_lookup,
    show_importance=False,
    show_feature_labels=True,
    show_node_indices=False,
    show_distance=False,
):
    final_layer = len(widths) - 1

    for layer_idx, width in enumerate(widths):
        for node_idx in range(int(width)):
            key = (layer_idx, node_idx)
            x, y = positions[key]

            is_selected = key in selected_lookup
            is_target = layer_idx == final_layer
            importance = importance_lookup.get(key, 0.0)

            if show_importance:
                node_size = 170 + 1400 * np.sqrt(max(importance, 0.0))
            else:
                node_size = 220

            if is_target:
                node_size = max(node_size, 500)

            marker = "D" if is_target else "o"
            linewidth = 2.5 if is_selected or is_target else 1.0

            ax.scatter(
                [x],
                [y],
                s=node_size,
                marker=marker,
                facecolors="white",
                edgecolors="black",
                linewidths=linewidth,
                zorder=3,
            )

            label_lines = []

            if is_selected and show_feature_labels:
                label_lines.append(f"F{selected_lookup[key]}")

            if show_node_indices:
                label_lines.append(f"L{layer_idx}:N{node_idx}")

            if show_importance and is_selected:
                label_lines.append(f"GT={importance:.3f}")

            if show_distance:
                label_lines.append(f"d={final_layer - layer_idx}")

            if label_lines:
                ax.text(
                    x,
                    y - 0.035,
                    "\n".join(label_lines),
                    ha="center",
                    va="top",
                    fontsize=7,
                    zorder=5,
                )


def draw_layer_labels(ax, widths):
    final_layer = len(widths) - 1

    for layer_idx, width in enumerate(widths):
        if layer_idx == 0:
            title = "Root layer"
        elif layer_idx == final_layer:
            title = "Target layer"
        else:
            title = f"Latent layer {layer_idx}"

        ax.text(layer_idx, 1.02, title, ha="center", va="bottom", fontsize=10)
        ax.text(layer_idx, -0.02, f"width={width}", ha="center", va="top", fontsize=8, alpha=0.7)


def visualize_scm(
    scm,
    feature_ids,
    feature_importance=None,
    show_edge_weights=False,
    show_feature_labels=True,
    show_importance=False,
    show_node_indices=False,
    show_distance=False,
    minimum_visible_weight=0.0,
    title=None,
    save_path=None,
):
    widths = [int(width) for width in scm.widths]
    feature_ids = to_numpy(feature_ids).astype(int)

    if feature_importance is None:
        feature_importance = np.zeros(len(feature_ids), dtype=float)
    else:
        feature_importance = to_numpy(feature_importance).astype(float)

    positions = make_layer_positions(widths)
    selected_lookup = get_selected_feature_lookup(feature_ids, widths)
    importance_lookup = get_feature_importance_lookup(feature_ids, feature_importance, widths)
    edges = extract_edges(scm)

    fig_width = max(10, 2.6 * len(widths))
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    draw_edges(
        ax=ax,
        edges=edges,
        positions=positions,
        show_edge_weights=show_edge_weights,
        minimum_visible_weight=minimum_visible_weight,
    )

    draw_nodes(
        ax=ax,
        widths=widths,
        positions=positions,
        selected_lookup=selected_lookup,
        importance_lookup=importance_lookup,
        show_importance=show_importance,
        show_feature_labels=show_feature_labels,
        show_node_indices=show_node_indices,
        show_distance=show_distance,
    )

    draw_layer_labels(ax, widths)

    ax.set_xlim(-0.35, len(widths) - 0.65)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is not None:
        ax.set_title(title, pad=20)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def print_selected_features(task):
    info = task.info

    feature_ids = to_numpy(info["feature_ids"]).astype(int)
    importance = to_numpy(info["importance_ratio"]).astype(float)
    widths = to_numpy(info["layer_widths"]).astype(int).tolist()

    print()
    print("Selected features:")
    print("-" * 70)

    for feature_column, (global_id, gt) in enumerate(zip(feature_ids, importance)):
        layer_idx, node_idx = global_node_id_to_layer(int(global_id), widths)
        distance = len(widths) - 1 - layer_idx

        print(
            f"F{feature_column:02d} | "
            f"global={global_id:02d} | "
            f"L{layer_idx}:N{node_idx} | "
            f"distance={distance} | "
            f"GT={gt:.5f}"
        )


def main():
    task = WeightedMixedScalarSCMTask(
        num_classes=NUM_CLASSES,
        dag_seed=BASE_SEED + TABLE_ID,
        aleatoric_seed=100_000 + BASE_SEED + TABLE_ID,
        x_seed=200_000 + BASE_SEED + TABLE_ID,
        **TASK_KWARGS,
    )

    info = task.info
    scm = task.scm

    feature_ids = info["feature_ids"]
    feature_importance = info["importance_ratio"]

    print_selected_features(task)

    pure_dag_path = OUTPUT_DIR / f"scm_table_{TABLE_ID:03d}_dag.png"
    importance_path = OUTPUT_DIR / f"scm_table_{TABLE_ID:03d}_importance.png"
    weighted_path = OUTPUT_DIR / f"scm_table_{TABLE_ID:03d}_weighted.png"

    visualize_scm(
        scm=scm,
        feature_ids=feature_ids,
        feature_importance=feature_importance,
        show_edge_weights=False,
        show_feature_labels=True,
        show_importance=False,
        show_node_indices=False,
        show_distance=False,
        minimum_visible_weight=0.0,
        title=f"SCM DAG | table={TABLE_ID}",
        save_path=pure_dag_path,
    )

    visualize_scm(
        scm=scm,
        feature_ids=feature_ids,
        feature_importance=feature_importance,
        show_edge_weights=False,
        show_feature_labels=True,
        show_importance=True,
        show_node_indices=False,
        show_distance=True,
        minimum_visible_weight=0.0,
        title=f"SCM DAG with GT influence | table={TABLE_ID}",
        save_path=importance_path,
    )

    visualize_scm(
        scm=scm,
        feature_ids=feature_ids,
        feature_importance=feature_importance,
        show_edge_weights=True,
        show_feature_labels=True,
        show_importance=True,
        show_node_indices=True,
        show_distance=False,
        minimum_visible_weight=0.02,
        title=f"SCM weighted DAG | table={TABLE_ID}",
        save_path=weighted_path,
    )

    print()
    print("Saved:")
    print(pure_dag_path.resolve())
    print(importance_path.resolve())
    print(weighted_path.resolve())

    plt.show()


if __name__ == "__main__":
    main()
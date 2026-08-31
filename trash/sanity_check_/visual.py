# sanity_check/visualize_scm_sampling.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from Trash.scm_task_v9 import WeightedMixedScalarSCMTask


BASE_SEED = 0
TABLE_ID = 4
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
    num_layers=5,
    hidden_width_min=6,
    hidden_width_max=10,
    final_width=1,
    connection_probs=(0.20, 0.20, 0.30, 0.85),
    edge_weight_concentration=0.25,
    latent_noise_scale=0.0,
    observation_noise_scale=0.03,
    dominant_mass_threshold=0.70,
    dominant_feature_fraction=0.80,
    observation_type_probs=(0.70, 0.15, 0.15),
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


def layer_node_to_global_id(layer_idx, node_idx, widths):
    return int(sum(widths[:layer_idx]) + node_idx)


def make_layer_positions(widths):
    positions = {}
    for layer_idx, width in enumerate(widths):
        y_values = np.array([0.5]) if int(width) == 1 else np.linspace(0.06, 0.94, int(width))
        for node_idx in range(int(width)):
            positions[(layer_idx, node_idx)] = (float(layer_idx), float(y_values[node_idx]))
    return positions


def extract_edges(scm):
    edges = []
    for layer_idx, connection in enumerate(scm.connections):
        weights = to_numpy(connection.weights).astype(float)
        for source_idx in range(weights.shape[0]):
            for target_idx in range(weights.shape[1]):
                weight = float(weights[source_idx, target_idx])
                if weight > 0:
                    edges.append((layer_idx, source_idx, layer_idx + 1, target_idx, weight))
    return edges


def normalize_edge_widths(edges, min_width=0.25, max_width=3.5):
    if not edges:
        return []
    weights = np.asarray([edge[4] for edge in edges], dtype=float)
    maximum = weights.max()
    if maximum <= 0:
        return [min_width] * len(edges)
    return (min_width + (weights / maximum) * (max_width - min_width)).tolist()


def build_selected_lookup(info, widths):
    feature_ids = to_numpy(info["feature_ids"]).astype(int)
    feature_strength = to_numpy(info["feature_strength"]).astype(float)
    importance_ratio = to_numpy(info["importance_ratio"]).astype(float)
    selected_from_dominant = to_numpy(info["selected_from_dominant_group"]).astype(bool)

    lookup = {}
    for feature_column, global_id in enumerate(feature_ids):
        layer_idx, node_idx = global_node_id_to_layer(global_id, widths)
        lookup[(layer_idx, node_idx)] = {
            "feature_column": feature_column,
            "global_id": global_id,
            "node_influence": feature_strength[feature_column],
            "importance_ratio": importance_ratio[feature_column],
            "from_dominant": selected_from_dominant[feature_column],
        }
    return lookup


def build_sampling_lookup(info, widths):
    flat_sampling = to_numpy(info["all_node_influence"]).astype(float)
    lookup = {}
    for global_id, value in enumerate(flat_sampling):
        layer_idx, node_idx = global_node_id_to_layer(global_id, widths)
        lookup[(layer_idx, node_idx)] = value
    return lookup


def build_dominant_set(info):
    return set(to_numpy(info["dominant_group_ids"]).astype(int).tolist())


def draw_edges(ax, scm, positions, show_edge_weights=True):
    edges = extract_edges(scm)
    widths = normalize_edge_widths(edges)

    for edge, linewidth in zip(edges, widths):
        source_layer, source_idx, target_layer, target_idx, weight = edge

        x1, y1 = positions[(source_layer, source_idx)]
        x2, y2 = positions[(target_layer, target_idx)]

        ax.annotate(
            "",
            xy=(x2 - 0.05, y2),
            xytext=(x1 + 0.05, y1),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=linewidth,
                alpha=0.22,
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=1,
        )

        if show_edge_weights:
            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2

            ax.text(
                xm,
                ym,
                f"{weight:.2f}",
                fontsize=6,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                    pad=0.15,
                ),
                zorder=4,
            )


def draw_nodes(ax, widths, positions, sampling_lookup, selected_lookup, dominant_set):
    final_layer = len(widths) - 1

    for layer_idx, width in enumerate(widths):
        for node_idx in range(int(width)):
            key = (layer_idx, node_idx)
            global_id = layer_node_to_global_id(layer_idx, node_idx, widths)
            x, y = positions[key]

            is_target = layer_idx == final_layer
            is_selected = key in selected_lookup
            is_dominant = global_id in dominant_set
            sampling_influence = sampling_lookup.get(key, 0.0)

            if is_target:
                marker = "D"
                node_size = 500
                linewidth = 2.5
            elif is_selected and is_dominant:
                marker = "s"
                node_size = 420
                linewidth = 3.0
            elif is_selected:
                marker = "s"
                node_size = 360
                linewidth = 2.5
            elif is_dominant:
                marker = "o"
                node_size = 260
                linewidth = 2.5
            else:
                marker = "o"
                node_size = 180
                linewidth = 1.0

            ax.scatter([x], [y], s=node_size, marker=marker, facecolors="white", edgecolors="black", linewidths=linewidth, zorder=3)

            labels = [f"L{layer_idx}:N{node_idx}", f"S={sampling_influence:.4f}"]

            if is_dominant:
                labels.append("MASS")

            if is_selected:
                selected = selected_lookup[key]
                source = "DOM" if selected["from_dominant"] else "OTHER"
                labels.append(f"F{selected['feature_column']}")
                labels.append(f"G={selected['node_influence']:.4f}")
                labels.append(source)

            ax.text(x, y - 0.025, "\n".join(labels), ha="center", va="top", fontsize=6.5, zorder=5)


def draw_layer_labels(ax, widths):
    final_layer = len(widths) - 1
    for layer_idx, width in enumerate(widths):
        if layer_idx == 0:
            title = "Root"
        elif layer_idx == final_layer:
            title = "Target"
        else:
            title = f"Latent {layer_idx}"

        ax.text(layer_idx, 1.02, title, ha="center", va="bottom", fontsize=10)
        ax.text(layer_idx, -0.025, f"width={width}", ha="center", va="top", fontsize=8)


def print_sampling_debug(task):
    info = task.info
    widths = to_numpy(info["layer_widths"]).astype(int).tolist()
    sampling = to_numpy(info["all_node_influence"]).astype(float)
    dominant_set = set(to_numpy(info["dominant_group_ids"]).astype(int).tolist())
    feature_ids = to_numpy(info["feature_ids"]).astype(int)
    feature_strength = to_numpy(info["feature_strength"]).astype(float)
    importance_ratio = to_numpy(info["importance_ratio"]).astype(float)
    selected_from_dominant = to_numpy(info["selected_from_dominant_group"]).astype(bool)
    selected_lookup = {global_id: i for i, global_id in enumerate(feature_ids)}

    print()
    print("=" * 100)
    print("ALL CANDIDATE NODES")
    print("=" * 100)
    print("global | node       | sampling_inf | in_mass | selected | selected_from | node_inf | gt_ratio")
    print("-" * 100)

    num_candidates = sum(widths[:-1])

    for global_id in range(num_candidates):
        layer_idx, node_idx = global_node_id_to_layer(global_id, widths)
        in_mass = global_id in dominant_set
        is_selected = global_id in selected_lookup

        if is_selected:
            feature_column = selected_lookup[global_id]
            source = "DOMINANT" if selected_from_dominant[feature_column] else "OTHER"
            node_inf = feature_strength[feature_column]
            gt_ratio = importance_ratio[feature_column]
            selected_text = f"F{feature_column:02d}"
        else:
            source = "-"
            node_inf = np.nan
            gt_ratio = np.nan
            selected_text = "-"

        print(f"{global_id:6d} | L{layer_idx}:N{node_idx:<3d} | {sampling[global_id]:12.6f} | {str(in_mass):7s} | {selected_text:8s} | {source:13s} | {node_inf:8.5f} | {gt_ratio:8.5f}")

    print()
    print("=" * 100)
    print("DOMINANT MASS GROUP")
    print("=" * 100)

    dominant_sorted = sorted(dominant_set, key=lambda node_id: sampling[node_id], reverse=True)

    for global_id in dominant_sorted:
        layer_idx, node_idx = global_node_id_to_layer(global_id, widths)
        selected_text = "-"
        if global_id in selected_lookup:
            feature_column = selected_lookup[global_id]
            selected_text = f"F{feature_column:02d}"
        print(f"global={global_id:3d} | L{layer_idx}:N{node_idx:<3d} | sampling={sampling[global_id]:.6f} | selected={selected_text}")

    print()
    print("=" * 100)
    print("OUTSIDE DOMINANT MASS")
    print("=" * 100)

    outside = [global_id for global_id in range(num_candidates) if global_id not in dominant_set]
    outside = sorted(outside, key=lambda node_id: sampling[node_id], reverse=True)

    for global_id in outside:
        layer_idx, node_idx = global_node_id_to_layer(global_id, widths)
        selected_text = "-"
        selected_source = "-"

        if global_id in selected_lookup:
            feature_column = selected_lookup[global_id]
            selected_text = f"F{feature_column:02d}"
            selected_source = "OTHER" if not selected_from_dominant[feature_column] else "DOMINANT"

        print(f"global={global_id:3d} | L{layer_idx}:N{node_idx:<3d} | sampling={sampling[global_id]:.6f} | selected={selected_text} | source={selected_source}")

    print()
    print("=" * 100)
    print("SELECTED FEATURES")
    print("=" * 100)
    print("feature | global | node       | sampling_inf | node_inf | gt_ratio | sampled_from")
    print("-" * 100)

    for feature_column, global_id in enumerate(feature_ids):
        layer_idx, node_idx = global_node_id_to_layer(global_id, widths)
        source = "DOMINANT" if selected_from_dominant[feature_column] else "OTHER"
        print(f"F{feature_column:02d}     | {global_id:6d} | L{layer_idx}:N{node_idx:<3d} | {sampling[global_id]:12.6f} | {feature_strength[feature_column]:8.5f} | {importance_ratio[feature_column]:8.5f} | {source}")

    print()
    print(f"dominant mass size: {len(dominant_set)}")
    print(f"selected from dominant: {int(selected_from_dominant.sum())}")
    print(f"selected from other: {int((~selected_from_dominant).sum())}")
    print()


def visualize_sampling_debug(task, save_path=None):
    info = task.info
    scm = task.scm
    widths = to_numpy(info["layer_widths"]).astype(int).tolist()

    positions = make_layer_positions(widths)
    sampling_lookup = build_sampling_lookup(info, widths)
    selected_lookup = build_selected_lookup(info, widths)
    dominant_set = build_dominant_set(info)

    fig, ax = plt.subplots(figsize=(max(12, 2.8 * len(widths)), 10))

    draw_edges(ax, scm, positions)
    draw_nodes(ax, widths, positions, sampling_lookup, selected_lookup, dominant_set)
    draw_layer_labels(ax, widths)

    ax.set_xlim(-0.35, len(widths) - 0.65)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        "SCM sampling debug\n"
        "S = sampling influence | G = gradient/node influence | MASS = dominant mass | "
        "square = selected feature | DOM/OTHER = sampling source",
        pad=22,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")

    return fig, ax


def main():
    task = WeightedMixedScalarSCMTask(
        num_classes=NUM_CLASSES,
        dag_seed=BASE_SEED + TABLE_ID,
        aleatoric_seed=100_000 + BASE_SEED + TABLE_ID,
        x_seed=200_000 + BASE_SEED + TABLE_ID,
        **TASK_KWARGS,
    )

    print_sampling_debug(task)

    save_path = OUTPUT_DIR / f"scm_sampling_debug_{TABLE_ID:03d}.png"
    visualize_sampling_debug(task, save_path=save_path)

    print(f"Saved figure: {save_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
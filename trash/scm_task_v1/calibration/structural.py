import math
import torch


def _normalized_entropy(values, eps=1e-12):
    values = torch.as_tensor(values, dtype=torch.float32).clamp_min(0)
    total = values.sum()
    if total <= eps or values.numel() <= 1:
        return 0.0
    probs = values / total
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return float((entropy / math.log(values.numel())).item())


def _effective_parent_count(weights, eps=1e-12):
    weights = torch.as_tensor(weights, dtype=torch.float32).abs()
    total = weights.sum()
    if total <= eps:
        return 0.0
    probs = weights / total
    return float((1.0 / probs.square().sum().clamp_min(eps)).item())


def compute_structural_metrics(task, info):
    scm = task.scm
    feature_importance = info["feature_importance"].detach().float().cpu()
    feature_ids = info["feature_ids"].detach().long().cpu()

    sorted_importance = torch.sort(feature_importance, descending=True).values
    top1_mass = float(sorted_importance[:1].sum().item())
    top3_mass = float(sorted_importance[:3].sum().item())
    importance_entropy = _normalized_entropy(feature_importance)

    target_connection = scm.connections[-1]
    target_parents = torch.where(target_connection.adj[:, 0])[0]
    target_weights = target_connection.weights[target_parents, 0].detach().cpu()
    target_indegree = int(target_parents.numel())
    target_effective_parent_count = _effective_parent_count(target_weights)

    all_indegrees = []
    all_effective_parent_counts = []
    layer_edge_densities = []

    for connection in scm.connections:
        adjacency = connection.adj.detach().cpu()
        weights = connection.weights.detach().cpu()
        layer_edge_densities.append(float(adjacency.float().mean().item()))

        for child in range(connection.out_width):
            parents = torch.where(adjacency[:, child])[0]
            all_indegrees.append(float(parents.numel()))
            if parents.numel() == 0:
                all_effective_parent_counts.append(0.0)
            else:
                all_effective_parent_counts.append(_effective_parent_count(weights[parents, child]))

    layer_offsets = []
    offset = 0
    for width in scm.widths:
        layer_offsets.append((offset, offset + width))
        offset += width

    selected_layers = []
    for global_id in feature_ids.tolist():
        for layer_idx, (start, end) in enumerate(layer_offsets[:-1]):
            if start <= global_id < end:
                selected_layers.append(layer_idx)
                break

    selected_layer_hist = torch.bincount(torch.tensor(selected_layers, dtype=torch.long), minlength=len(scm.widths) - 1).float()
    if selected_layer_hist.sum() > 0:
        selected_layer_hist /= selected_layer_hist.sum()

    return {
        "target_indegree": float(target_indegree),
        "target_effective_parent_count": target_effective_parent_count,
        "mean_indegree": float(torch.tensor(all_indegrees).mean().item()) if all_indegrees else 0.0,
        "mean_effective_parent_count": float(torch.tensor(all_effective_parent_counts).mean().item()) if all_effective_parent_counts else 0.0,
        "mean_edge_density": float(torch.tensor(layer_edge_densities).mean().item()) if layer_edge_densities else 0.0,
        "importance_top1_mass": top1_mass,
        "importance_top3_mass": top3_mass,
        "importance_entropy": importance_entropy,
        "selected_layer_entropy": _normalized_entropy(selected_layer_hist),
    }
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


def _program_stats(node):
    """
    Compute structural statistics for one symbolic RandomMultivariateFunction program.

    Returns:
        depth:
            Maximum number of operations along any input-to-output path.

        unary_count / binary_count / ternary_count:
            Number of operations of each arity.

        input_count:
            Number of input leaves appearing in the symbolic program.

        total_ops:
            Total number of operation nodes.
    """
    kind = node[0]

    if kind == "input":
        return {"depth": 0, "unary_count": 0, "binary_count": 0, "ternary_count": 0, "input_count": 1, "total_ops": 0}

    if kind == "unary":
        child_stats = _program_stats(node[3])
        return {
            "depth": child_stats["depth"] + 1,
            "unary_count": child_stats["unary_count"] + 1,
            "binary_count": child_stats["binary_count"],
            "ternary_count": child_stats["ternary_count"],
            "input_count": child_stats["input_count"],
            "total_ops": child_stats["total_ops"] + 1,
        }

    if kind == "binary":
        left_stats = _program_stats(node[2])
        right_stats = _program_stats(node[3])

        return {
            "depth": max(left_stats["depth"], right_stats["depth"]) + 1,
            "unary_count": left_stats["unary_count"] + right_stats["unary_count"],
            "binary_count": left_stats["binary_count"] + right_stats["binary_count"] + 1,
            "ternary_count": left_stats["ternary_count"] + right_stats["ternary_count"],
            "input_count": left_stats["input_count"] + right_stats["input_count"],
            "total_ops": left_stats["total_ops"] + right_stats["total_ops"] + 1,
        }

    if kind == "ternary":
        first_stats = _program_stats(node[2])
        second_stats = _program_stats(node[3])
        third_stats = _program_stats(node[4])

        return {
            "depth": max(first_stats["depth"], second_stats["depth"], third_stats["depth"]) + 1,
            "unary_count": first_stats["unary_count"] + second_stats["unary_count"] + third_stats["unary_count"],
            "binary_count": first_stats["binary_count"] + second_stats["binary_count"] + third_stats["binary_count"],
            "ternary_count": first_stats["ternary_count"] + second_stats["ternary_count"] + third_stats["ternary_count"] + 1,
            "input_count": first_stats["input_count"] + second_stats["input_count"] + third_stats["input_count"],
            "total_ops": first_stats["total_ops"] + second_stats["total_ops"] + third_stats["total_ops"] + 1,
        }

    raise RuntimeError(f"Unknown program node kind: {kind}")


def _collect_program_stats(scm):
    stats = []

    for connection in scm.connections:
        for child_function in connection.child_functions:
            if child_function is None:
                continue

            program_stats = _program_stats(child_function.program)
            stats.append(program_stats)

    return stats


def _mean_stat(program_stats, name):
    if not program_stats:
        return 0.0

    return float(sum(float(stats[name]) for stats in program_stats) / len(program_stats))


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
    target_indegree = int(target_parents.numel())
    target_effective_parent_count = float(target_indegree)

    target_function = target_connection.child_functions[0]

    if target_function is None:
        target_program_stats = {"depth": 0, "unary_count": 0, "binary_count": 0, "ternary_count": 0, "input_count": 0, "total_ops": 0}
    else:
        target_program_stats = _program_stats(target_function.program)

    target_total_ops = int(target_program_stats["total_ops"])

    if target_total_ops > 0:
        target_unary_fraction = float(target_program_stats["unary_count"] / target_total_ops)
        target_binary_fraction = float(target_program_stats["binary_count"] / target_total_ops)
        target_ternary_fraction = float(target_program_stats["ternary_count"] / target_total_ops)
    else:
        target_unary_fraction = 0.0
        target_binary_fraction = 0.0
        target_ternary_fraction = 0.0

    all_indegrees = []
    all_effective_parent_counts = []
    layer_edge_densities = []

    for connection in scm.connections:
        adjacency = connection.adj.detach().cpu()
        layer_edge_densities.append(float(adjacency.float().mean().item()))

        for child in range(connection.out_width):
            parents = torch.where(adjacency[:, child])[0]
            indegree = float(parents.numel())
            all_indegrees.append(indegree)
            all_effective_parent_counts.append(indegree)

    program_stats = _collect_program_stats(scm)

    total_unary = sum(stats["unary_count"] for stats in program_stats)
    total_binary = sum(stats["binary_count"] for stats in program_stats)
    total_ternary = sum(stats["ternary_count"] for stats in program_stats)
    total_operations = total_unary + total_binary + total_ternary

    if total_operations > 0:
        unary_fraction = float(total_unary / total_operations)
        binary_fraction = float(total_binary / total_operations)
        ternary_fraction = float(total_ternary / total_operations)
    else:
        unary_fraction = 0.0
        binary_fraction = 0.0
        ternary_fraction = 0.0

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

        "target_program_depth": float(target_program_stats["depth"]),
        "target_program_total_ops": float(target_program_stats["total_ops"]),
        "target_program_input_count": float(target_program_stats["input_count"]),
        "target_unary_count": float(target_program_stats["unary_count"]),
        "target_binary_count": float(target_program_stats["binary_count"]),
        "target_ternary_count": float(target_program_stats["ternary_count"]),
        "target_unary_fraction": target_unary_fraction,
        "target_binary_fraction": target_binary_fraction,
        "target_ternary_fraction": target_ternary_fraction,

        "mean_program_depth": _mean_stat(program_stats, "depth"),
        "mean_program_total_ops": _mean_stat(program_stats, "total_ops"),
        "mean_program_input_count": _mean_stat(program_stats, "input_count"),
        "mean_unary_count": _mean_stat(program_stats, "unary_count"),
        "mean_binary_count": _mean_stat(program_stats, "binary_count"),
        "mean_ternary_count": _mean_stat(program_stats, "ternary_count"),

        "actual_unary_fraction": unary_fraction,
        "actual_binary_fraction": binary_fraction,
        "actual_ternary_fraction": ternary_fraction,
    }
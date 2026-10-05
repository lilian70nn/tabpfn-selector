import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ancestor_descendant_ratio(info):
    feature_ids = info["feature_ids"].detach().cpu().numpy().astype(int)
    widths = info["layer_widths"].detach().cpu().numpy().astype(int)
    adjacency = [a.detach().cpu().numpy() for a in info["adjacency_matrices"]]
    offsets = np.cumsum(np.r_[0, widths])

    selected_nodes = []
    for feature_id in feature_ids:
        layer = np.searchsorted(offsets[1:], feature_id, side="right")
        node = feature_id - offsets[layer]
        selected_nodes.append((layer, node))

    def is_ancestor(a, b):
        layer_a, node_a = a
        layer_b, node_b = b
        if layer_a >= layer_b:
            return False

        reachable = {node_a}
        for layer in range(layer_a, layer_b):
            next_reachable = set()
            for node in reachable:
                children = np.where(adjacency[layer][node])[0]
                next_reachable.update(children.tolist())
            reachable = next_reachable
            if not reachable:
                return False

        return node_b in reachable

    related = 0
    total = 0

    for i in range(len(selected_nodes)):
        for j in range(i + 1, len(selected_nodes)):
            total += 1
            if is_ancestor(selected_nodes[i], selected_nodes[j]) or is_ancestor(selected_nodes[j], selected_nodes[i]):
                related += 1

    return related / total if total > 0 else 0.0


def mean_abs_selected_feature_correlation(task):
    X = np.concatenate([
        task.X_train.detach().cpu().numpy(),
        task.X_test.detach().cpu().numpy(),
    ], axis=0)

    d = X.shape[1]

    if d < 2:
        return np.nan

    correlations = []

    for i in range(d):
        for j in range(i + 1, d):
            xi = X[:, i]
            xj = X[:, j]

            mask = np.isfinite(xi) & np.isfinite(xj)

            if mask.sum() < 5:
                continue

            xi_valid = xi[mask]
            xj_valid = xj[mask]

            if np.unique(xi_valid).size < 2 or np.unique(xj_valid).size < 2:
                continue

            rho = spearmanr(xi_valid, xj_valid).statistic

            if np.isfinite(rho):
                correlations.append(abs(rho))

    if len(correlations) == 0:
        return np.nan

    return float(np.mean(correlations))


def diagnose_scm_dataset(dataset):
    rows = []

    for i in range(len(dataset)):
        print(f"{i + 1}/{len(dataset)}")

        task = dataset[i]
        scm = task.scm
        info = task.info

        row = {
            "dataset_id": i,
            "n": int(task.X_train.shape[0] + task.X_test.shape[0]),
            "d": int(task.X_train.shape[1]),
            "ancestor_descendant_ratio": ancestor_descendant_ratio(info),
            "mean_abs_selected_feature_correlation": mean_abs_selected_feature_correlation(task),
        }

        if task.n_classes is None:
            row["task_kind"] = "regression"
            row["n_classes"] = np.nan
        else:
            row["task_kind"] = "classification"
            row["n_classes"] = int(task.n_classes)

        all_weights = []
        all_fan_in = []
        all_child_l1 = []
        all_child_l2 = []
        all_methods = []

        for layer, connection in enumerate(scm.connections):
            adjacency = connection.adj.detach().cpu().numpy()
            weights = connection.weights.detach().cpu().numpy()
            methods = connection.child_methods.detach().cpu().numpy()

            fan_in = adjacency.sum(axis=0)
            all_fan_in.extend(fan_in.tolist())

            active_weights = weights[adjacency]

            if active_weights.size > 0:
                all_weights.extend(active_weights.tolist())

            child_l1 = np.abs(weights).sum(axis=0)
            child_l2 = np.sqrt((weights ** 2).sum(axis=0))

            all_child_l1.extend(child_l1.tolist())
            all_child_l2.extend(child_l2.tolist())

            valid_methods = methods[methods >= 0]

            if valid_methods.size > 0:
                all_methods.extend(valid_methods.tolist())

            row[f"layer_{layer}_fan_in_mean"] = float(fan_in.mean())
            row[f"layer_{layer}_fan_in_median"] = float(np.median(fan_in))
            row[f"layer_{layer}_fan_in_max"] = int(fan_in.max())

            if active_weights.size > 0:
                row[f"layer_{layer}_abs_weight_mean"] = float(np.abs(active_weights).mean())
                row[f"layer_{layer}_abs_weight_median"] = float(np.median(np.abs(active_weights)))
                row[f"layer_{layer}_abs_weight_max"] = float(np.abs(active_weights).max())
                row[f"layer_{layer}_negative_weight_ratio"] = float((active_weights < 0).mean())
            else:
                row[f"layer_{layer}_abs_weight_mean"] = np.nan
                row[f"layer_{layer}_abs_weight_median"] = np.nan
                row[f"layer_{layer}_abs_weight_max"] = np.nan
                row[f"layer_{layer}_negative_weight_ratio"] = np.nan

            row[f"layer_{layer}_child_l1_mean"] = float(child_l1.mean())
            row[f"layer_{layer}_child_l2_mean"] = float(child_l2.mean())

            if valid_methods.size > 0:
                row[f"layer_{layer}_edgewise_ratio"] = float((valid_methods == 0).mean())
                row[f"layer_{layer}_post_aggregate_ratio"] = float((valid_methods == 1).mean())
                row[f"layer_{layer}_joint_mlp_ratio"] = float((valid_methods == 2).mean())
            else:
                row[f"layer_{layer}_edgewise_ratio"] = np.nan
                row[f"layer_{layer}_post_aggregate_ratio"] = np.nan
                row[f"layer_{layer}_joint_mlp_ratio"] = np.nan

        all_weights = np.asarray(all_weights, dtype=float)
        all_fan_in = np.asarray(all_fan_in, dtype=float)
        all_child_l1 = np.asarray(all_child_l1, dtype=float)
        all_child_l2 = np.asarray(all_child_l2, dtype=float)
        all_methods = np.asarray(all_methods, dtype=int)

        row["fan_in_mean"] = float(all_fan_in.mean())
        row["fan_in_median"] = float(np.median(all_fan_in))
        row["fan_in_max"] = float(all_fan_in.max())

        row["abs_weight_mean"] = float(np.abs(all_weights).mean())
        row["abs_weight_median"] = float(np.median(np.abs(all_weights)))
        row["abs_weight_max"] = float(np.abs(all_weights).max())
        row["negative_weight_ratio"] = float((all_weights < 0).mean())

        row["child_l1_mean"] = float(all_child_l1.mean())
        row["child_l1_median"] = float(np.median(all_child_l1))
        row["child_l2_mean"] = float(all_child_l2.mean())
        row["child_l2_median"] = float(np.median(all_child_l2))

        row["edgewise_ratio"] = float((all_methods == 0).mean())
        row["post_aggregate_ratio"] = float((all_methods == 1).mean())
        row["joint_mlp_ratio"] = float((all_methods == 2).mean())

        target_connection = scm.connections[-1]
        target_adjacency = target_connection.adj[:, 0].detach().cpu().numpy()
        target_weights = target_connection.weights[:, 0].detach().cpu().numpy()

        target_weights = target_weights[target_adjacency]

        row["target_fan_in"] = int(target_adjacency.sum())
        row["target_abs_weight_mean"] = float(np.abs(target_weights).mean())
        row["target_abs_weight_max"] = float(np.abs(target_weights).max())
        row["target_negative_weight_ratio"] = float((target_weights < 0).mean())
        row["target_weight_l1"] = float(np.abs(target_weights).sum())
        row["target_weight_l2"] = float(np.sqrt((target_weights ** 2).sum()))

        feature_ids = info["feature_ids"].detach().cpu().numpy().astype(int)
        widths = info["layer_widths"].detach().cpu().numpy().astype(int)
        offsets = np.cumsum(np.r_[0, widths])

        selected_layers = np.array([
            np.searchsorted(offsets[1:], feature_id, side="right")
            for feature_id in feature_ids
        ])

        for layer in range(len(widths) - 1):
            row[f"selected_layer_{layer}_ratio"] = float((selected_layers == layer).mean())

        importance = info["feature_importance"].detach().cpu().numpy()

        row["importance_mean"] = float(importance.mean())
        row["importance_median"] = float(np.median(importance))
        row["importance_max"] = float(importance.max())
        row["importance_min"] = float(importance.min())

        rows.append(row)

    return pd.DataFrame(rows)


def print_scm_diagnostics(df):
    cols = [
        "fan_in_mean",
        "fan_in_median",
        "fan_in_max",
        "target_fan_in",
        "abs_weight_mean",
        "abs_weight_median",
        "abs_weight_max",
        "negative_weight_ratio",
        "child_l1_mean",
        "child_l2_mean",
        "target_weight_l1",
        "target_weight_l2",
        "target_negative_weight_ratio",
        "edgewise_ratio",
        "post_aggregate_ratio",
        "joint_mlp_ratio",
        "ancestor_descendant_ratio",
        "mean_abs_selected_feature_correlation",
        "importance_mean",
        "importance_median",
        "importance_max",
    ]

    print()
    print("=" * 100)
    print("SCM STRUCTURE SUMMARY")
    print("=" * 100)

    print(df[cols].agg(["mean", "median", "std"]).T)

    print()
    print("=" * 100)
    print("SELECTED FEATURE LAYER DISTRIBUTION")
    print("=" * 100)

    layer_cols = [c for c in df.columns if c.startswith("selected_layer_")]
    print(df[layer_cols].mean())

    print()
    print("=" * 100)
    print("PER-LAYER FAN-IN")
    print("=" * 100)

    fanin_cols = [c for c in df.columns if c.startswith("layer_") and c.endswith("_fan_in_mean")]
    print(df[fanin_cols].mean())

    print()
    print("=" * 100)
    print("PER-LAYER WEIGHT SCALE")
    print("=" * 100)

    weight_cols = [c for c in df.columns if c.startswith("layer_") and c.endswith("_abs_weight_mean")]
    print(df[weight_cols].mean())

    print()
    print("=" * 100)
    print("PER-LAYER NEGATIVE WEIGHT RATIO")
    print("=" * 100)

    negative_cols = [c for c in df.columns if c.startswith("layer_") and c.endswith("_negative_weight_ratio")]
    print(df[negative_cols].mean())


if __name__ == "__main__":
    df = diagnose_scm_dataset(dataset)

    print_scm_diagnostics(df)

    df.to_csv("scm_structure_diagnostics.csv", index=False)
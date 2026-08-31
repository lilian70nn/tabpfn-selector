import torch
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from .task import SCMTask
from .priors import PRIOR


def _program_stats(node, depth=1):
    kind = node[0]

    if kind == "input":
        return {"max_depth": 0, "num_ops": 0, "num_unary": 0, "num_binary": 0, "num_ternary": 0, "op_counts": Counter(), "ops_by_depth": Counter()}

    if kind == "unary":
        _, op, parameter, child = node
        child_stats = _program_stats(child, depth + 1)
        return {
            "max_depth": max(depth, child_stats["max_depth"]),
            "num_ops": child_stats["num_ops"] + 1,
            "num_unary": child_stats["num_unary"] + 1,
            "num_binary": child_stats["num_binary"],
            "num_ternary": child_stats["num_ternary"],
            "op_counts": child_stats["op_counts"] + Counter({f"unary:{op}": 1}),
            "ops_by_depth": child_stats["ops_by_depth"] + Counter({depth: 1}),
        }

    if kind == "binary":
        _, op, parameter, left, right = node
        a = _program_stats(left, depth + 1)
        b = _program_stats(right, depth + 1)
        return {
            "max_depth": max(depth, a["max_depth"], b["max_depth"]),
            "num_ops": a["num_ops"] + b["num_ops"] + 1,
            "num_unary": a["num_unary"] + b["num_unary"],
            "num_binary": a["num_binary"] + b["num_binary"] + 1,
            "num_ternary": a["num_ternary"] + b["num_ternary"],
            "op_counts": a["op_counts"] + b["op_counts"] + Counter({f"binary:{op}": 1}),
            "ops_by_depth": a["ops_by_depth"] + b["ops_by_depth"] + Counter({depth: 1}),
        }

    if kind == "ternary":
        _, op, parameter, first, second, third = node
        a = _program_stats(first, depth + 1)
        b = _program_stats(second, depth + 1)
        c = _program_stats(third, depth + 1)
        return {
            "max_depth": max(depth, a["max_depth"], b["max_depth"], c["max_depth"]),
            "num_ops": a["num_ops"] + b["num_ops"] + c["num_ops"] + 1,
            "num_unary": a["num_unary"] + b["num_unary"] + c["num_unary"],
            "num_binary": a["num_binary"] + b["num_binary"],
            "num_ternary": a["num_ternary"] + b["num_ternary"] + c["num_ternary"] + 1,
            "op_counts": a["op_counts"] + b["op_counts"] + c["op_counts"] + Counter({f"ternary:{op}": 1}),
            "ops_by_depth": a["ops_by_depth"] + b["ops_by_depth"] + c["ops_by_depth"] + Counter({depth: 1}),
        }

    raise RuntimeError(f"Unknown node kind: {kind}")


def program_to_string(node):
    kind = node[0]

    if kind == "input":
        return f"x{node[1]}"

    if kind == "unary":
        _, op, parameter, child = node
        child_str = program_to_string(child)

        if op == "scale":
            return f"scale({parameter:.2f} * {child_str})"

        return f"{op}({child_str})"

    if kind == "binary":
        _, op, parameter, left, right = node
        a = program_to_string(left)
        b = program_to_string(right)

        if op == "add":
            return f"({a} + {b})"
        if op == "sub":
            return f"({a} - {b})"
        if op == "mul":
            return f"({a} * {b})"
        if op == "safe_div":
            return f"safe_div({a}, {b})"
        if op == "activated_affine":
            weight, bias, activation = parameter
            return f"{activation}({weight[0].item():.2f}*{a} + {weight[1].item():.2f}*{b} + {bias.item():.2f})"

        return f"{op}({a}, {b})"

    if kind == "ternary":
        _, op, parameter, first, second, third = node
        a = program_to_string(first)
        b = program_to_string(second)
        c = program_to_string(third)

        if op == "sum3":
            return f"({a} + {b} + {c})"
        if op == "mul_add":
            return f"({a} * {b} + {c})"
        if op == "mul_sub":
            return f"({a} * {b} - {c})"
        if op == "gated_mix":
            return f"gated_mix({a}, {b}, {c})"
        if op == "activated_affine":
            weight, bias, activation = parameter
            return f"{activation}({weight[0].item():.2f}*{a} + {weight[1].item():.2f}*{b} + {weight[2].item():.2f}*{c} + {bias.item():.2f})"

        return f"{op}({a}, {b}, {c})"

    raise RuntimeError(f"Unknown node kind: {kind}")


def _target_stats(task, target_idx=0):
    scm = task.scm
    connection = scm.connections[-1]
    parents = torch.where(connection.adj[:, target_idx])[0]
    num_parents = int(parents.numel())
    child_function = connection.child_functions[target_idx]

    if child_function is None:
        return {
            "num_parents": num_parents,
            "program_depth": 0,
            "num_ops": 0,
            "num_unary": 0,
            "num_binary": 0,
            "num_ternary": 0,
            "op_counts": Counter(),
            "ops_by_depth": Counter(),
            "program": "SOURCE",
        }

    stats = _program_stats(child_function.program, depth=1)
    return {
        "num_parents": num_parents,
        "program_depth": stats["max_depth"],
        "num_ops": stats["num_ops"],
        "num_unary": stats["num_unary"],
        "num_binary": stats["num_binary"],
        "num_ternary": stats["num_ternary"],
        "op_counts": stats["op_counts"],
        "ops_by_depth": stats["ops_by_depth"],
        "program": program_to_string(child_function.program),
    }


def analyze_programs(prior, n_tasks=1000, base_seed=0, n_examples=30):
    rows = []
    examples = []
    total_ops = Counter()
    total_ops_by_depth = Counter()

    for idx in tqdm(range(int(n_tasks)), desc="Analyzing programs", unit="task"):
        task = SCMTask(**prior, dag_seed=base_seed + idx * 3, x_seed=base_seed + idx * 3 + 1, aleatoric_seed=base_seed + idx * 3 + 2)
        stats = _target_stats(task)

        rows.append({
            "task_idx": idx,
            "num_parents": stats["num_parents"],
            "program_depth": stats["program_depth"],
            "num_ops": stats["num_ops"],
            "num_unary": stats["num_unary"],
            "num_binary": stats["num_binary"],
            "num_ternary": stats["num_ternary"],
            "program": stats["program"],
        })

        if len(examples) < n_examples:
            examples.append({
                "task_idx": idx,
                "num_parents": stats["num_parents"],
                "program_depth": stats["program_depth"],
                "num_ops": stats["num_ops"],
                "program": stats["program"],
            })

        total_ops.update(stats["op_counts"])
        total_ops_by_depth.update(stats["ops_by_depth"])

    df = pd.DataFrame(rows)

    summary = pd.DataFrame({
        "metric": ["num_parents", "program_depth", "num_ops", "num_unary", "num_binary", "num_ternary"],
        "mean": [df[c].mean() for c in ["num_parents", "program_depth", "num_ops", "num_unary", "num_binary", "num_ternary"]],
        "median": [df[c].median() for c in ["num_parents", "program_depth", "num_ops", "num_unary", "num_binary", "num_ternary"]],
        "min": [df[c].min() for c in ["num_parents", "program_depth", "num_ops", "num_unary", "num_binary", "num_ternary"]],
        "max": [df[c].max() for c in ["num_parents", "program_depth", "num_ops", "num_unary", "num_binary", "num_ternary"]],
    })

    parent_depth = df.groupby("num_parents").agg(
        count=("program_depth", "size"),
        mean_depth=("program_depth", "mean"),
        median_depth=("program_depth", "median"),
        mean_ops=("num_ops", "mean"),
        mean_unary=("num_unary", "mean"),
        mean_binary=("num_binary", "mean"),
        mean_ternary=("num_ternary", "mean"),
    ).reset_index()

    op_df = pd.DataFrame(total_ops.items(), columns=["operator", "count"]).sort_values("count", ascending=False)
    if len(op_df):
        op_df["ratio"] = op_df["count"] / op_df["count"].sum()

    depth_df = pd.DataFrame(sorted(total_ops_by_depth.items()), columns=["depth", "num_ops"])
    example_df = pd.DataFrame(examples)

    return df, summary, parent_depth, op_df, depth_df, example_df




if __name__ == "__main__":
    raw, summary, parent_depth, operators, depth_distribution, examples = analyze_programs(PRIOR, n_tasks=1000, base_seed=0, n_examples=30)

    print("\n=== Overall summary ===")
    print(summary.to_string(index=False))

    print("\n=== Parent count -> program complexity ===")
    print(parent_depth.to_string(index=False))

    print("\n=== Operator frequencies ===")
    print(operators.to_string(index=False))

    print("\n=== Operators by program depth ===")
    print(depth_distribution.to_string(index=False))

    print("\n=== Example target programs ===")
    for _, row in examples.iterrows():
        print(f"\n[{int(row['task_idx'])}] parents={int(row['num_parents'])} depth={int(row['program_depth'])} ops={int(row['num_ops'])}")
        print(row["program"])

    raw.to_csv("program_raw.csv", index=False)
    summary.to_csv("program_summary.csv", index=False)
    parent_depth.to_csv("program_parent_depth.csv", index=False)
    operators.to_csv("program_operators.csv", index=False)
    depth_distribution.to_csv("program_depth_distribution.csv", index=False)
    examples.to_csv("program_examples.csv", index=False)








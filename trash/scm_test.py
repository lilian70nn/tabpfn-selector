from unittest import loader


from src.data.datasets import SyntheticTaskDataset

import torch
from torch.utils.data import DataLoader
from src.data.collate import collate_tasks

from torch.utils.data import DataLoader
from src.model.tabpfn_v2 import TabularPFNModel
from src.data.linear_task import MixedLinearTask
from functools import partial


# def main():
#     device = torch.device("cpu")

#     dataset = SyntheticTaskDataset(
#         task_factory=MixedSCMTask,
#         length=8,
#         task_kind="classification",
#         min_classes=2,
#         max_classes=6,
#         task_kwargs={
#             "n_min": 128,
#             "n_max": 256,
#             "d_min": 2,
#             "d_max": 10,
#             "test_frac": 0.15,
#             "p_missing": 0.05,
#             "node_noise_scale": 0.05,
#             "device": device,
#             "num_roots": 3,
#             "num_layers": 4,
#             "max_nodes_per_layer": 5,
#             "edge_prob": 0.35,
#             "p_cat": 0.3,
#             "max_cardinality": 4,
#             "min_parents_per_node": 1,
#             "num_bins": 5,
#         },
#         base_seed=123,
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size=4,
#         shuffle=False,
#         collate_fn=collate_tasks,
#     )

#     batch = next(iter(loader))
#     print("Batch X_train shape:", batch.X_train.shape)
#     print("Batch y_train shape:", batch.y_train.shape)
#     print("Batch X_test shape:", batch.X_test.shape)
#     print("Batch y_test shape:", batch.y_test.shape)
#     print("Batch Ntr_max:", batch.Ntr_max)
#     print("Batch Nte_max:", batch.Nte_max)
#     print("Batch d_max:", batch.d_max) 
#     print("Batch n_train:", batch.n_train)
#     print("Batch n_test:", batch.n_test)
#     print("Batch d_emb:", batch.d_emb)
#     print("Batch feature_type:", batch.feature_type)
#     print("Batch cardinality:", batch.cardinality)
#     print("Batch is_active:", batch.is_active)
#     print("Batch importance_ratio:", batch.importance_ratio)
#     print("Batch feature_strength:", batch.feature_strength)
#     print("Batch cell_mask:", batch.cell_mask)
#     print("Batch x_mean:", batch.x_mean)
#     print("Batch x_std:", batch.x_std)
#     print("Batch y_mean:", batch.y_mean)
#     print("Batch y_std:", batch.y_std)
#     print("Batch n_classes:", batch.n_classes)


# def main():


#     OPENML_DATASETS = {
#         "diabetes": 46921,
#     }

#     loader = DataLoader(
#         list(OPENML_DATASETS.items()),
#         batch_size=1,
#         shuffle=False,
#         collate_fn=collate_openml_task,
#     )

#     batch = next(iter(loader))

#     print(batch.X_train.shape)
#     print(batch.y_train.shape)
#     print(batch.X_test.shape)
#     print(batch.y_test.shape)
#     print(batch.feature_type)
#     print(batch.cardinality)
#     print(batch.cell_mask.shape)
#     print(batch.n_classes)
#     print(batch.y_mean, batch.y_std)


#     model = TabularPFNModel(
#         k=64,
#         m=64,
#         n_heads=4,
#         depth=4,
#         max_cardinality=15,
#         task_kind="classification",
#         max_classes=5
#     ).to(batch.X_train.device)

#     model.eval()

#     with torch.no_grad():
#         out = model(batch)

#     print("logits:", out["logits"].shape)

#     if out.get("importance_logits") is not None:
#         print("importance_logits:", out["importance_logits"].shape)
#     else:
#         print("importance_logits: None")


# def main():
#     model =  MixedLinearTask(
#         num_classes=3,
#         n_min=128,
#         n_max=256,
#         d_min=2,
#         d_max=10,
#         test_frac=0.15,
#         p_missing=0.05,
#         p_categorical=0.3,
#         max_cardinality=4,
#         p_active=0.5,
#         noise_level=0.1,
#         device=torch.device("cpu"),
#         dag_seed=123,
#         aleatoric_seed=456,
#         x_seed=789,
#     )

#     print("Task X_train shape:", model.X_train.shape)
#     print("Task y_train shape:", model.y_train.shape)
#     print("Task X_test shape:", model.X_test.shape)
#     print("Task y_test shape:", model.y_test.shape)
#     print("Task info:", model.info)

# def main():

#     train_dataset = SyntheticTaskDataset(
#         length=10,
#         task_factory=MixedLinearTask,
#         task_kind="classification",
#         min_classes=2,
#         max_classes=6,
#         base_seed=0,
#         task_kwargs=dict(
#             n_min=256,
#             n_max=512,
#             d_min=8,
#             d_max=12,
#             test_frac=0.15,
#             p_categorical=0.4,
#             max_cardinality=5,
#             p_active=0.8,
#             p_missing=0.1,
#             noise_level=0.1,
#             device="cpu",
#         ),
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=8,
#         shuffle=True,
#         num_workers=2,
#         pin_memory=True,
#         collate_fn=partial(collate_tasks, use_selector=True),
#     )

#     batch = next(iter(train_loader))
#     # print("Batch X_train shape:", batch.X_train.shape)
#     # print("Batch y_train shape:", batch.y_train.shape)
#     # print("Batch X_test shape:", batch.X_test.shape)
#     # print("Batch y_test shape:", batch.y_test.shape)
#     # print("Batch Ntr_max:", batch.Ntr_max)
#     # print("Batch Nte_max:", batch.Nte_max)
#     # print("Batch d_max:", batch.d_max) 
#     # print("Batch n_train:", batch.n_train)
#     # print("Batch n_test:", batch.n_test)
#     # print("Batch d_emb:", batch.d_emb)
#     # print("Batch feature_type:", batch.feature_type)
#     # print("Batch cardinality:", batch.cardinality)
#     # print("Batch is_active:", batch.is_active)
#     # print("Batch importance_ratio:", batch.importance_ratio)
#     # print("Batch feature_strength:", batch.feature_strength)
#     # print("Batch cell_mask:", batch.cell_mask)
#     # print("Batch x_mean:", batch.x_mean)
#     # print("Batch x_std:", batch.x_std)
#     # print("Batch y_mean:", batch.y_mean)
#     # print("Batch y_std:", batch.y_std)
#     print("Batch n_classes:", batch.n_classes)
#     for i in range(8):
#         print(f"Batch y_train[{i}, :]:", batch.y_train[i, :])
#         print(f"Batch y_test[{i}, :]:", batch.y_test[i, :])

# def main():

#     dataset = SyntheticTaskDataset(
#         length=1000,
#         task_factory=MixedSCMTask,
#         task_kind="classification",
#         min_classes=2,
#         max_classes=6,
#         base_seed=0,
#         task_kwargs=dict(
#             n_min=256,
#             n_max=512,
#             d_min=8,
#             d_max=16,
#             test_frac=0.15,
#             p_cat=0.3,
#             max_cardinality=5,
#             p_missing=0.1,
#             node_noise_scale=0.05,
#             num_roots=3,
#             num_layers=4,
#             max_nodes_per_layer=8,
#             edge_prob=0.35,
#             min_parents_per_node=1,
#             num_bins=5,
#             device=torch.device("cpu"),
#         ),
#     )




#     # print("Task X_train shape:", data.X_train.shape)
#     # print("Task y_train shape:", data.y_train.shape)
#     # print("Task X_test shape:", data.X_test.shape)
#     # print("Task y_test shape:", data.y_test.shape)
#     # print("Task info:", data.info)

#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=4,
#         shuffle=False,
#         collate_fn=partial(collate_tasks, use_selector=True),
#     )

#     batch = next(iter(loader))
#     print("Batch X_train shape:", batch.X_train.shape)
#     print("Batch y_train shape:", batch.y_train.shape)
#     print("Batch X_test shape:", batch.X_test.shape)
#     print("Batch y_test shape:", batch.y_test.shape)


#     model = TabularPFNModel(
#         k=72,
#         m=256,
#         n_heads=6,
#         depth=16,
#         max_cardinality=5,
#         task_kind="classification",
#         max_classes=6,
#     )

#     results = model(batch)

#     print("Logits shape:", results["logits"].shape)
#     print("Importance logits shape:", results["importance_logits"].shape)




# import torch
# from collections import Counter, defaultdict

# from data.scm_generator import MixedSCMTask


# def flatten_specs(scm):
#     flat_specs = []
#     flat_index = []

#     for l, specs in enumerate(scm.layers):
#         for j, spec in enumerate(specs):
#             flat_specs.append(spec)
#             flat_index.append((l, j))

#     return flat_specs, flat_index


# def inspect_one_task(task_id: int = 0, device="cpu"):
#     task = MixedSCMTask(
#         num_classes=3,
#         n_max=512,
#         d_max=20,
#         n_min=128,
#         d_min=2,
#         test_frac=0.15,
#         p_missing=0.05,
#         node_noise_scale=0.05,
#         device=torch.device(device),
#         num_roots=3,
#         num_layers=4,
#         max_nodes_per_layer=8,
#         edge_prob=0.35,
#         p_cat=0.3,
#         max_cardinality=10,
#         min_parents_per_node=1,
#         num_bins=5,
#     )

#     X_train, y_train, X_test, y_test, info = task._generate()

#     scm = task.scm
#     flat_specs, flat_index = flatten_specs(scm)

#     feature_ids = info["feature_ids"].detach().cpu().tolist()
#     target_id = int(info["target_id"].detach().cpu().item())

#     feature_strength = info["feature_strength"].detach().cpu()
#     importance_ratio = info["importance_ratio"].detach().cpu()

#     target_layer, target_node = flat_index[target_id]
#     target_spec = flat_specs[target_id]

#     print("=" * 80)
#     print(f"TASK {task_id}")
#     print(f"n_train={X_train.shape[0]}, n_test={X_test.shape[0]}, d={len(feature_ids)}")
#     print(f"SCM widths: {scm.widths}")
#     print(f"target_id={target_id}, target=(layer={target_layer}, node={target_node}), kind={target_spec.kind}")
#     print()

#     counts = Counter()
#     imp_sum = defaultdict(float)
#     strength_sum = defaultdict(float)

#     rows = []

#     for j, fid in enumerate(feature_ids):
#         flayer, fnode = flat_index[fid]
#         fspec = flat_specs[fid]

#         if flayer < target_layer:
#             relation = "before"
#         elif flayer == target_layer:
#             relation = "same"
#         else:
#             relation = "after"

#         strength = float(feature_strength[j].item())
#         ratio = float(importance_ratio[j].item())

#         counts[relation] += 1
#         imp_sum[relation] += ratio
#         strength_sum[relation] += strength

#         rows.append(
#             {
#                 "j": j,
#                 "feature_id": fid,
#                 "layer": flayer,
#                 "node": fnode,
#                 "kind": fspec.kind,
#                 "K": fspec.K,
#                 "relation": relation,
#                 "strength": strength,
#                 "importance_ratio": ratio,
#             }
#         )

#     print("Summary by relation:")
#     for rel in ["before", "same", "after"]:
#         print(
#             f"  {rel:6s} | count={counts[rel]:2d} "
#             f"| imp_sum={imp_sum[rel]:.6f} "
#             f"| strength_sum={strength_sum[rel]:.6f}"
#         )

#     print()
#     print("Per feature:")
#     print(
#         f"{'j':>3s} {'fid':>4s} {'layer':>5s} {'node':>4s} "
#         f"{'kind':>5s} {'K':>4s} {'rel':>6s} "
#         f"{'strength':>12s} {'imp_ratio':>12s}"
#     )

#     for r in rows:
#         K = "-" if r["K"] is None else str(r["K"])
#         print(
#             f"{r['j']:3d} {r['feature_id']:4d} {r['layer']:5d} {r['node']:4d} "
#             f"{r['kind']:>5s} {K:>4s} {r['relation']:>6s} "
#             f"{r['strength']:12.6f} {r['importance_ratio']:12.6f}"
#         )

#     print()
#     return rows


# def inspect_many(num_tasks: int = 100, device="cpu"):
#     total_counts = Counter()
#     total_imp_sum = defaultdict(float)
#     total_strength_sum = defaultdict(float)
#     total_features = 0

#     for t in range(num_tasks):
#         task = MixedSCMTask(
#             num_classes=3,
#             n_max=512,
#             d_max=20,
#             n_min=128,
#             d_min=2,
#             test_frac=0.15,
#             p_missing=0.05,
#             node_noise_scale=0.05,
#             device=torch.device(device),
#             num_roots=5,
#             num_layers=8,
#             max_nodes_per_layer=8,
#             edge_prob=0.35,
#             p_cat=0.3,
#             max_cardinality=10,
#             min_parents_per_node=1,
#             num_bins=5,
#         )

#         _, _, _, _, info = task._generate()

#         flat_specs, flat_index = flatten_specs(task.scm)

#         feature_ids = info["feature_ids"].detach().cpu().tolist()
#         target_id = int(info["target_id"].detach().cpu().item())
#         target_layer, _ = flat_index[target_id]

#         feature_strength = info["feature_strength"].detach().cpu()
#         importance_ratio = info["importance_ratio"].detach().cpu()

#         for j, fid in enumerate(feature_ids):
#             flayer, _ = flat_index[fid]

#             if flayer < target_layer:
#                 relation = "before"
#             elif flayer == target_layer:
#                 relation = "same"
#             else:
#                 relation = "after"

#             total_counts[relation] += 1
#             total_imp_sum[relation] += float(importance_ratio[j].item())
#             total_strength_sum[relation] += float(feature_strength[j].item())
#             total_features += 1

#     print("=" * 80)
#     print(f"Across {num_tasks} tasks")
#     print(f"total_features={total_features}")
#     print()

#     for rel in ["before", "same", "after"]:
#         count = total_counts[rel]
#         frac = count / max(total_features, 1)
#         avg_imp = total_imp_sum[rel] / max(num_tasks, 1)
#         avg_strength = total_strength_sum[rel] / max(count, 1)

#         print(
#             f"{rel:6s} | count={count:5d} "
#             f"| frac={frac:.4f} "
#             f"| avg_task_imp_sum={avg_imp:.6f} "
#             f"| avg_feature_strength={avg_strength:.6f}"
#         )


# if __name__ == "__main__":
#     inspect_one_task(task_id=0, device="cpu")
#     inspect_many(num_tasks=100, device="cpu")



# if __name__ == "__main__":
#     main()

import torch
import matplotlib.pyplot as plt
import networkx as nx

from src.data.scm_task import MixedSCMTask


def build_node_id_maps(scm):
    pair_to_gid = {}
    gid_to_pair = {}

    gid = 0
    for layer_id, specs in enumerate(scm.layers):
        for node_id, _ in enumerate(specs):
            pair_to_gid[(layer_id, node_id)] = gid
            gid_to_pair[gid] = (layer_id, node_id)
            gid += 1

    return pair_to_gid, gid_to_pair


def print_feature_table(scm, info):
    feature_ids = info["feature_ids"].detach().cpu().tolist()
    target_id = int(info["target_id"].detach().cpu().item())

    feature_strength = info["feature_strength"].detach().cpu()
    importance_ratio = info["importance_ratio"].detach().cpu()
    feature_type = info["feature_type"].detach().cpu()
    cardinality = info["cardinality"].detach().cpu()

    _, gid_to_pair = build_node_id_maps(scm)

    target_layer, target_local = gid_to_pair[target_id]

    print()
    print("=" * 80)
    print("FEATURE TABLE")
    print("=" * 80)
    print(f"target_id={target_id}, target_local=({target_layer},{target_local})")
    print()

    print(
        f"{'j':>3s} "
        f"{'gid':>4s} "
        f"{'local':>10s} "
        f"{'kind':>5s} "
        f"{'K':>4s} "
        f"{'rel':>8s} "
        f"{'strength':>12s} "
        f"{'imp_ratio':>12s}"
    )

    for j, gid in enumerate(feature_ids):
        layer_id, local_node_id = gid_to_pair[int(gid)]

        if layer_id < target_layer:
            relation = "before"
        elif layer_id == target_layer:
            relation = "same"
        else:
            relation = "after"

        kind = "cont" if int(feature_type[j].item()) == 0 else "cat"
        K = "-" if int(cardinality[j].item()) == 0 else str(int(cardinality[j].item()))

        print(
            f"{j:3d} "
            f"{gid:4d} "
            f"{str((layer_id, local_node_id)):>10s} "
            f"{kind:>5s} "
            f"{K:>4s} "
            f"{relation:>8s} "
            f"{float(feature_strength[j].item()):12.6f} "
            f"{float(importance_ratio[j].item()):12.6f}"
        )


def print_graph_with_ids(scm, info):
    feature_ids = info["feature_ids"].detach().cpu().tolist()
    target_id = int(info["target_id"].detach().cpu().item())
    feature_set = set(int(x) for x in feature_ids)

    pair_to_gid, gid_to_pair = build_node_id_maps(scm)

    print()
    print("=" * 80)
    print("FULL SCM GRAPH WITH GLOBAL NODE IDS")
    print("=" * 80)
    print("Global node ids are assigned layer by layer:")
    print("layer 0 first, then layer 1, etc.")
    print()

    print("widths:", scm.widths)
    print("num_nodes:", len(gid_to_pair))
    print("feature_ids:", feature_ids)
    print("target_id:", target_id)

    print()
    print("-" * 80)
    print("NODES")
    print("-" * 80)

    for layer_id, specs in enumerate(scm.layers):
        print(f"Layer {layer_id}:")

        for local_node_id, spec in enumerate(specs):
            gid = pair_to_gid[(layer_id, local_node_id)]

            tags = []
            if gid in feature_set:
                tags.append("FEATURE")
            if gid == target_id:
                tags.append("TARGET")

            tag_str = ""
            if tags:
                tag_str = " [" + ", ".join(tags) + "]"

            if spec.kind == "cont":
                print(
                    f"  id={gid:3d} | local=({layer_id},{local_node_id}) "
                    f"| kind=cont{tag_str}"
                )
            else:
                print(
                    f"  id={gid:3d} | local=({layer_id},{local_node_id}) "
                    f"| kind=cat | K={spec.K}{tag_str}"
                )

        print()

    print("-" * 80)
    print("EDGES")
    print("-" * 80)

    for layer_id, conn in enumerate(scm.connections):
        print(f"Layer {layer_id} -> Layer {layer_id + 1}:")

        edge_count = 0

        for parent_local_id in range(conn.in_width):
            for child_local_id in range(conn.out_width):
                edge = conn.edges[parent_local_id][child_local_id]

                if edge is None:
                    continue

                parent_gid = pair_to_gid[(layer_id, parent_local_id)]
                child_gid = pair_to_gid[(layer_id + 1, child_local_id)]

                edge_name = edge.name() if hasattr(edge, "name") else edge.__class__.__name__

                print(
                    f"  {parent_gid:3d} ({layer_id},{parent_local_id}) "
                    f"-> {child_gid:3d} ({layer_id + 1},{child_local_id}) "
                    f"| {edge.__class__.__name__}: {edge_name}"
                )

                edge_count += 1

        print(f"  num_edges = {edge_count}")
        print()


def plot_scm_graph(scm, info, save_path="scm_graph.png"):
    feature_ids = info["feature_ids"].detach().cpu().tolist()
    target_id = int(info["target_id"].detach().cpu().item())
    feature_set = set(int(x) for x in feature_ids)

    pair_to_gid, gid_to_pair = build_node_id_maps(scm)

    G = nx.DiGraph()

    for gid, (layer_id, local_node_id) in gid_to_pair.items():
        spec = scm.layers[layer_id][local_node_id]

        if gid == target_id:
            node_label = f"{gid}\nTARGET"
        elif gid in feature_set:
            node_label = f"{gid}\nF"
        else:
            node_label = str(gid)

        G.add_node(
            gid,
            layer=layer_id,
            local=local_node_id,
            kind=spec.kind,
            label=node_label,
        )

    edge_labels = {}

    for layer_id, conn in enumerate(scm.connections):
        for parent_local_id in range(conn.in_width):
            for child_local_id in range(conn.out_width):
                edge = conn.edges[parent_local_id][child_local_id]
                if edge is None:
                    continue

                parent_gid = pair_to_gid[(layer_id, parent_local_id)]
                child_gid = pair_to_gid[(layer_id + 1, child_local_id)]

                edge_name = edge.name() if hasattr(edge, "name") else edge.__class__.__name__

                G.add_edge(parent_gid, child_gid)
                edge_labels[(parent_gid, child_gid)] = edge_name

    pos = {}
    x_gap = 4.0
    y_gap = 1.4

    for layer_id, specs in enumerate(scm.layers):
        width = len(specs)
        for local_node_id, _ in enumerate(specs):
            gid = pair_to_gid[(layer_id, local_node_id)]
            x = layer_id * x_gap
            y = (width - 1) / 2 - local_node_id
            pos[gid] = (x, y * y_gap)

    labels = nx.get_node_attributes(G, "label")

    target_nodes = [target_id]
    feature_nodes = [gid for gid in G.nodes if gid in feature_set and gid != target_id]
    other_nodes = [gid for gid in G.nodes if gid not in feature_set and gid != target_id]

    plt.figure(figsize=(18, 10))

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        width=1.0,
        alpha=0.45,
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=other_nodes,
        node_size=900,
        node_shape="o",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=feature_nodes,
        node_size=1100,
        node_shape="s",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=target_nodes,
        node_size=1600,
        node_shape="*",
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=9,
        font_weight="bold",
    )

    if G.number_of_edges() <= 60:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=6,
            rotate=False,
        )

    plt.title("RandomLayeredSCM graph\nsquare = selected feature, star = target")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"saved graph to {save_path}")


def main():
    task = MixedSCMTask(
        num_classes=3,
        n_min=400,
        n_max=512,
        d_min=8,
        d_max=16,
        test_frac=0.15,
        p_missing=0.05,
        node_noise_scale=0.05,
        device=torch.device("cpu"),
        dag_seed=1,
        x_seed=2,
        aleatoric_seed=3,
        num_roots=5,
        num_layers=4,
        max_nodes_per_layer=12,
        edge_prob=0.3,
        p_cat=0.3,
        max_cardinality=10,
        min_parents_per_node=1,
        num_bins=5,
    )

    X_train, y_train, X_test, y_test, info = (
        task.X_train,
        task.y_train,
        task.X_test,
        task.y_test,
        task.info,
    )

    print("=" * 80)
    print("BASIC SHAPES")
    print("=" * 80)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)
    print("n_features:", task.n_features)
    print("n_classes:", task.n_classes)

    print()
    print("=" * 80)
    print("FEATURE METADATA")
    print("=" * 80)
    print("feature_type:", info["feature_type"])
    print("  0 = continuous, 1 = categorical")
    print("cardinality :", info["cardinality"])
    print("feature_ids :", info["feature_ids"])
    print("target_id   :", info["target_id"])

    print()
    print("=" * 80)
    print("IMPORTANCE")
    print("=" * 80)
    print("feature_strength :", info["feature_strength"])
    print("importance_ratio :", info["importance_ratio"])
    print("importance sum   :", info["importance_ratio"].sum().item())
    print("is_active        :", info["is_active"])

    print()
    print("=" * 80)
    print("MISSING")
    print("=" * 80)
    print("train missing ratio:", torch.isnan(X_train).float().mean().item())
    print("test missing ratio :", torch.isnan(X_test).float().mean().item())

    print()
    print("=" * 80)
    print("Y DISTRIBUTION")
    print("=" * 80)
    print("y_train unique:", torch.unique(y_train, return_counts=True))
    print("y_test unique :", torch.unique(y_test, return_counts=True))

    print()
    print("=" * 80)
    print("FIRST 5 ROWS")
    print("=" * 80)
    print("X_train[:5]:")
    print(X_train[:5])
    print("y_train[:5]:")
    print(y_train[:5])

    print()
    print("=" * 80)
    print("SCM GRAPH SUMMARY")
    print("=" * 80)
    print("widths:", task.scm.widths)

    for layer_id, specs in enumerate(task.scm.layers):
        desc = []
        for spec in specs:
            if spec.kind == "cont":
                desc.append("cont")
            else:
                desc.append(f"cat(K={spec.K})")
        print(f"layer {layer_id}:", desc)

    print_feature_table(task.scm, info)
    print_graph_with_ids(task.scm, info)

    plot_scm_graph(
        task.scm,
        info,
        save_path="scm_graph.png",
    )


if __name__ == "__main__":
    main()
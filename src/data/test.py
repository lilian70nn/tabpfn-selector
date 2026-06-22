from src.data.scm_task import MixedSCMTask
from src.data.datasets import SyntheticTaskDataset

import torch
from torch.utils.data import DataLoader
from src.data.collate import collate_tasks


def main():
    device = torch.device("cpu")

    dataset = SyntheticTaskDataset(
        task_factory=MixedSCMTask,
        length=8,
        task_kind="classification",
        min_classes=2,
        max_classes=6,
        task_kwargs={
            "n_min": 128,
            "n_max": 256,
            "d_min": 2,
            "d_max": 10,
            "test_frac": 0.15,
            "p_missing": 0.05,
            "node_noise_scale": 0.05,
            "device": device,
            "num_roots": 3,
            "num_layers": 4,
            "max_nodes_per_layer": 5,
            "edge_prob": 0.35,
            "p_cat": 0.3,
            "max_cardinality": 4,
            "min_parents_per_node": 1,
            "num_bins": 5,
        },
        base_seed=123,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_tasks,
    )

    batch = next(iter(loader))
    print("Batch X_train shape:", batch.X_train.shape)
    print("Batch y_train shape:", batch.y_train.shape)
    print("Batch X_test shape:", batch.X_test.shape)
    print("Batch y_test shape:", batch.y_test.shape)
    print("Batch Ntr_max:", batch.Ntr_max)
    print("Batch Nte_max:", batch.Nte_max)
    print("Batch d_max:", batch.d_max) 
    print("Batch n_train:", batch.n_train)
    print("Batch n_test:", batch.n_test)
    print("Batch d_emb:", batch.d_emb)
    print("Batch feature_type:", batch.feature_type)
    print("Batch cardinality:", batch.cardinality)
    print("Batch is_active:", batch.is_active)
    print("Batch importance_ratio:", batch.importance_ratio)
    print("Batch feature_strength:", batch.feature_strength)
    print("Batch cell_mask:", batch.cell_mask)
    print("Batch x_mean:", batch.x_mean)
    print("Batch x_std:", batch.x_std)
    print("Batch y_mean:", batch.y_mean)
    print("Batch y_std:", batch.y_std)
    print("Batch n_classes:", batch.n_classes)






if __name__ == "__main__":
    main()

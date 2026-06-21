import torch
from .helper import move_batch_to_device, infer_loader_use_selector
from .metrics import classification_metrics, importance_metrics

@torch.no_grad()
def evaluate_synthetic(
    model,
    loader,
    device,
    max_batches=50,
    importance_weight=None,
):
    model.eval()

    loader_use_selector = infer_loader_use_selector(loader)

    if loader_use_selector:
        assert importance_weight is not None
        assert importance_weight > 0
    else:
        assert importance_weight is None

    total_loss_sum = 0.0
    pred_loss_sum = 0.0
    imp_loss_sum = 0.0
    n_batches = 0
    n_imp_batches = 0

    metric_sums = {}
    metric_counts = {}

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        assert bool(batch.use_selector) == loader_use_selector

        batch = move_batch_to_device(batch, device)

        out = model(batch)
        loss_dict = model.total_loss(
            batch,
            out,
            importance_weight=importance_weight,
        )

        total_loss_sum += float(loss_dict["loss"].detach())
        pred_loss_sum += float(loss_dict["pred_loss"].detach())
        n_batches += 1

        if loader_use_selector:
            imp_loss_sum += float(loss_dict["importance_loss"].detach())
            n_imp_batches += 1

        if model.task_kind == "classification":
            metrics = classification_metrics(batch, out)
        else:
            metrics = {}

        if loader_use_selector:
            metrics.update(importance_metrics(batch, out))

        for k, v in metrics.items():
            if v != v:  # skip nan
                continue
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            metric_counts[k] = metric_counts.get(k, 0) + 1

    result = {
        "loss": total_loss_sum / max(n_batches, 1),
        "pred_loss": pred_loss_sum / max(n_batches, 1),
    }

    if loader_use_selector:
        result["importance_loss"] = imp_loss_sum / max(n_imp_batches, 1)

    for k, v in metric_sums.items():
        result[k] = v / max(metric_counts[k], 1)

    return result
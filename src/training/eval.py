import torch
import math
from tqdm.auto import tqdm
from dataclasses import replace
from src.training.helper import move_batch_to_device, infer_loader_use_selector
from src.training.metrics import classification_metrics, importance_metrics, regression_metrics


def make_topk_cell_mask(batch, scores, topk_frac=0.25):

    B = scores.shape[0]
    device = scores.device

    keep_feature = torch.zeros(B, batch.d_max, dtype=torch.bool, device=device)
    for b in range(B):
        d = int(batch.d_emb[b].item())
        k = max(1, math.ceil(topk_frac * d))
        topk_idx = torch.topk(scores[b, :d], k=k).indices
        keep_feature[b, topk_idx] = True

    topk_cell_mask = batch.cell_mask.clone()
    topk_cell_mask[:, :, :batch.d_max] &= keep_feature[:, None, :]
    return topk_cell_mask


@torch.no_grad()
def evaluate_synthetic(
    model,
    loader,
    device,
    max_batches=50,
    importance_weight=None,
    topk_frac=0.25,
):
    model.eval()

    loader_use_selector = infer_loader_use_selector(loader)

    if loader_use_selector:
        assert importance_weight is not None
        assert importance_weight >= 0
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
            metrics = regression_metrics(batch, out, model.encoder.regression_borders)

        if loader_use_selector:
            metrics.update(importance_metrics(batch, out, topk_frac))

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


@torch.no_grad()
def evaluate_synthetic_extended(
    model,
    loader,
    device,
    max_batches=50,
    topk_fracs=(0.25, 0.50, 0.75),
    random_repeats=20,
):
    model.eval()

    assert infer_loader_use_selector(loader), ("Extended evaluation requires use_selector=True")
    assert random_repeats > 0

    gt_sums = {frac: 0.0 for frac in topk_fracs}
    pred_sums = {frac: 0.0 for frac in topk_fracs}
    random_sums = {frac: [0.0] * random_repeats for frac in topk_fracs}
    metric_sums = {}
    metric_counts = {}
    n_batches = 0

    for i, batch in enumerate(tqdm(loader, total=max_batches, desc="Extended eval")):
        if i >= max_batches:
            break

        assert bool(batch.use_selector)

        batch = move_batch_to_device(batch, device)
        out = model(batch)
        metrics = importance_metrics(batch, out, topk_frac=0.25)

        for key in ("importance_pearson", "importance_topk_overlap"):
            value = float(metrics[key])

            if not math.isnan(value):
                metric_sums[key] = metric_sums.get(key, 0.0) + value
                metric_counts[key] = metric_counts.get(key, 0) + 1

        gt_scores = batch.feature_importance.float()
        importance_logits = out["importance_logits"]

        feat_idx = torch.arange(batch.d_max, device=importance_logits.device)[None, :]
        feat_mask = feat_idx < batch.d_emb[:, None]
        pred_scores = torch.softmax(
            importance_logits.masked_fill(~feat_mask, float("-inf")),
            dim=-1,
        )

        for frac in topk_fracs:

            gt_mask = make_topk_cell_mask(batch, gt_scores, frac)
            gt_batch = replace(batch, cell_mask=gt_mask)
            gt_out = model(gt_batch)
            gt_loss = model.prediction_loss(gt_batch, gt_out)
            gt_sums[frac] += float(gt_loss.detach())

            pred_mask = make_topk_cell_mask(batch, pred_scores, frac)
            pred_batch = replace(batch, cell_mask=pred_mask)
            pred_out = model(pred_batch)
            pred_loss = model.prediction_loss(pred_batch, pred_out)
            pred_sums[frac] += float(pred_loss.detach())

            for r in range(random_repeats):
                random_scores = torch.rand(
                    batch.X_train.shape[0],
                    batch.d_max,
                    device=importance_logits.device,
                )

                random_mask = make_topk_cell_mask(batch, random_scores, frac)
                random_batch = replace(batch, cell_mask=random_mask)
                random_out = model(random_batch)
                random_loss = model.prediction_loss(random_batch, random_out)
                random_sums[frac][r] += float(random_loss.detach())

        n_batches += 1

    if n_batches == 0:
        raise ValueError("Evaluation loader returned no batches")

    result = {}

    for key, value in metric_sums.items():
        result[key] = value / metric_counts[key]

    for frac in topk_fracs:
        pct = round(frac * 100)
        result[f"gt_topk_pred_loss_{pct}"] = gt_sums[frac] / n_batches
        result[f"pred_topk_pred_loss_{pct}"] = pred_sums[frac] / n_batches
        random_losses = torch.tensor(
            [value / n_batches for value in random_sums[frac]],
            dtype=torch.float64,
        )
        result[f"random_topk_pred_loss_{pct}"] = random_losses.mean().item()
        result[f"random_topk_pred_loss_std_{pct}"] = random_losses.std(unbiased=False).item()

    return result
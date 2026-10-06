import torch
import math
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

    gt_topk_pred_loss_sum = 0.0
    pred_topk_pred_loss_sum = 0.0

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

        if loader_use_selector:
            gt_scores = batch.feature_importance.float()
            gt_topk_mask = make_topk_cell_mask(batch, gt_scores, topk_frac)
            gt_topk_batch = replace(batch, cell_mask=gt_topk_mask)
            gt_topk_out = model(gt_topk_batch)
            gt_topk_loss = model.prediction_loss(gt_topk_batch, gt_topk_out)
            gt_topk_pred_loss_sum += float(gt_topk_loss.detach())

            importance_logits = out["importance_logits"]
            feat_idx = torch.arange(batch.d_max, device=importance_logits.device)[None, :]
            feat_mask = feat_idx < batch.d_emb[:, None]
            pred_scores = torch.softmax(importance_logits.masked_fill(~feat_mask, float("-inf")), dim=-1)
            pred_topk_mask = make_topk_cell_mask(batch, pred_scores, topk_frac)
            pred_topk_batch = replace(batch, cell_mask=pred_topk_mask)
            pred_topk_out = model(pred_topk_batch)
            pred_topk_loss = model.prediction_loss(pred_topk_batch, pred_topk_out)
            pred_topk_pred_loss_sum += float(pred_topk_loss.detach())


    result = {
        "loss": total_loss_sum / max(n_batches, 1),
        "pred_loss": pred_loss_sum / max(n_batches, 1),
    }

    if loader_use_selector:
        result["importance_loss"] = imp_loss_sum / max(n_imp_batches, 1)
        result["gt_topk_pred_loss"] = gt_topk_pred_loss_sum / max(n_batches, 1)
        result["pred_topk_pred_loss"] = pred_topk_pred_loss_sum / max(n_batches, 1)

    for k, v in metric_sums.items():
        result[k] = v / max(metric_counts[k], 1)

    return result
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
import torch.nn.functional as F


@torch.no_grad()
def classification_metrics(batch, out):
    logits = out["logits"]          # [B, Nte_max, C]
    test_mask = out["test_mask"]    # [B, Nte_max]

    assert batch.n_classes is not None

    B, Nte_max, C = logits.shape

    class_idx = torch.arange(C, device=logits.device)[None, None, :]
    valid_class = class_idx < batch.n_classes[:, None, None]
    logits = logits.masked_fill(~valid_class, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    y_pred = logits.argmax(dim=-1)  # [B, Nte_max]
    y_true = batch.y_test.long()

    accs = []
    balanced_accs = []
    precisions = []
    recalls = []
    f1s = []
    roc_aucs = []

    for b in range(B):
        mask_b = test_mask[b]

        yt = y_true[b, mask_b].detach().cpu().numpy()
        yp = y_pred[b, mask_b].detach().cpu().numpy()

        if len(yt) == 0:
            continue

        c_b = int(batch.n_classes[b].item())
        labels = list(range(c_b))

        p, r, f1, _ = precision_recall_fscore_support(
            yt,
            yp,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        accs.append(accuracy_score(yt, yp))
        balanced_accs.append(r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        # ROC-AUC
        prob_b = probs[b, mask_b, :c_b].detach().cpu().numpy()
        # AUC requires at least 2 classes in y_true for this task
        if len(set(yt.tolist())) >= 2:
            try:
                if c_b == 2:
                    auc = roc_auc_score(yt, prob_b[:, 1])

                else:
                    auc = roc_auc_score(
                        yt,
                        prob_b,
                        labels=labels,
                        multi_class="ovr",
                        average="macro",
                    )
                roc_aucs.append(auc)

            except ValueError:
                pass

    def avg(xs):
        return float(sum(xs) / max(len(xs), 1))

    return {
        "accuracy": avg(accs),
        "balanced_accuracy": avg(balanced_accs),
        "precision": avg(precisions),
        "recall": avg(recalls),
        "f1": avg(f1s),
        "auc": avg(roc_aucs),
    }



@torch.no_grad()
def regression_metrics(batch, out, borders):
    logits = out["logits"]          # [B, Nte_max, K]
    test_mask = out["test_mask"]    # [B, Nte_max]
    y_true = batch.y_test.float()

    borders = borders.to(logits.device)
    centers = (borders[:-1] + borders[1:]) / 2.0

    probs = torch.softmax(logits, dim=-1)
    z_pred = (probs * centers[None, None, :]).sum(dim=-1)

    y_pred = z_pred * batch.y_std[:, None] + batch.y_mean[:, None]

    r2s = []
    maes = []
    rmses = []

    B = y_pred.shape[0]

    for b in range(B):
        mask_b = test_mask[b]

        yt = y_true[b, mask_b]
        yp = y_pred[b, mask_b]

        if yt.numel() == 0:
            continue

        err = yp - yt

        mae = err.abs().mean()
        rmse = torch.sqrt((err ** 2).mean())

        ss_res = ((yt - yp) ** 2).sum()
        ss_tot = ((yt - yt.mean()) ** 2).sum()

        if ss_tot > 1e-12:
            r2 = 1.0 - ss_res / ss_tot
            r2s.append(float(r2.detach()))

        maes.append(float(mae.detach()))
        rmses.append(float(rmse.detach()))

    def avg(xs):
        return float(sum(xs) / max(len(xs), 1))

    return {
        "r2": avg(r2s),
        "mae": avg(maes),
        "rmse": avg(rmses),
    }



@torch.no_grad()
def importance_metrics(batch, out):

    assert bool(batch.use_selector)
    assert out["importance_logits"] is not None

    pred = torch.sigmoid(out["importance_logits"])  # [B, d_max]
    target = batch.feature_importance.float()

    feat_idx = torch.arange(batch.d_max, device=pred.device)[None, :]
    feat_mask = feat_idx < batch.d_emb[:, None]

    p = pred[feat_mask]
    t = target[feat_mask]

    mse = F.mse_loss(p, t)

    p_center = p - p.mean()
    t_center = t - t.mean()

    pearson = (
        (p_center * t_center).mean()
        / (p_center.std(unbiased=False) * t_center.std(unbiased=False)).clamp_min(1e-12)
    )

    return {
        "importance_mse": float(mse.detach()),
        "importance_pearson": float(pearson.detach()),
    }

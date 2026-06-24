import torch.nn.functional as F
from torch import nn
from src.model.encoder import TabularInputEncoder
from src.model.backbone import  TabularBackbone
import torch
from src.model.encoder import bucketize_y

class TabularPFNModel(nn.Module):
    def __init__(
        self,
        k: int,
        m: int,
        n_heads: int,
        depth: int,
        max_cardinality: int,
        task_kind: str,
        max_classes: int | None = None,
        num_y_buckets: int | None = None,
    ):
        super().__init__()

        assert task_kind in ["classification", "regression"]
        self.task_kind = task_kind

        self.encoder = TabularInputEncoder(
            k=k,
            max_cardinality=max_cardinality,
            task_kind=task_kind,
            max_classes=max_classes,
            num_y_buckets=num_y_buckets,
        )

        self.backbone = TabularBackbone(
            k=k,
            m=m,
            n_heads=n_heads,
            depth=depth,
        )

        self.importance_head = nn.Sequential(
            nn.LayerNorm(k),
            nn.Linear(k, k),
            nn.GELU(),
            nn.Linear(k, 1),
          )

        if task_kind == "classification":
            assert max_classes is not None
            self.head = nn.Linear(k, max_classes)
        else:
            assert num_y_buckets is not None
            self.head = nn.Linear(k, num_y_buckets)

    def forward(self, batch):
        tokens, cell_mask, meta = self.encoder(batch)

        h = self.backbone(tokens, cell_mask, meta)

        test_start = meta["test_start"]
        y_slot = meta["y_slot"]
        selector_idx = meta["selector_idx"]
        d_max = meta["d_max"]

        test_repr = h[:, test_start:, y_slot, :]      # [B, Nte_max, K]
        test_mask = cell_mask[:, test_start:, y_slot] # [B, Nte_max]

        importance_logits = None

        if bool(batch.use_selector):
            selector_repr = h[:, selector_idx, :d_max, :]          # [B, d_max, K]
            importance_logits = self.importance_head(selector_repr).squeeze(-1)  # [B, d_max]

        y_test_logits = self.head(test_repr)

        return {
            "logits": y_test_logits,
            "test_mask": test_mask,
            "importance_logits": importance_logits,
            "meta": meta,
        }

    def prediction_loss(self, batch, out):
        logits = out["logits"]
        test_mask = out["test_mask"]

        if self.task_kind == "classification":
            assert batch.n_classes is not None

            B, Nte_max, C = logits.shape
            class_idx = torch.arange(C, device=logits.device)[None, None, :]
            valid_class = class_idx < batch.n_classes[:, None, None]

            logits = logits.masked_fill(~valid_class, float("-inf"))
            target = batch.y_test.long()

            target_ok = target < batch.n_classes[:, None]
            assert bool(target_ok[test_mask].all())

            return F.cross_entropy(
                logits[test_mask],
                target[test_mask],
            )

        else:
            assert batch.y_mean is not None
            assert batch.y_std is not None

            y_z = (
                batch.y_test.float()
                - batch.y_mean[:, None]
            ) / batch.y_std[:, None]

            target_bucket = bucketize_y(
                y_z,
                self.encoder.regression_borders,
            )

            return F.cross_entropy(
                logits[test_mask],
                target_bucket[test_mask],
            )

    def importance_loss(self, batch, out):

        assert bool(batch.use_selector)
        assert out["importance_logits"] is not None

        logits = out["importance_logits"]  # [B, d_max]
        pred = torch.sigmoid(logits)

        feat_idx = torch.arange(batch.d_max, device=pred.device)[None, :]
        feat_mask = feat_idx < batch.d_emb[:, None]

        target = batch.importance_ratio.float()  # [B, d_max]

        return F.mse_loss(
            pred[feat_mask],
            target[feat_mask],
        )


    def total_loss(self, batch, out, importance_weight=None):
        pred_loss = self.prediction_loss(batch, out)

        loss = pred_loss

        result = {
            "loss": loss,
            "pred_loss": pred_loss,
        }

        if bool(batch.use_selector):
            assert importance_weight is not None
            assert importance_weight > 0

            importance_loss = self.importance_loss(batch, out)
            loss = pred_loss + importance_weight * importance_loss

            result["loss"] = loss
            result["importance_loss"] = importance_loss

        else:
            assert importance_weight is None

        return result

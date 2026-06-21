import torch
from torch import nn

def make_regression_borders(
    num_bins: int = 100,
    low: float = -3.0,
    high: float = 3.0,
    device=None,
    dtype=torch.float32,
):
    assert num_bins >= 2
    assert low < high

    return torch.linspace(
        low,
        high,
        num_bins + 1,
        device=device,
        dtype=dtype,
    )


def bucketize_y(
    y_z: torch.Tensor,
    borders: torch.Tensor,
):
    num_bins = borders.numel() - 1
    y_bucket = torch.bucketize(
        y_z,
        borders.to(device=y_z.device, dtype=y_z.dtype),
        right=False,
    ) - 1
    y_bucket = y_bucket.clamp(0, num_bins - 1)

    return y_bucket.long()

class TabularInputEncoder(nn.Module):
    CONTINUOUS = 0
    CATEGORICAL = 1

    def __init__(
        self,
        k: int,
        max_cardinality: int,
        task_kind: str,
        max_classes: int | None = None,
        num_y_buckets: int | None = None,
    ):
        super().__init__()

        assert task_kind in ["classification", "regression"]

        self.k = int(k)
        self.max_cardinality = int(max_cardinality)
        self.task_kind = task_kind

        self.feature_id_dim = max(1, k // 4)
        self.feature_id_proj = nn.Linear(self.feature_id_dim, k)

        self.cont_encoder = nn.Linear(1, self.k)
        self.cat_encoder = nn.Embedding(self.max_cardinality + 1, self.k)
        self.feature_type_embed = nn.Embedding(2, self.k)

        self.missing_token = nn.Parameter(torch.randn(self.k) * 0.02)
        self.y_unknown_token = nn.Parameter(torch.randn(self.k) * 0.02)
        self.selector_token = nn.Parameter(torch.randn(self.k) * 0.02)

        self.max_classes = None
        self.num_y_buckets = None
        self.y_class_encoder = None
        self.y_reg_encoder = None
        #self.regression_borders = None

        if self.task_kind == "classification":
            assert max_classes is not None
            assert num_y_buckets is None
            assert max_classes >= 2
            self.max_classes = int(max_classes)
            self.y_class_encoder = nn.Embedding(max_classes, self.k)
            self.regression_borders = None

        else:

            assert num_y_buckets is not None
            assert max_classes is None
            assert num_y_buckets >= 2

            self.num_y_buckets = int(num_y_buckets)
            self.y_reg_encoder = nn.Embedding(self.num_y_buckets, self.k)

            self.register_buffer(
                "regression_borders",
                make_regression_borders(
                    num_bins=self.num_y_buckets,
                    low=-3.0,
                    high=3.0,
                ),
            )


    def forward(self, batch):
        device = batch.X_train.device

        B = batch.X_train.shape[0]
        Ntr_max = batch.Ntr_max
        Nte_max = batch.Nte_max
        d_max = batch.d_max

        selector_idx = Ntr_max
        test_start = Ntr_max + 1

        N = Ntr_max + 1 + Nte_max
        F = d_max + 1
        y_slot = d_max

        cell_mask = batch.cell_mask.to(device=device)
        assert cell_mask.shape == (B, N, F)
        assert batch.x_mean.shape == (B, d_max)
        assert batch.x_std.shape == (B, d_max)

        tokens = torch.zeros(B, N, F, self.k, device=device)

        feature_type = batch.feature_type.clamp(0, 1)  # [B, d_max]
        is_cont = feature_type == self.CONTINUOUS      # [B, d_max]
        is_cat = feature_type == self.CATEGORICAL      # [B, d_max]

        row_idx = torch.arange(N, device=device)[None, :, None]
        not_selector_row = row_idx != selector_idx

        type_tokens = self.feature_type_embed(feature_type)  # [B, d_max, K]

        # X feature slots
        feature_tokens = torch.zeros(B, N, d_max, self.k, device=device)

        X_all = torch.full(
            (B, N, d_max),
            torch.nan,
            dtype=torch.float32,
            device=device,
        )

        X_all[:, :Ntr_max, :] = batch.X_train
        X_all[:, test_start:, :] = batch.X_test
        # selector row stays NaN; later overwritten by selector token

        feature_mask = cell_mask[:, :, :d_max]  # [B, N, d_max]
        X_nan = torch.isnan(X_all)

        X_all_norm = (
            torch.nan_to_num(X_all, nan=0.0)
            - batch.x_mean[:, None, :]
        ) / batch.x_std[:, None, :]

        type_all = type_tokens[:, None, :, :].expand(B, N, d_max, self.k)
        cont_cell = (feature_mask & is_cont[:, None, :] & ~X_nan)
        cat_cell = (feature_mask & is_cat[:, None, :] & ~X_nan)
        missing_cell = (feature_mask & X_nan & not_selector_row)

        # continuous cells
        if bool(cont_cell.any()):
            vals = X_all_norm[cont_cell]          # [num_cont_cells]
            enc = self.cont_encoder(vals[:, None])
            feature_tokens[cont_cell] = enc + type_all[cont_cell]

        # categorical cells
        cat_ids = torch.nan_to_num(X_all, nan=0.0).long()
        cat_ids = cat_ids.clamp(0, self.max_cardinality)

        if bool(cat_cell.any()):
            ids = cat_ids[cat_cell]
            enc = self.cat_encoder(ids)
            feature_tokens[cat_cell] = enc + type_all[cat_cell]

        # missing cells
        if bool(missing_cell.any()):
            feature_tokens[missing_cell] = (
                self.missing_token[None, :]
                + type_all[missing_cell]
            )

        # selector row: overwrite selector feature cells
        feature_tokens[:, selector_idx, :, :] = (
            self.selector_token.view(1, 1, self.k) + type_tokens
        )

        tokens[:, :, :d_max, :] = feature_tokens

        # y slot
        train_y_mask = cell_mask[:, :Ntr_max, y_slot]       # [B, Ntr]
        test_y_mask = cell_mask[:, test_start:, y_slot]     # [B, Nte]

        if self.task_kind == "classification":
            assert batch.n_classes is not None
            assert self.y_class_encoder is not None
            assert int(batch.n_classes.max().item()) <= self.max_classes

            y_train_ids = batch.y_train.long().clamp(0, self.max_classes - 1)
            y_train_tokens = self.y_class_encoder(y_train_ids)

        else:
            assert batch.n_classes is None
            assert batch.y_mean is not None
            assert batch.y_std is not None
            assert self.y_reg_encoder is not None
            assert self.regression_borders is not None

            y_z = (
                batch.y_train.float()
                - batch.y_mean[:, None]
            ) / batch.y_std[:, None]
            y_bucket = bucketize_y(y_z, self.regression_borders)
            y_train_tokens = self.y_reg_encoder(y_bucket)

        tokens[:, :Ntr_max, y_slot, :] = torch.where(train_y_mask[:, :, None], y_train_tokens, tokens[:, :Ntr_max, y_slot, :],)
        y_unknown = self.y_unknown_token.view(1, 1, self.k).expand(B, Nte_max, self.k,)
        tokens[:, test_start:, y_slot, :] = torch.where(test_y_mask[:, :, None], y_unknown, tokens[:, test_start:, y_slot, :],)
        feature_noise = torch.randn(B, d_max, self.feature_id_dim, device=tokens.device, dtype=tokens.dtype,)
        feature_id = self.feature_id_proj(feature_noise)  # [B, d_max, K]
        row_has_x = torch.ones(B, N, device=tokens.device, dtype=torch.bool)
        row_has_x[:, selector_idx] = False
        feat_id_mask = cell_mask[:, :, :d_max] & row_has_x[:, :, None]
        tokens[:, :, :d_max, :] = tokens[:, :, :d_max, :] + (feature_id[:, None, :, :] * feat_id_mask[:, :, :, None].to(tokens.dtype))
        # padding / invalid cells -> zero vector
        tokens = tokens * cell_mask[:, :, :, None].to(tokens.dtype)

        assert tokens.shape == (B, N, F, self.k)
        assert torch.all(tokens[~cell_mask] == 0)

        meta = {
            "B": B,
            "N": N,
            "F": F,
            "Ntr_max": Ntr_max,
            "Nte_max": Nte_max,
            "d_max": d_max,
            "selector_idx": selector_idx,
            "test_start": test_start,
            "y_slot": y_slot,
            "n_train_keys": Ntr_max + 1,
        }

        if self.task_kind == "regression":
            meta["y_mean"] = batch.y_mean
            meta["y_std"] = batch.y_std
            meta["regression_borders"] = self.regression_borders
            meta["num_y_buckets"] = self.num_y_buckets

        return tokens, cell_mask, meta
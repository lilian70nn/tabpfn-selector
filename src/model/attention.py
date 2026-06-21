import torch
from torch import nn


class AxisAttention(nn.Module):
    """
    Generic self-attention over the second-to-last axis.

    Expected input:
        x:         [B, G, L, K]
        cell_mask: [B, G, L]
                   True  = valid token/cell
                   False = padding token/cell

    It computes attention over L:
        scores: [B, heads, G, L_query, L_key]

    For feature attention:
        x         = data_full                         # [B, N, F, K]
        cell_mask = cell_mask                         # [B, N, F]
        restrict_to_train_keys = False

    For sample attention:
        x         = data_full.permute(0, 2, 1, 3)      # [B, F, N, K]
        cell_mask = cell_mask.permute(0, 2, 1)         # [B, F, N]
        restrict_to_train_keys = True
    """

    def __init__(self, k, n_heads):
        super().__init__()
        self.k = k
        self.n_heads = n_heads

        assert k % n_heads == 0, "k must be divisible by n_heads"
        self.d_k = k // n_heads

        self.W_q = nn.Linear(k, k, bias=False)
        self.W_k = nn.Linear(k, k, bias=False)
        self.W_v = nn.Linear(k, k, bias=False)
        self.W_c = nn.Linear(k, k, bias=False)

    def forward(
        self,
        data,
        cell_mask,
        restrict_to_train_keys=False,
        n_train_keys=None,
        attn_bias=None,
    ):
        """
        data:         [B, G, L, K]
        cell_mask: [B, G, L], bool

        restrict_to_train_keys:
            If True, keys with index >= n_train_keys are masked out.
            This is useful for sample/row attention where only train rows
            can be attended to.

        n_train_keys:
            Usually Ntr_max when L is the row/sample axis.

        attn_bias:
            Optional additive attention bias.
            Shape should be broadcastable to [B, heads, G, L, L].
            Example: [B, 1, G, L, L] or [B, 1, 1, 1, L].

        return_attn:
            If True, return (proj, attn). Otherwise return proj.
        """
        B, G, L, K = data.shape
        assert K == self.k, f"Expected hidden dim {self.k}, got {K}"

        if cell_mask.dtype != torch.bool:
            cell_mask = cell_mask.bool()

        # Q/K/V: [B, G, L, K]
        Q = self.W_q(data)
        K_ = self.W_k(data)
        V = self.W_v(data)

        # [B, G, L, K] -> [B, heads, G, L, d_k]
        Q = Q.view(B, G, L, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)
        K_ = K_.view(B, G, L, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)
        V = V.view(B, G, L, self.n_heads, self.d_k).permute(0, 3, 1, 2, 4)

        # scores: [B, heads, G, L_query, L_key]
        scores = (Q @ K_.transpose(-2, -1)) / (self.d_k ** 0.5)

        # pair_mask: [B, G, L_query, L_key]
        # valid query can attend valid key
        pair_mask = cell_mask[:, :, :, None] & cell_mask[:, :, None, :]

        # Optional: for sample attention, only allow keys before n_train_keys.
        # This reproduces your old key_ok = train_ok logic.
        if restrict_to_train_keys:
            if n_train_keys is None:
                raise ValueError("n_train_keys must be provided when restrict_to_train_keys=True")

            pair_mask = pair_mask.clone()
            pair_mask[..., n_train_keys:] = False

        # Apply hard mask.
        # Masked positions become -inf before softmax.
        scores = scores.masked_fill(~pair_mask[:, None, :, :, :], float("-inf"))

        # Optional additive bias, e.g. selector score bias.
        if attn_bias is not None:
            scores = scores + attn_bias

        attn = torch.softmax(scores, dim=-1)

        # If a padding query has no valid keys, softmax([-inf, ...]) gives NaN.
        # We intentionally convert those rows to zero attention.
        attn = torch.nan_to_num(attn, nan=0.0)

        out = attn @ V
        # out: [B, heads, G, L, d_k]

        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, G, L, self.k)

        proj = self.W_c(out)

        # Clear padding cells after projection.
        proj = proj * cell_mask[:, :, :, None].to(proj.dtype)

        return proj
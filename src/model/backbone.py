from torch import nn
from .attention import AxisAttention

class ResidualNorm(nn.Module):
    def __init__(self, k, sublayer):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(k)

    def forward(self, data, *args, **kwargs):
        data_temp = self.sublayer(self.norm(data), *args, **kwargs)
        return data_temp + data


class Feedforward(nn.Module):
    """
    Feedforward neural network.
    d_emb: embedding dimension
    m: hidden dimension

    Input: x of shape (n, k)
    Output: x of shape (n, k)
    """
    def __init__(self, k, m, dropout=0.1):
        super().__init__()
        self.m = m
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(k, m),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(m, k),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        data = self.net(data)
        return data



class TransformerBlock(nn.Module):
    def __init__(self, k, m, n_heads):
        super().__init__()
        self.fAtt = ResidualNorm(k, AxisAttention(k, n_heads))
        self.sAtt = ResidualNorm(k, AxisAttention(k, n_heads))
        self.forw = ResidualNorm(k, Feedforward(k, m))

    def forward(self, data, cell_mask, meta):
        data = self.fAtt(data, cell_mask)
        data = data * cell_mask[:, :, :, None].to(data.dtype)

        data = data.permute(0, 2, 1, 3).contiguous()       # [B, F, N, K]
        mask_t = cell_mask.permute(0, 2, 1).contiguous()   # [B, F, N]

        data = self.sAtt(
            data,
            mask_t,
            restrict_to_train_keys=True,
            n_train_keys=meta["n_train_keys"],
        )
        data = data * mask_t[:, :, :, None].to(data.dtype)

        data = data.permute(0, 2, 1, 3).contiguous()       # [B, N, F, K]
        data = self.forw(data)
        data = data * cell_mask[:, :, :, None].to(data.dtype)

        return data


class TabularBackbone(nn.Module):
    def __init__(self, k, m, n_heads, depth):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(k=k, m=m, n_heads=n_heads)
            for _ in range(depth)
        ])

    def forward(self, tokens, cell_mask, meta):
        x = tokens

        for block in self.blocks:
            x = block(x, cell_mask, meta)

        return x
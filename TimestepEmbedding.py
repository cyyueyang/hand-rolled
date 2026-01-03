import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}

def get_activation(act_fn):
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"Unknown activation function: {act_fn}")

class TimestepEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int,
                 time_embed_dim: int,
                 act_fn: str = 'silu',
                 out_dim: int = None,
                 post_act_fn: Optional[str] = None,
                 cond_proj_dim: int = None,
                 sample_proj_bias: bool = True,
                 ):

        super(TimestepEmbedding, self).__init__()
        self.linear1 = nn.Linear(in_channels, time_embed_dim)
        self.act_fn = get_activation(act_fn)
        out_dim = time_embed_dim if out_dim is None else out_dim
        self.linear2 = nn.Linear(time_embed_dim, out_dim, bias=sample_proj_bias)
        self.post_act_fn = get_activation(post_act_fn) if post_act_fn is not None else None
        self.cond_proj = nn.Linear(cond_proj_dim, in_channels) if cond_proj_dim is not None else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor=None) -> torch.Tensor:
        if self.cond_proj is not None and cond is not None:
            x = self.cond_proj(cond) + x

        x = self.linear2(self.act_fn(self.linear1(x)))

        if self.post_act_fn is not None:
            x = self.post_act_fn(x)

        return x

if __name__ == '__main__':
    TimestepEmbeddings = TimestepEmbedding(in_channels=256, time_embed_dim=512)
    x = torch.randn(16, 256)
    print(f"x size: {x.size()}")
    y = TimestepEmbeddings(x)
    print(f"y size: {y.size()}")

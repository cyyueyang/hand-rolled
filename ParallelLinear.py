import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional

world_size = 4
rank = 0

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: Optional[torch.dtype] = None):
        super(ColumnParallelLinear, self).__init__()
        assert out_features % world_size == 0, f"out_features must be divisible by {world_size}"
        self.in_features = in_features
        self.out_features = out_features
        self.part_out_features = out_features // world_size
        self.weight = nn.Parameter(torch.empty(self.part_out_features, self.in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        y = F.linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: Optional[torch.dtype] = None):
        super(RowParallelLinear, self).__init__()
        assert in_features % world_size == 0, f"in_features must be divisible by {world_size}"
        self.in_features = in_features
        self.out_features = out_features
        self.part_in_features = in_features // world_size
        self.weight = nn.Parameter(torch.empty(self.out_features, self.part_in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
    def forward(self, x: torch.Tensor):
        y = F.linear(x, self.weight)

        if world_size > 1:
            dist.all_reduce(y)

        if self.bias is not None:
            y = y + self.bias
        return y


import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, Literal, Optional

gemm_impl: Literal["bf16", "fp8"] = "bf16"
block_size = 128

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None) -> torch.Tensor:
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)  # weight_dequant 没实现
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size, scale_fmt) # act_quant 没实现
        y = fp8_gemm(x, scale, weight, weight.scale) # fp8_gemm没实现
        if bias is not None:
            y = y + bias
        return y

class Linear(nn.Module):

    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype: Optional[torch.dtype] = None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))

        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, in_features, dtype=torch.float32))
        else:
            self.register_parameter('scale', None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor):

        return linear(x, self.weight, self.bias, Linear.scale_fmt)





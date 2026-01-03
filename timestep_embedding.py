import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps: torch.Tensor,
                           embedding_dim: int,
                           flip_sin_to_cos: bool = False,
                           downscale_freq_shift: float = 1.0,
                           scale: float = 1.0,
                           max_period: float = 10000.0
                           ):
    assert timesteps.dim() == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim).float()
    exponent = exponent / (max_period - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = torch.outer(timesteps.float(), emb)
    emb = emb * scale

    if flip_sin_to_cos:
        t_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    else:
        t_emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

    if half_dim % 2 == 1:
        t_emb = F.pad(t_emb, (0, 1, 0, 0))
    return t_emb

class Timesteps(nn.Module):
    def __init__(self,
                 num_channels: int,
                 flip_sin_to_cos: bool = False,
                 downscale_freq_shift: float = 1.0,
                 scale: float = 1.0):

        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor):
        t_emb = get_timestep_embedding(
            timesteps=timesteps,
            embedding_dim=self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale
        )
        return t_emb


if __name__ == "__main__":
    timesteps = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1.0)
    x = torch.randn(10)
    print(f"x dim is {x.dim()}")
    y = timesteps(x)
    print(f"y size is {y.size()}")

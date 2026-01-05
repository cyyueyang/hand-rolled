import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist

world_size = 4
rank = 0

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()

        assert vocab_size // world_size == 0, f"vocab_size must be divisible by {vocab_size}"
        self.vocab_size = vocab_size
        self.part_vocab_size = vocab_size // world_size
        self.dim = dim
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor):
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x -= self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y

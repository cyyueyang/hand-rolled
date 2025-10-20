import math

import torch
import torch.nn.functional as F
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, n_embed, n_head):
        super(MHA, self).__init__()

        self.n_embed = n_embed
        self.n_head = n_head
        self.hs = n_embed // n_head
        self.qkv = nn.Linear(n_embed, n_embed * 3)
        self.out = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        bs, seq_len, dim = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        assert dim % self.n_head == 0, "dim must be divisible by n_head."
        q = q.view(bs, seq_len, self.n_head, dim // self.n_head).transpose(1, 2)
        k = k.view(bs, seq_len, self.n_head, dim // self.n_head).transpose(1, 2)
        v = v.view(bs, seq_len, self.n_head, dim // self.n_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.hs)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        attn = attn.masked_fill(mask == 0, -float("inf"))
        attn = F.softmax(attn, dim=-1)

        o = attn @ v
        o = o.transpose(1, 2).contiguous().view(bs, seq_len, dim)
        o = self.out(o)

        return o


if __name__ == '__main__':
    n_embed = 512
    n_head = 8
    model = MHA(n_embed, n_head)
    y = model(torch.randn(32, 128, 512))
    print(y.shape)





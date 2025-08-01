from torch import nn
import torch
from einops import einsum
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, device=device, dtype=dtype), requires_grad=True)
        std = np.sqrt(2/(in_dim + out_dim))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
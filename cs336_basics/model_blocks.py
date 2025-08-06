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


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype), requires_grad=True)

    def set(self, weights):
        self.weight = torch.nn.Parameter(weights.to(device=self.weight.device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_model, device=device, dtype=dtype), requires_grad=True)
        self.d_model = d_model
        self.dtype = dtype
        self.eps = eps

    def apply(self, weight):
        self.weight = torch.nn.Parameter(weight.to(device=self.weight.device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        rms = torch.sqrt(norm ** 2 / self.d_model + self.eps)

        result = einsum(x / rms, self.weight, '... d, d -> ... d')
        return result.to(self.dtype)

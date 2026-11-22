# meme_llm/model/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """y = x / rms(x) * weight, rms = sqrt(mean(x^2) + eps)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute in float32 for stability under mixed precision, cast back after
        x_f = x.float()
        norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


class SwiGLU(nn.Module):
    """FFN: w3( silu(w1 x) * w2 x ).
    Hidden width is ~(8/3)*dim rounded to a multiple of 64, which keeps the
    parameter count close to a standard 4x GELU FFN (3 matrices instead of 2)."""

    def __init__(self, dim: int, dropout: float = 0.0, multiple_of: int = 64):
        super().__init__()
        hidden = int(8 * dim / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate branch
        self.w2 = nn.Linear(dim, hidden, bias=False)   # value branch
        self.w3 = nn.Linear(hidden, dim, bias=False)   # back to dim
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


if __name__ == "__main__":
    x = torch.randn(2, 5, 384)
    print(RMSNorm(384)(x).shape, SwiGLU(384)(x).shape)
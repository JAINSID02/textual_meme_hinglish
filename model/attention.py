# meme_llm/model/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPE


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE.
    x: (B, T, C) -> (B, T, C)"""

    def __init__(self, n_embd: int, n_head: int, max_len: int,
                 dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.d_head = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = RoPE(self.d_head, max_len, rope_base)
        self.dropout_p = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # (B, H, T, D)

        q, k = self.rope(q, k)

        # Fused attention; is_causal applies the lower-triangular mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


if __name__ == "__main__":
    attn = CausalSelfAttention(n_embd=384, n_head=8, max_len=256)
    x = torch.randn(2, 20, 384)
    print(attn(x).shape)

    # Causality check: changing a FUTURE token must not change earlier outputs
    attn.eval()
    x2 = x.clone()
    x2[:, 15:] = torch.randn(2, 5, 384)
    y1, y2 = attn(x), attn(x2)
    print(torch.allclose(y1[:, :15], y2[:, :15], atol=1e-5))
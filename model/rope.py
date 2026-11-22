# meme_llm/model/rope.py
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary positional embedding.
    Precomputes cos/sin once for positions 0..max_len-1 and rotates q and k.
    Buffers are non-persistent: they are rebuilt on load, not stored in checkpoints."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)              # (max_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)       # (max_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """q, k: (B, H, T, D). Returns rotated q, k of the same shape."""
        T = q.size(2)
        cos = self.cos[:T].to(q.dtype)[None, None]    # (1, 1, T, D)
        sin = self.sin[:T].to(q.dtype)[None, None]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


if __name__ == "__main__":
    rope = RoPE(head_dim=48, max_len=128)
    q = torch.randn(2, 8, 20, 48)
    k = torch.randn(2, 8, 20, 48)
    q2, k2 = rope(q, k)
    print(q2.shape, k2.shape)

    # Key property: the attention score depends only on RELATIVE position.
    # Same q/k vectors placed at positions (3,1) and (13,11) must score the same.
    qv, kv = torch.randn(48), torch.randn(48)
    Q = torch.zeros(1, 1, 128, 48); K = torch.zeros(1, 1, 128, 48)
    Q[0, 0, 3], K[0, 0, 1] = qv, kv
    Q[0, 0, 13], K[0, 0, 11] = qv, kv
    Qr, Kr = rope(Q, K)
    print(torch.allclose(Qr[0, 0, 3] @ Kr[0, 0, 1], Qr[0, 0, 13] @ Kr[0, 0, 11], atol=1e-4))
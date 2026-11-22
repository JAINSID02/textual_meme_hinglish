# meme_llm/model/transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLU
from .attention import CausalSelfAttention


class Block(nn.Module):
    """Pre-norm block: x + Attn(Norm(x)), then x + FFN(Norm(x))."""

    def __init__(self, n_embd, n_head, max_len, dropout, rope_base):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_len, dropout, rope_base)
        self.norm2 = RMSNorm(n_embd)
        self.ffn = SwiGLU(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Backbone(nn.Module):
    """Token ids (B, T) -> hidden states (B, T, C). No LM head here,
    so the classifier can reuse it directly."""

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 dropout=0.0, rope_base=10000.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, rope_base) for _ in range(n_layer)]
        )
        self.norm_f = RMSNorm(n_embd)   # final norm (the old Part 3 model skipped this)

        self.apply(self._init_weights)
        # scaled init for the layers that write into the residual stream
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w3.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        assert idx.size(1) <= self.block_size, "sequence longer than block_size"
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x)
        return self.norm_f(x)


class TransformerLM(nn.Module):
    """Backbone + language-model head, used for Hinglish pretraining."""

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 dropout=0.0, rope_base=10000.0):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                           n_head=n_head, n_embd=n_embd, dropout=dropout, rope_base=rope_base)
        self.backbone = Backbone(**self.config)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.backbone.tok_emb.weight   # weight tying

    def forward(self, idx, targets=None):
        logits = self.lm_head(self.backbone(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-100)
        return logits, loss

    @classmethod
    def from_config(cls, mcfg):
        """Build from config.ModelConfig."""
        return cls(mcfg.vocab_size, mcfg.block_size, mcfg.n_layer, mcfg.n_head,
                   mcfg.n_embd, mcfg.dropout, mcfg.rope_base)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config import cfg

    model = TransformerLM.from_config(cfg.model)
    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e6:.2f}M")

    x = torch.randint(0, cfg.model.vocab_size, (2, 32))
    logits, loss = model(x, x)
    print(logits.shape, f"loss={loss.item():.3f}  (expect ~ln(vocab)={math.log(cfg.model.vocab_size):.3f})")
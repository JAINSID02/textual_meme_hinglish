# meme_llm/model/classifier.py
import torch
import torch.nn as nn

from .transformer import Backbone


class MemeClassifier(nn.Module):
    """Backbone -> masked mean pooling -> dropout -> Linear(n_classes).
    Input ids are right-padded with pad_id. Output: logits (B, n_classes)."""

    def __init__(self, backbone_cfg: dict, n_classes: int, pad_id: int, head_dropout: float = 0.1):
        super().__init__()
        self.backbone_cfg = dict(backbone_cfg)
        self.pad_id = pad_id
        self.backbone = Backbone(**self.backbone_cfg)
        n_embd = self.backbone_cfg["n_embd"]
        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(n_embd, n_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.backbone(idx)                              # (B, T, C)
        mask = (idx != self.pad_id).unsqueeze(-1).to(h.dtype)   # (B, T, 1), 1 = real token
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(self.head_drop(pooled))            # (B, n_classes)

    def load_pretrained_backbone(self, ckpt_path: str, map_location="cpu"):
        """Load only the backbone weights from a pretraining checkpoint
        (saved by training/pretrain.py as {'model': TransformerLM state_dict, 'config': {...}})."""
        ckpt = torch.load(ckpt_path, map_location=map_location)
        saved_cfg = ckpt["config"]
        for k in ("vocab_size", "n_layer", "n_head", "n_embd"):
            assert saved_cfg[k] == self.backbone_cfg[k], \
                f"{k} mismatch: checkpoint={saved_cfg[k]} vs classifier={self.backbone_cfg[k]}"
        prefix = "backbone."
        state = {k[len(prefix):]: v for k, v in ckpt["model"].items() if k.startswith(prefix)}
        missing, unexpected = self.backbone.load_state_dict(state, strict=True)
        print(f"[classifier] loaded pretrained backbone from {ckpt_path}")

    @classmethod
    def from_config(cls, mcfg, n_classes: int, pad_id: int):
        backbone_cfg = dict(vocab_size=mcfg.vocab_size, block_size=mcfg.block_size,
                            n_layer=mcfg.n_layer, n_head=mcfg.n_head, n_embd=mcfg.n_embd,
                            dropout=mcfg.dropout, rope_base=mcfg.rope_base)
        return cls(backbone_cfg, n_classes, pad_id)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config import cfg, PAD_ID

    model = MemeClassifier.from_config(cfg.model, cfg.num_classes, PAD_ID).eval()
    x = torch.randint(5, cfg.model.vocab_size, (3, 16))
    x[:, 10:] = PAD_ID                      # right-pad the last 6 positions
    logits = model(x)
    print(logits.shape)                     # (3, num_classes)

    # Padding check: changing pad positions must not change the output
    x2 = x.clone(); x2[:, 10:] = PAD_ID
    print(torch.allclose(model(x), model(x2)))

    # Length check: the same text with MORE padding must give the same logits
    x3 = torch.cat([x, torch.full((3, 8), PAD_ID)], dim=1)
    print(torch.allclose(model(x), model(x3), atol=1e-5))
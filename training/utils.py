# meme_llm/training/utils.py
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- learning-rate schedule ----------------
class WarmupCosine:
    """Linear warmup, then cosine decay down to min_lr_ratio * base_lr.
    Stateless: the lr is a pure function of the step index, so resuming
    only needs the saved step number (no scheduler state in checkpoints)."""

    def __init__(self, optimizer, base_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.min_lr = base_lr * min_lr_ratio

    def lr_at(self, step: int) -> float:
        """step = 0-based index of the optimizer step about to be taken."""
        if step < self.warmup:
            return self.base_lr * (step + 1) / self.warmup
        progress = min(1.0, (step - self.warmup) / (self.total - self.warmup))
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def set(self, step: int) -> float:
        lr = self.lr_at(step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


# ---------------- optimizer ----------------
def build_optimizer(model, lr, weight_decay, betas=(0.9, 0.95)):
    """AdamW with weight decay on matrices only (dim >= 2).
    Norm weights and biases (dim < 2) are not decayed."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


# ---------------- checkpoints ----------------
def save_checkpoint(path, model, optimizer=None, scaler=None, step=0, config=None, extra=None):
    """Atomic save: write to a temp file, then rename, so a crash or Ctrl+C
    mid-write can never leave a corrupt checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "config": config if config is not None else getattr(model, "config", None),
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model=None, optimizer=None, scaler=None, map_location="cpu"):
    """Returns the full checkpoint dict. Restores whatever objects are passed in."""
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def prune_checkpoints(directory, pattern="step_*.pt", keep_last=2):
    files = sorted(Path(directory).glob(pattern))
    for f in files[:-keep_last]:
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    import tempfile
    import torch.nn as nn

    # 1) schedule shape
    m = nn.Linear(4, 4)
    opt = build_optimizer(m, lr=1e-3, weight_decay=0.1)
    sch = WarmupCosine(opt, base_lr=1e-3, warmup_steps=10, total_steps=100)
    lrs = [sch.set(s) for s in range(100)]
    print("peak:", max(lrs), "| first:", lrs[0], "| last:", round(lrs[-1], 6))
    print("warmup rises:", lrs[0] < lrs[9], "| decays:", lrs[-1] < lrs[10])

    # 2) weight-decay groups: weight (2D) decayed, bias (1D) not
    print([g["weight_decay"] for g in opt.param_groups])

    # 3) checkpoint round trip
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ck.pt"
        save_checkpoint(p, m, opt, None, step=7, config={"n_embd": 4})
        m2 = nn.Linear(4, 4)
        ck = load_checkpoint(p, m2)
        print("step:", ck["step"], "| weights equal:", torch.equal(m.weight, m2.weight))
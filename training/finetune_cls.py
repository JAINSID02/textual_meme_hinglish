# meme_llm/training/finetune_cls.py
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, CATEGORIES, PAD_ID
from tokenizer.train_tokenizer import HinglishTokenizer
from model.classifier import MemeClassifier
from training.cls_dataset import make_cls_loaders
from training.utils import (set_seed, get_device, WarmupCosine, build_optimizer,
                            save_checkpoint)


@torch.no_grad()
def evaluate(model, loader, device, weights=None):
    """Returns (loss, accuracy, macro_f1) over a loader."""
    model.eval()
    all_pred, all_true, total_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, weight=weights, reduction="sum").item()
        n += y.numel()
        all_pred += logits.argmax(-1).tolist()
        all_true += y.tolist()
    acc = sum(p == t for p, t in zip(all_pred, all_true)) / max(1, n)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / max(1, n), acc, macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=str, default=str(Path(cfg.paths.pretrain_dir) / "model_best.pt"),
                    help="pretraining checkpoint to start from")
    ap.add_argument("--from_scratch", action="store_true",
                    help="skip pretrained weights (use this once to measure how much pretraining helps)")
    ap.add_argument("--out", type=str, default=str(cfg.paths.cls_dir))
    args = ap.parse_args()

    cc = cfg.cls
    set_seed(cc.seed)
    device = get_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = HinglishTokenizer()
    train_loader, val_loader, _, weights = make_cls_loaders(tok)
    if weights is not None:
        weights = weights.to(device)

    model = MemeClassifier.from_config(cfg.model, cfg.num_classes, PAD_ID).to(device)
    if args.from_scratch:
        print("[finetune] training from RANDOM init (no pretraining)")
    else:
        model.load_pretrained_backbone(args.pretrained, map_location=device)

    opt = build_optimizer(model, cc.lr, cc.weight_decay, betas=(0.9, 0.999))
    total_steps = cc.epochs * len(train_loader)
    sched = WarmupCosine(opt, cc.lr, warmup_steps=max(1, total_steps // 10), total_steps=total_steps)

    best_f1, bad_epochs, step = -1.0, 0, 0
    best_path = out_dir / "model_best.pt"
    classifier_cfg = {"backbone": model.backbone_cfg, "n_classes": cfg.num_classes,
                      "pad_id": PAD_ID, "categories": CATEGORIES}

    for epoch in range(1, cc.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            sched.set(step)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            run_loss += loss.item() * y.numel()
            seen += y.numel()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, weights)
        print(f"epoch {epoch:2d}/{cc.epochs} | train loss {run_loss/seen:.4f} "
              f"| val loss {val_loss:.4f} | val acc {val_acc:.3f} | val macro-F1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            save_checkpoint(best_path, model, step=step, config=classifier_cfg,
                            extra={"val_macro_f1": val_f1, "val_acc": val_acc, "epoch": epoch})
            print(f"           new best macro-F1 -> {best_path.name}")
        else:
            bad_epochs += 1
            if bad_epochs >= cc.patience:
                print(f"[finetune] early stop (no val macro-F1 gain for {cc.patience} epochs)")
                break

    print(f"[finetune] best val macro-F1 = {best_f1:.3f}  ({best_path})")


if __name__ == "__main__":
    main()
# meme_llm/eval/metrics.py
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, CATEGORIES
from tokenizer.train_tokenizer import HinglishTokenizer
from model.classifier import MemeClassifier
from training.cls_dataset import make_cls_loaders
from training.utils import get_device


def load_classifier(ckpt_path, device):
    """Rebuild a MemeClassifier from a checkpoint saved by finetune_cls.py.
    Returns (model, categories). Reused by predict.py."""
    ck = torch.load(ckpt_path, map_location=device)
    c = ck["config"]
    model = MemeClassifier(c["backbone"], c["n_classes"], c["pad_id"]).to(device)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, c["categories"]


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Returns (y_true, y_pred, probs) as numpy arrays, in loader order."""
    ys, preds, probs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = F.softmax(logits, dim=-1).cpu()
        probs.append(p)
        preds.append(p.argmax(-1))
        ys.append(y)
    return (torch.cat(ys).numpy(), torch.cat(preds).numpy(), torch.cat(probs).numpy())


def summarize(y_true, y_pred, categories):
    """Prints the headline numbers + per-class report. Returns a dict of the headlines."""
    labels = list(range(len(categories)))
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    print(f"accuracy    : {acc:.3f}")
    print(f"macro-F1    : {macro:.3f}   (every class counts equally)")
    print(f"weighted-F1 : {weighted:.3f}   (weighted by class size)")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=categories, zero_division=0, digits=3))
    return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}


def plot_confusion(y_true, y_pred, categories, out_path):
    labels = list(range(len(categories)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)     # row-normalized (recall view)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, vmin=0, vmax=1)
    ax.set_xticks(labels); ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticks(labels); ax.set_yticklabels(categories)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (rows sum to 1; numbers = counts)")
    for i in labels:
        for j in labels:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm_norm[i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[metrics] confusion matrix -> {out_path}")


def save_errors(texts, y_true, y_pred, probs, categories, out_path):
    """CSV of every misclassified test meme, most confident mistakes first.
    Read this file: it shows whether errors are model failures or label noise."""
    rows = []
    for t, yt, yp, p in zip(texts, y_true, y_pred, probs):
        if yt != yp:
            rows.append({"text": t, "true": categories[yt], "pred": categories[yp],
                         "confidence": float(p[yp])})
    df = pd.DataFrame(rows).sort_values("confidence", ascending=False)
    df.to_csv(out_path, index=False)
    print(f"[metrics] {len(df)} errors -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(Path(cfg.paths.cls_dir) / "model_best.pt"))
    ap.add_argument("--out_dir", type=str, default=None, help="default: same folder as the checkpoint")
    args = ap.parse_args()

    device = get_device()
    out_dir = Path(args.out_dir or Path(args.ckpt).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = HinglishTokenizer()
    _, _, test_loader, _ = make_cls_loaders(tok)
    model, categories = load_classifier(args.ckpt, device)

    print(f"[metrics] evaluating {args.ckpt} on the TEST set ({len(test_loader.dataset)} memes)\n")
    y_true, y_pred, probs = collect_predictions(model, test_loader, device)
    summarize(y_true, y_pred, categories)
    plot_confusion(y_true, y_pred, categories, out_dir / "confusion_matrix.png")
    save_errors(test_loader.dataset.texts, y_true, y_pred, probs, categories, out_dir / "test_errors.csv")


if __name__ == "__main__":
    main()
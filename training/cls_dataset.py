# meme_llm/training/cls_dataset.py
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, LABEL2ID
from tokenizer.train_tokenizer import HinglishTokenizer


class MemeDataset(Dataset):
    """Reads a split CSV (columns: text,label). Each text is tokenized once,
    truncated to max_len-1 tokens, and ends with <eos> (same as pretraining)."""

    def __init__(self, csv_path, tok: HinglishTokenizer, max_len: int):
        df = pd.read_csv(csv_path)
        self.labels = [LABEL2ID[l] for l in df["label"]]
        self.texts = df["text"].astype(str).tolist()
        self.items = []
        for t in self.texts:
            ids = tok.encode(t)[: max_len - 1] + [tok.eos_id]
            self.items.append(ids)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.labels[i]


def make_collate(pad_id: int):
    def collate(batch):
        """Right-pad to the longest sequence IN THE BATCH (not always max_len),
        which is faster. The classifier ignores pads when pooling."""
        max_t = max(len(ids) for ids, _ in batch)
        x = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor([lab for _, lab in batch], dtype=torch.long)
        return x, y
    return collate


def compute_class_weights(labels, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized so the average weight is 1.
    Rare classes get a bigger weight in the loss."""
    counts = torch.bincount(torch.tensor(labels), minlength=n_classes).float()
    counts = counts.clamp_min(1.0)
    w = counts.sum() / (n_classes * counts)
    return w / w.mean()


def make_cls_loaders(tok: HinglishTokenizer, processed_dir=None):
    processed_dir = Path(processed_dir or cfg.paths.processed_dir)
    cc = cfg.cls
    collate = make_collate(tok.pad_id)

    train_ds = MemeDataset(processed_dir / "train.csv", tok, cc.max_len)
    val_ds = MemeDataset(processed_dir / "val.csv", tok, cc.max_len)
    test_ds = MemeDataset(processed_dir / "test.csv", tok, cc.max_len)

    train_loader = DataLoader(train_ds, batch_size=cc.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cc.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cc.batch_size, shuffle=False, collate_fn=collate)

    weights = compute_class_weights(train_ds.labels, cfg.num_classes) if cc.use_class_weights else None
    print(f"[cls_data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if weights is not None:
        print(f"[cls_data] class weights: {[round(w, 2) for w in weights.tolist()]}")
    return train_loader, val_loader, test_loader, weights


if __name__ == "__main__":
    tok = HinglishTokenizer()
    tr, va, te, w = make_cls_loaders(tok)
    x, y = next(iter(tr))
    print(x.shape, y.shape, y.tolist()[:8])
    print("pad id used:", tok.pad_id, "| pads in batch:", int((x == tok.pad_id).sum()))
    # truncation check: no sequence may exceed max_len
    print("max len ok:", x.size(1) <= cfg.cls.max_len)
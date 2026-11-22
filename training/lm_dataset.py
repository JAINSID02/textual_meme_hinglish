# meme_llm/training/lm_dataset.py
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg
from tokenizer.train_tokenizer import HinglishTokenizer


def encode_corpus(path, tok: HinglishTokenizer, seed: int = 42) -> torch.Tensor:
    """Read one example per line, shuffle the lines, encode each and append
    <eos> after it, then concatenate everything into one long id tensor."""
    path = Path(path)
    cache = path.with_suffix(".ids.pt")
    if cache.exists():
        return torch.load(cache)

    lines = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    random.Random(seed).shuffle(lines)
    ids = []
    for ln in lines:
        ids.extend(tok.encode(ln))
        ids.append(tok.eos_id)
    ids = torch.tensor(ids, dtype=torch.long)
    torch.save(ids, cache)
    print(f"[lm_data] {len(lines)} lines -> {ids.numel()} tokens (cached to {cache.name})")
    return ids


class BlockDataset(Dataset):
    """Non-overlapping blocks: sample i = tokens [i*B, (i+1)*B) with targets shifted by 1.
    One pass over the dataset = exactly one epoch of the corpus."""

    def __init__(self, ids: torch.Tensor, block_size: int):
        self.ids = ids
        self.block_size = block_size
        self.n_blocks = (ids.numel() - 1) // block_size

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, i):
        s = i * self.block_size
        x = self.ids[s : s + self.block_size]
        y = self.ids[s + 1 : s + self.block_size + 1]
        return x, y


def make_loaders(tok: HinglishTokenizer, corpus_path=None, val_frac: float = 0.02):
    corpus_path = corpus_path or cfg.paths.corpus_txt.with_name("hinglish_corpus_clean.txt")
    ids = encode_corpus(corpus_path, tok)

    n_val = int(ids.numel() * val_frac)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    bs, pc = cfg.model.block_size, cfg.pretrain
    train_ds = BlockDataset(train_ids, bs)
    val_ds = BlockDataset(val_ids, bs)
    train_loader = DataLoader(train_ds, batch_size=pc.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=pc.batch_size, shuffle=False, drop_last=True)

    steps_per_epoch = len(train_loader) // pc.grad_accum_steps
    print(f"[lm_data] train tokens={train_ids.numel()} val tokens={val_ids.numel()}")
    print(f"[lm_data] micro-batches/epoch={len(train_loader)}  "
          f"optimizer steps/epoch={steps_per_epoch}  <- set PretrainConfig.steps to this x epochs")
    return train_loader, val_loader, steps_per_epoch


if __name__ == "__main__":
    tok = HinglishTokenizer()
    train_loader, val_loader, spe = make_loaders(tok)
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)
    print("shift ok:", torch.equal(x[:, 1:], y[:, :-1]))
# meme_llm/data/split.py
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, CATEGORIES


def split_dataset(in_path=None, out_dir=None, val_frac=0.1, test_frac=0.1, seed=None):
    """Read the cleaned CSV (text,label) and write train/val/test CSVs.
    Stratified so each split keeps the same category proportions."""
    seed = cfg.cls.seed if seed is None else seed
    in_path = Path(in_path or cfg.paths.raw_csv.with_name("memes_clean.csv"))
    out_dir = Path(out_dir or cfg.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # 1) label sanity: drop unknown labels, warn about them
    bad = ~df["label"].isin(CATEGORIES)
    if bad.any():
        print(f"[split] dropping {bad.sum()} rows with unknown labels: "
              f"{sorted(df.loc[bad, 'label'].astype(str).unique())}")
        df = df[~bad]

    # 2) dedupe again so identical text can never land in two splits
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    # 3) classes with too few rows cannot be stratified into 3 splits
    counts = df["label"].value_counts()
    rare = counts[counts < 10].index.tolist()
    if rare:
        print(f"[split] WARNING: classes with <10 examples: {rare}. "
              f"Collect more data or merge them into 'other'.")
        df = df[~df["label"].isin(rare)]

    # 4) stratified split: first carve out test, then val from the remainder
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed)
    val_rel = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val["label"], random_state=seed)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out_dir / f"{name}.csv", index=False)

    print(f"[split] train={len(train)} val={len(val)} test={len(test)} -> {out_dir}")
    print("[split] class counts (train):")
    print(train["label"].value_counts().to_string())
    return train, val, test


if __name__ == "__main__":
    split_dataset()
# meme_llm/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent

# ---------------- categories (EDIT THESE) ----------------
CATEGORIES: List[str] = [
    "relatable",
    "political",
    "bollywood",
    "relationship",
    "exam_job",
    "dark_humor",
    "other",
]
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

# ---------------- special tokens (must match tokenizer order) ----------------
SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
PAD_ID = 2  # id of "<pad>"


@dataclass
class Paths:
    raw_csv: Path = ROOT / "data" / "raw" / "memes_labeled.csv"      # columns: text,label
    corpus_txt: Path = ROOT / "data" / "raw" / "hinglish_corpus.txt"  # unlabeled, for pretraining
    processed_dir: Path = ROOT / "data" / "processed"                 # train/val/test csv
    tokenizer_dir: Path = ROOT / "tokenizer" / "hinglish_bpe"
    pretrain_dir: Path = ROOT / "runs" / "pretrain"
    cls_dir: Path = ROOT / "runs" / "classifier"


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.1
    rope_base: float = 10000.0


@dataclass
class PretrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 50
    steps: int = 320          # recompute so it is ~1 epoch of your corpus
    save_every: int = 50
    mixed_precision: bool = True


@dataclass
class ClassifierConfig:
    max_len: int = 128        # memes are short
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 8
    patience: int = 2         # early stopping on val macro-F1
    use_class_weights: bool = True
    seed: int = 42


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    cls: ClassifierConfig = field(default_factory=ClassifierConfig)
    num_classes: int = len(CATEGORIES)


cfg = Config()
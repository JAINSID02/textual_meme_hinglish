# meme_llm/tokenizer/train_tokenizer.py
import sys
from pathlib import Path
from typing import List

from tokenizers import ByteLevelBPETokenizer, Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, SPECIAL_TOKENS, PAD_ID


def train_tokenizer(files: List[str] = None, out_dir=None, vocab_size=None):
    """Train a byte-level BPE on Hinglish text (corpus + meme texts)."""
    files = files or [str(cfg.paths.corpus_txt.with_name("hinglish_corpus_clean.txt"))]
    out_dir = Path(out_dir or cfg.paths.tokenizer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_size = vocab_size or cfg.model.vocab_size

    tok = ByteLevelBPETokenizer()
    tok.train(files=files, vocab_size=vocab_size, min_frequency=2,
              special_tokens=SPECIAL_TOKENS)

    # special tokens must get ids 0..4 in this exact order (pad must be 2)
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(t) == i, f"{t} got id {tok.token_to_id(t)}, expected {i}"
    assert tok.token_to_id("<pad>") == PAD_ID

    tok.save(str(out_dir / "tokenizer.json"))
    print(f"[tokenizer] vocab={tok.get_vocab_size()} saved to {out_dir}")


class HinglishTokenizer:
    """Thin wrapper used everywhere else: load, encode, decode."""

    def __init__(self, tokenizer_dir=None):
        path = Path(tokenizer_dir or cfg.paths.tokenizer_dir) / "tokenizer.json"
        self._tok = Tokenizer.from_file(str(path))
        self.pad_id = self._tok.token_to_id("<pad>")
        self.bos_id = self._tok.token_to_id("<s>")
        self.eos_id = self._tok.token_to_id("</s>")
        assert self.pad_id == PAD_ID, "tokenizer pad id does not match config.PAD_ID"

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)


if __name__ == "__main__":
    train_tokenizer()
    t = HinglishTokenizer()
    s = "bhai kya scene hai 😂 exam kal hai"
    ids = t.encode(s)
    print(len(ids), "tokens:", ids)
    print(t.decode(ids))
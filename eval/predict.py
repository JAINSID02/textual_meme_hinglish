# meme_llm/eval/predict.py
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg
from data.clean import clean_text
from tokenizer.train_tokenizer import HinglishTokenizer
from eval.metrics import load_classifier
from training.utils import get_device


class MemePredictor:
    """Load once, predict many times.
    Applies the SAME cleaning, truncation and EOS handling as training."""

    def __init__(self, ckpt_path=None, tokenizer_dir=None, device=None, max_len=None):
        self.device = device or get_device()
        ckpt_path = ckpt_path or Path(cfg.paths.cls_dir) / "model_best.pt"
        self.tok = HinglishTokenizer(tokenizer_dir)
        self.model, self.categories = load_classifier(ckpt_path, self.device)
        self.max_len = max_len or cfg.cls.max_len

    def _encode(self, text: str) -> List[int]:
        ids = self.tok.encode(clean_text(text))[: self.max_len - 1]
        return ids + [self.tok.eos_id]

    @torch.no_grad()
    def predict_proba(self, texts: List[str]) -> torch.Tensor:
        """Returns (N, n_classes) probabilities."""
        encoded = [self._encode(t) for t in texts]
        max_t = max(len(e) for e in encoded)
        x = torch.full((len(encoded), max_t), self.tok.pad_id, dtype=torch.long)
        for i, e in enumerate(encoded):
            x[i, : len(e)] = torch.tensor(e)
        logits = self.model(x.to(self.device))
        return F.softmax(logits, dim=-1).cpu()

    def predict(self, texts, top_k: int = 3, min_confidence: float = 0.0) -> List[Dict]:
        """texts: a string or a list of strings.
        Each result: {'text', 'label', 'confidence', 'uncertain', 'top': [(label, prob), ...]}.
        'uncertain' is True when the top probability is below min_confidence."""
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        probs = self.predict_proba(texts)

        results = []
        for text, p in zip(texts, probs):
            vals, idx = p.topk(min(top_k, len(self.categories)))
            top = [(self.categories[i], float(v)) for v, i in zip(vals, idx)]
            results.append({
                "text": text,
                "label": top[0][0],
                "confidence": top[0][1],
                "uncertain": top[0][1] < min_confidence,
                "top": top,
            })
        return results[0] if single else results


if __name__ == "__main__":
    predictor = MemePredictor()
    demo = [
        "bhai kal exam hai aur maine syllabus khola bhi nahi 😭",
        "jab crush ne seen karke reply nahi diya",
    ]
    for r in predictor.predict(demo, min_confidence=0.4):
        flag = "  (uncertain)" if r["uncertain"] else ""
        print(f"{r['text']}\n  -> {r['label']} ({r['confidence']:.2f}){flag}")
        print("     top:", ", ".join(f"{l} {p:.2f}" for l, p in r["top"]))
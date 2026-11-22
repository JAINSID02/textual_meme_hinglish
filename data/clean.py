# meme_llm/data/clean.py
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")      # 3+ repeated chars -> 2
SPACE_RE = re.compile(r"\s+")

# Small starter map of common Hinglish spelling variants -> one canonical form.
# Extend this as you inspect your data.
VARIANTS = {
    "kyaa": "kya", "kyon": "kyu", "kyun": "kyu", "kyu": "kyu",
    "nahi": "nahi", "nahin": "nahi", "nhi": "nahi", "nai": "nahi",
    "hai": "hai", "hain": "hai", "h": "hai",
    "bhai": "bhai", "bhaiya": "bhai", "bro": "bhai",
    "acha": "accha", "achha": "accha", "acchha": "accha",
    "mera": "mera", "meraa": "mera",
    "ho": "ho", "hoga": "hoga", "hogaa": "hoga",
}


def clean_text(text: str, normalize_variants: bool = True) -> str:
    """Clean one meme/corpus line. Emojis and Devanagari are kept on purpose."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)      # keep the word, drop '#'
    text = text.lower()
    text = REPEAT_RE.sub(r"\1\1", text)     # "kyaaaaa" -> "kyaa"
    if normalize_variants:
        text = " ".join(VARIANTS.get(w, w) for w in text.split())
    return SPACE_RE.sub(" ", text).strip()


def clean_labeled_csv(in_path=None, out_path=None) -> pd.DataFrame:
    """Clean memes_labeled.csv (columns: text,label); drop empties and exact duplicates."""
    in_path = Path(in_path or cfg.paths.raw_csv)
    out_path = Path(out_path or in_path.with_name("memes_clean.csv"))
    df = pd.read_csv(in_path)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset="text").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"[clean] {in_path.name}: {len(df)} rows -> {out_path}")
    return df


def clean_corpus(in_path=None, out_path=None) -> None:
    """Clean the unlabeled pretraining corpus, one example per line."""
    in_path = Path(in_path or cfg.paths.corpus_txt)
    out_path = Path(out_path or in_path.with_name("hinglish_corpus_clean.txt"))
    n = 0
    with open(in_path, encoding="utf-8") as fi, open(out_path, "w", encoding="utf-8") as fo:
        for line in fi:
            t = clean_text(line)
            if t:
                fo.write(t + "\n")
                n += 1
    print(f"[clean] corpus: {n} lines -> {out_path}")


if __name__ == "__main__":
    print(clean_text("Bhaiii kyaaaa scene hai!! 😂😂 https://x.com/abc @user #Exam_Stress"))
    # clean_labeled_csv()
    # clean_corpus()
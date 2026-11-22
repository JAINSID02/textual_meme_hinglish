# meme_llm/eval/baseline.py
import sys
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import cfg, CATEGORIES, LABEL2ID
from eval.metrics import summarize


def load_split(name):
    df = pd.read_csv(Path(cfg.paths.processed_dir) / f"{name}.csv")
    return df["text"].astype(str).tolist(), [LABEL2ID[l] for l in df["label"]]


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # Word n-grams capture phrases; char n-grams absorb Hinglish spelling variants
    # ("nahi"/"nhi"/"nahin") and emojis, which word tokens would miss.
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)

    # fit the vectorizers on TRAIN only (no peeking at val/test vocabulary)
    Xtr = hstack([word_vec.fit_transform(X_train), char_vec.fit_transform(X_train)]).tocsr()
    transform = lambda texts: hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()
    Xva, Xte = transform(X_val), transform(X_test)

    # pick C on the validation set, then report on test once
    best = None
    for C in [0.3, 1, 3, 10, 30]:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y_train)
        from sklearn.metrics import f1_score
        f1 = f1_score(y_val, clf.predict(Xva), average="macro", zero_division=0)
        print(f"C={C:<5} val macro-F1 = {f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)

    f1, C, clf = best
    print(f"\n[baseline] best C={C} (val macro-F1 {f1:.3f}). TEST results:\n")
    summarize(y_test, clf.predict(Xte), CATEGORIES)


if __name__ == "__main__":
    main()
"""Train Nuvio's emotion classifier on GoEmotions.

TF-IDF over word unigrams/bigrams plus character n-grams, fed to a class-balanced
logistic regression. The regularisation strength C is picked on the dev split;
the final numbers are reported once, on the untouched test split.

    bash scripts/download_goemotions.sh
    python -m training.train_emotion --data data/goemotions --out models/emotion/latest
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from training.goemotions import MODEL_LABELS, load_split


def build_model(c: float) -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                  max_features=60000, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
    return Pipeline([("tfidf", features), ("lr", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/goemotions", help="folder with train.tsv, dev.tsv, test.tsv")
    ap.add_argument("--out", default="models/emotion/latest")
    args = ap.parse_args()

    data = Path(args.data)
    x_tr, y_tr = load_split(data / "train.tsv")
    x_dev, y_dev = load_split(data / "dev.tsv")
    x_te, y_te = load_split(data / "test.tsv")
    print(f"train={len(x_tr)} dev={len(x_dev)} test={len(x_te)}")

    best_c, best_f1 = None, -1.0
    for c in (0.5, 1.0, 2.0, 4.0):
        f1 = f1_score(y_dev, build_model(c).fit(x_tr, y_tr).predict(x_dev), average="macro")
        print(f"C={c:<4} dev macro-F1={f1:.3f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1

    model = build_model(best_c).fit(x_tr + x_dev, y_tr + y_dev)
    pred = model.predict(x_te)
    report = classification_report(y_te, pred, labels=MODEL_LABELS, output_dict=True, zero_division=0)
    metrics = {
        "dataset": "GoEmotions (mapped to Nuvio labels)",
        "model": "TF-IDF (word 1-2 + char 3-5 grams) + LogisticRegression(class_weight=balanced)",
        "C": best_c,
        "test_size": len(y_te),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "macro_f1": round(f1_score(y_te, pred, average="macro"), 4),
        "per_class_f1": {k: round(report[k]["f1-score"], 4) for k in MODEL_LABELS},
        "trained_at": int(time.time()),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"[OK] saved {out / 'model.joblib'}")


if __name__ == "__main__":
    main()

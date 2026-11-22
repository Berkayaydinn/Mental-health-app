"""Load GoEmotions and map its 27 fine-grained emotions onto Nuvio's labels.

GoEmotions (Demszky et al., 2020, Apache-2.0) is ~58k Reddit comments, each tagged
with one or more of 27 emotions or "neutral". Nuvio only needs a handful of
labels, so we collapse them with the Ekman grouping the dataset authors publish:

    happy   <- joy, amusement, approval, excitement, gratitude, love, optimism,
               relief, pride, admiration, desire, caring
    sad     <- sadness, disappointment, embarrassment, grief, remorse
    anxious <- fear, nervousness
    angry   <- anger, annoyance, disapproval, disgust
    neutral <- neutral

The "surprise" group (surprise, realization, confusion, curiosity) has no Nuvio
counterpart and is dropped. "stressed" has no GoEmotions counterpart either, so
it is detected with keywords at inference time (see inference/emotion_inference.py).
A comment whose emotions map to more than one Nuvio label is ambiguous and skipped.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

GOEMOTIONS = (
    "admiration amusement anger annoyance approval caring confusion curiosity desire "
    "disappointment disapproval disgust embarrassment excitement fear gratitude grief joy "
    "love nervousness optimism pride realization relief remorse sadness surprise neutral"
).split()

TO_NUVIO = {
    **dict.fromkeys(["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
                     "relief", "pride", "admiration", "desire", "caring"], "happy"),
    **dict.fromkeys(["sadness", "disappointment", "embarrassment", "grief", "remorse"], "sad"),
    **dict.fromkeys(["fear", "nervousness"], "anxious"),
    **dict.fromkeys(["anger", "annoyance", "disapproval", "disgust"], "angry"),
    "neutral": "neutral",
}

MODEL_LABELS = ["angry", "anxious", "happy", "neutral", "sad"]


def map_row(label_ids: str) -> Optional[str]:
    """'14,19' -> 'anxious'; None if the row is dropped or ambiguous."""
    mapped = {TO_NUVIO.get(GOEMOTIONS[int(i)]) for i in label_ids.split(",")}
    mapped.discard(None)
    return mapped.pop() if len(mapped) == 1 else None


def load_split(path: str | Path) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            label = map_row(parts[1])
            if label:
                texts.append(parts[0])
                labels.append(label)
    return texts, labels

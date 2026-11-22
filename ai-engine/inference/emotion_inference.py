"""Emotion detection for chat messages.

Uses the trained classifier in models/emotion/latest when it is present and falls
back to keyword rules otherwise, so the app keeps working without the model file.
"Stressed" is not a GoEmotions category, so it is always detected with keywords
and takes precedence when stress words appear.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

ENGINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ENGINE_ROOT / "models" / "emotion" / "latest" / "model.joblib"

LABELS: Dict[str, int] = json.loads((ENGINE_ROOT / "data" / "label_mapping.json").read_text(encoding="utf-8"))

STRESS = re.compile(r"\b(stress\w*|overwhelm\w*|burn(ed|t)?\s?out|deadline\w*|too much on my plate|exhaust\w*|swamped)\b")
KEYWORDS = [
    ("anxious", re.compile(r"\b(anxious|anxiety|panic\w*|worr(y|ied|ying)|nervous|scared|afraid)\b")),
    ("sad", re.compile(r"\b(sad|down|depress\w*|lonely|cry\w*|hopeless|miserable)\b")),
    ("angry", re.compile(r"\b(angry|mad|furious|annoyed|hate|irritat\w*)\b")),
    ("happy", re.compile(r"\b(happy|great|good|awesome|glad|excited|grateful|calm)\b")),
]


@lru_cache(maxsize=1)
def _load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None
    import joblib  # imported lazily so the keyword fallback has no ML dependency
    return joblib.load(p)


def _result(emotion: str, confidence: float, source: str) -> Dict:
    return {"emotion": emotion, "label": LABELS[emotion], "confidence": round(confidence, 3), "source": source}


def classify(text: str, model_path: Optional[str | Path] = None) -> Dict:
    t = (text or "").lower()
    if STRESS.search(t):
        return _result("stressed", 0.9, "keywords")

    model = _load_model(str(model_path or DEFAULT_MODEL))
    if model is not None and t.strip():
        proba = model.predict_proba([text])[0]
        best = int(proba.argmax())
        return _result(str(model.classes_[best]), float(proba[best]), "model")

    for emotion, pattern in KEYWORDS:
        if pattern.search(t):
            return _result(emotion, 0.6, "keywords")
    return _result("neutral", 0.5, "keywords")


def predict_emotion(text: str) -> Dict:
    """Public entry point used by the backend."""
    return classify(text)

from pathlib import Path

import pytest

from inference import emotion_inference as ei

MODEL = ei.DEFAULT_MODEL


def test_stress_words_always_win():
    out = ei.classify("I'm so overwhelmed with deadlines this week")
    assert out["emotion"] == "stressed"
    assert out["label"] == ei.LABELS["stressed"]


def test_keyword_fallback_without_model(tmp_path: Path):
    missing = tmp_path / "nope.joblib"
    assert ei.classify("I feel so lonely and sad", model_path=missing)["emotion"] == "sad"
    assert ei.classify("I keep worrying about everything", model_path=missing)["emotion"] == "anxious"
    assert ei.classify("the bus was late", model_path=missing)["emotion"] == "neutral"
    assert ei.classify("", model_path=missing)["source"] == "keywords"


def test_every_label_is_known():
    for text in ["", "great day", "I hate this", "so sad", "panic attack", "deadline stress"]:
        out = ei.classify(text)
        assert out["emotion"] in ei.LABELS
        assert 0.0 <= out["confidence"] <= 1.0


@pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")
@pytest.mark.parametrize("text,expected", [
    ("I can't stop worrying about my exam tomorrow", "anxious"),
    ("thank you so much, this made my whole day", "happy"),
    ("I miss her so much it hurts", "sad"),
    ("this is ridiculous, stop wasting my time", "angry"),
])
def test_trained_model_on_clear_examples(text, expected):
    out = ei.classify(text)
    assert out["source"] == "model"
    assert out["emotion"] == expected

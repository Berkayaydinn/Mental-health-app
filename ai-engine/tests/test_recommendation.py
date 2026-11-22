import math

import pandas as pd

from inference.recommendation_logic import IBCFRecommender

RATINGS = pd.DataFrame([
    ("u1", "breathing_exercise", 5), ("u1", "mindfulness_video", 4),
    ("u2", "breathing_exercise", 3), ("u2", "journaling_prompt", 5),
    ("u3", "mindfulness_video", 2), ("u3", "journaling_prompt", 4),
], columns=["user_id", "item_id", "rating"])


def fitted():
    m = IBCFRecommender()
    m.fit(RATINGS)
    return m


def test_never_recommends_rated_items():
    recs = fitted().recommend("u1", k=10)
    assert {r.item_id for r in recs} == {"journaling_prompt"}
    assert all(math.isfinite(r.score) for r in recs)


def test_cold_start_returns_catalogue():
    recs = fitted().recommend("someone_new", k=3)
    assert len(recs) == 3


def test_emotion_reorders_suggestions():
    m = fitted()
    assert m.recommend("someone_new", k=1, emotion="anxious")[0].item_id == "breathing_exercise"
    top_sad = m.recommend("someone_new", k=1, emotion="sad")[0]
    assert top_sad.item_id == "journaling_prompt"
    assert "you seem sad" in top_sad.explanation


def test_unknown_emotion_is_ignored():
    m = fitted()
    plain = [r.item_id for r in m.recommend("someone_new", k=3)]
    assert [r.item_id for r in m.recommend("someone_new", k=3, emotion="bored")] == plain


def test_save_and_load_roundtrip(tmp_path):
    m = fitted()
    m.save(tmp_path)
    loaded = IBCFRecommender.load(tmp_path)
    assert [r.item_id for r in loaded.recommend("u2")] == [r.item_id for r in m.recommend("u2")]

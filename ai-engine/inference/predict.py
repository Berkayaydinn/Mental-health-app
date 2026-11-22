"""Command-line and Python entry points for the AI engine.

    python -m inference.predict --task emotion --text "I can't stop worrying about tomorrow"
    python -m inference.predict --task recommend --user u1 --emotion anxious
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from inference.emotion_inference import predict_emotion
from inference.recommendation_logic import IBCFRecommender
from inference.schemas import RecItem, RecommendResponse

ENGINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REC_MODEL_DIR = ENGINE_ROOT / "models" / "recommendation" / "latest"


def _resolve(model_dir: str | Path) -> Path:
    """Relative model paths are taken from the ai-engine folder, not the caller's working directory."""
    p = Path(model_dir)
    return p if p.is_absolute() else ENGINE_ROOT / p


def recommend(user_id: str, k: int = 10, model_dir: str | Path = DEFAULT_REC_MODEL_DIR,
              emotion: Optional[str] = None) -> RecommendResponse:
    model = IBCFRecommender.load(_resolve(model_dir))
    recs = model.recommend(user_id=user_id, k=k, emotion=emotion)
    return RecommendResponse(
        user_id=user_id,
        results=[RecItem(item_id=r.item_id, score=r.score, explanation=r.explanation) for r in recs],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="recommend", choices=["recommend", "emotion"])
    ap.add_argument("--user", help="user id (recommend)")
    ap.add_argument("--text", help="message text (emotion)")
    ap.add_argument("--emotion", help="optional detected emotion to re-rank recommendations")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default=str(DEFAULT_REC_MODEL_DIR))
    args = ap.parse_args()

    if args.task == "emotion":
        print(json.dumps(predict_emotion(args.text or ""), indent=2))
    else:
        if not args.user:
            ap.error("--user is required for --task recommend")
        print(recommend(args.user, args.k, args.model, args.emotion).model_dump_json(indent=2))


if __name__ == "__main__":
    main()

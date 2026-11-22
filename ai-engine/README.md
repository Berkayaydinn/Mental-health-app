# Nuvio AI engine

The machine-learning side of Nuvio: it reads how a user feels from a chat message and suggests a coping activity that fits.

```
chat message ──► emotion classifier ──► emotion ──┐
                                                   ├──► ranked activities + explanation
user's past ratings ──► item-based CF ─────────────┘
```

## Emotion detection

**Labels:** happy · sad · anxious · angry · neutral · stressed

A TF-IDF model (word 1–2-grams plus character 3–5-grams) feeds a class-balanced logistic regression. It is trained on [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Demszky et al., 2020, Apache-2.0), about 58k Reddit comments. The 27 GoEmotions categories are collapsed onto Nuvio's labels using the dataset authors' Ekman grouping (see `training/goemotions.py`). Comments that map to more than one label are skipped. "Stressed" has no GoEmotions category, so it is caught with keyword rules, and those rules win when stress words appear. If the model file is missing, the engine falls back to keyword rules for every label.

**Results on the held-out GoEmotions test split (4,690 comments):**

| | Accuracy | Macro-F1 |
|---|---|---|
| Original keyword rules | 0.417 | 0.236 |
| **Trained classifier** | **0.705** | **0.651** |

Per-class F1: happy 0.80 · neutral 0.70 · anxious 0.63 · sad 0.57 · angry 0.57. Regularisation (C) was chosen on the dev split and the test split was used once. Full numbers are in `models/emotion/latest/metrics.json`.

## Activity recommendations

Item-based collaborative filtering with cosine similarity over mean-centred ratings (`inference/recommendation_logic.py`). When an emotion is known, a per-emotion fit table leads the ranking and the CF score breaks ties. An anxious user is steered toward a breathing exercise first, and a sad user toward a journaling prompt. Every suggestion carries a one-line explanation.

The ratings in `data/sample_inputs.json` are demo data (2 users, 3 activities), so the HitRate/MRR figures in `evaluation/results.json` don't mean anything yet. The pipeline is ready for real ratings via `--ratings ratings.csv`.

## Usage

```bash
pip install -r requirements.txt -r requirements-dev.txt

python -m inference.predict --task emotion --text "I can't stop worrying about my exam"
python -m inference.predict --task recommend --user u1 --emotion anxious

python -m pytest -v                       # 13 tests, also run in CI
```

Retrain the emotion model:

```bash
bash scripts/download_goemotions.sh
python -m training.train_emotion --data data/goemotions --out models/emotion/latest
python -m evaluation.evaluate_models
```

## Layout

```
ai-engine/
├── inference/    emotion_inference.py, recommendation_logic.py, predict.py (CLI + backend entry point)
├── training/     train_emotion.py, goemotions.py (label mapping), train_recommendation.py
├── evaluation/   evaluate_models.py → results.json
├── models/       emotion/latest (classifier + metrics), recommendation/latest
├── scripts/      download_goemotions.sh
└── tests/        pytest suite
```

## Limits

- GoEmotions is Reddit text, not therapy-app chat. Short, informal messages are covered better than long clinical descriptions.
- Angry and sad are the hardest classes (F1 ≈ 0.57) and get confused with each other.
- The engine never diagnoses. It only picks a supportive activity.

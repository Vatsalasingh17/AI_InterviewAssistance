# modules/emotion_text.py
from transformers import pipeline

emotion_analyzer = pipeline("text-classification",
                            model="j-hartmann/emotion-english-distilroberta-base",
                            return_all_scores=True)

def analyze_text_emotion(text: str):
    results = emotion_analyzer(text)
    top = sorted(results[0], key=lambda x: x["score"], reverse=True)[0]
    return top["label"], results[0]

# This module loads the pre-trained emotion analysis model using the hugging face transformer and exposes a function analyze_text_emotion that returns the emotion which has the highest value or the top most emotion as well as the scores for other emotions
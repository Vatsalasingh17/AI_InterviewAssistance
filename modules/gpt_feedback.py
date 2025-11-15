# modules/gpt_feedback.py
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_gpt_feedback(question, answer, emotion, audio_metrics):
    prompt = f"""
    Interview Question: {question}
    Candidate Answer: {answer}
    Detected Emotion: {emotion}
    Audio Metrics: {audio_metrics}

    Give detailed feedback (max 150 words) covering:
    1. Tone and confidence
    2. Clarity and structure
    3. Content relevance
    4. Improvements
    5. A numeric overall score (1–10)
    """
    response = client.chat.completions.create(       #client.chat.completions.create() sends a chat request.
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    #The GPT model returns a structured response in the response object.
    return response.choices[0].message.content

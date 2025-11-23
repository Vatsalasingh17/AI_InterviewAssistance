# modules/speech_to_text.py

import whisper  # Imports OpenAI's Whisper speech-to-text library. This library converts audio → text using neural networks.

# ✅ Load model once at import time so it doesn't reload on every transcription call (saves ~2–5 seconds).
model = whisper.load_model("base")   # Options: "tiny", "base", "small", "medium", "large" (larger = more accurate but slower)

def transcribe_audio(path: str) -> str:
    """
    Transcribes an audio file using OpenAI Whisper.

    Args:
        path (str): Path to the audio file.

    Returns:
        str: The transcribed text.
    """
    try:
        # Whisper handles:
        # - automatic language detection
        # - noisy audio
        # - accents
        # - long recordings (splits into segments)
        
        result = model.transcribe(path)  # Performs full transcription (timestamped segments + detected language + text)

        # Extract only the final combined text from Whisper’s full output dictionary.
        # .get("text", "") avoids KeyErrors if Whisper returns unexpected output.
        # .strip() cleans trailing whitespace or newline.
        return result.get("text", "").strip()

    except Exception as e:
        # Catch and format any unexpected errors (bad path, unsupported format, etc.)
        return f"⚠️ Transcription error: {str(e)}"

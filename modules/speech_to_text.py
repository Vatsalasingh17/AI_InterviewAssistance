# modules/speech_to_text.py
import whisper  #Imports OpenAI's Whisper speech-to-text library.This library converts audio → text using neural networks.

# ✅ Load model once (not every time!)
model = whisper.load_model("base")   # change to "small" or "large" if needed

def transcribe_audio(path: str) -> str:
    """
    Transcribes an audio file using OpenAI Whisper.

    Args:
        path (str): Path to the audio file.

    Returns:
        str: The transcribed text.
    """
    try:
        result = model.transcribe(path)
        return result.get("text", "").strip()  #Extracts the "text" field safely 
    #..get("text", "") → avoids KeyErrors if something goes wrong #
    #..strip() → removes extra spaces or newlines.
    except Exception as e:
        return f"⚠️ Transcription error: {str(e)}"
    

## Whisper automatically:

## detects language

## handles noise

## handles different accents

## splits the audio into segments
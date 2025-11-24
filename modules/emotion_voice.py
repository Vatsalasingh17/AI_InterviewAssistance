# modules/emotion_voice.py

import numpy as np
import soundfile as sf              # soundfile → safe and reliable for reading audio files
import librosa                      # librosa → for pitch, energy, tempo, and other audio features

def analyze_voice_emotion(audio_path):
    """
    Analyze an audio file and extract simple voice-related features
    that can be used as indicators of emotional state.
    
    Parameters:
        audio_path (str): Path to the audio file to analyze.
    
    Returns:
        dict: Dictionary containing pitch, energy, and tempo values.
    """

    # Load audio using soundfile (more robust and safer than librosa.load)
    y, sr = sf.read(audio_path)

    # If audio has multiple channels (e.g., stereo), convert to mono by averaging
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Convert audio data to float for signal processing operations
    y = y.astype(float)

    # -----------------------------
    # Feature Extraction
    # -----------------------------

    # Estimate pitch using YIN algorithm
    # fmin/fmax define expected voice pitch range (50–400 Hz)
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=400, sr=sr))

    # Calculate RMS energy (indicator of loudness)
    energy = np.mean(librosa.feature.rms(y=y))

    # Estimate tempo (beats per minute)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Return rounded feature values for readability
    return {
        "pitch": round(float(pitch), 2),
        "energy": round(float(energy), 4),
        "tempo": round(float(tempo), 2)
    }

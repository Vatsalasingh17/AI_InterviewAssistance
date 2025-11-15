# modules/emotion_voice.py
import numpy as np
import soundfile as sf  #safe/well-supported audio loading library
import librosa #librosa → audio signal processing (pitch, tempo, rm librosa is a very popular python library for audio and music analysis for finding out the pitch energy and tempo in the audio

def analyze_voice_emotion(audio_path):
    # Load audio safely using soundfile instead of librosa.load
    y, sr = sf.read(audio_path)

    # Ensure mono audio
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Convert to float
    y = y.astype(float)

    # Extract basic audio features
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=400, sr=sr))
    energy = np.mean(librosa.feature.rms(y=y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return {
        "pitch": round(float(pitch), 2),
        "energy": round(float(energy), 4),
        "tempo": round(float(tempo), 2)
    }

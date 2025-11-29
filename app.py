# app.py
from pydub import AudioSegment     #pydub for analysing the audio processing it and converting it into desired wav(waveform audio file format)  
AudioSegment.converter = r"C:\Users\Vatsala Singh\Downloads\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
AudioSegment.ffmpeg = AudioSegment.converter #convert the audio to wav form
AudioSegment.ffprobe = r"C:\Users\Vatsala Singh\Downloads\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin\ffprobe.exe"

import streamlit as st # for creating the UI
import plotly.graph_objects as go #Used for creating the graphs,charts
import plotly.express as px
import re  #python's regex(regular expression) library for extracting of text,finding the text and replacing the text
import os

from modules import database, analysis
from modules.speech_to_text import transcribe_audio
from modules.emotion_text import analyze_text_emotion
from modules.emotion_voice import analyze_voice_emotion
from modules.gpt_feedback import get_gpt_feedback

st.set_page_config(page_title="AI Interview Coach 2.0", layout="wide")
st.title("🎤 Advanced AI Interview Coach") # for setting the title of the page

# Initialize database
database.init_db() # ensure that database exists before storing the interview attempts.

def convert_to_wav(input_path, output_path):#conversion of audio file from mp3 or m4a to wav input_path-the original file path output_path the file format in wav form
    sound = AudioSegment.from_file(input_path)
    sound.export(output_path, format="wav")

question = st.text_input("💬 Enter your interview question:")
audio = st.file_uploader("🎧 Upload your recorded answer", type=["wav", "mp3", "m4a"])

if st.button("Analyze") and audio:
    with st.spinner("Processing audio..."):
        ext = audio.name.split(".")[-1].lower()
        temp_path_raw = f"temp_uploaded_audio.{ext}"
        with open(temp_path_raw, "wb") as f:
            f.write(audio.read())

        if ext in ["mp3", "m4a"]:
            temp_path_wav = "temp_uploaded_audio.wav"
            convert_to_wav(temp_path_raw, temp_path_wav)
            audio_path = temp_path_wav
        else:
            audio_path = temp_path_raw

        # 1. Transcribe
        text = transcribe_audio(audio_path) # for converting speech to text

        # 2. Emotion analysis (text + voice)
        text_emotion, emotion_scores = analyze_text_emotion(text)
        voice_stats = analyze_voice_emotion(audio_path) # pitch,tempo,energy,jitter,shimmer

        # 3. GPT feedback
        feedback = get_gpt_feedback(question, text, text_emotion, voice_stats)

    st.subheader("🗣️ Transcript")
    st.write(text)

    st.subheader("🎭 Emotion Analysis (Text)")
    st.write(f"Dominant Emotion: **{text_emotion}**")

    # for creating the Bar Graphs
    fig = go.Figure(
        go.Bar(
            x=[e["label"] for e in emotion_scores],
            y=[e["score"] for e in emotion_scores],
            marker_color="lightskyblue",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎵 Voice Metrics")
    st.json(voice_stats)

    st.subheader("🧠 AI Feedback")
    st.markdown(feedback)

    # extract numeric score from feedback (best-effort)
    match = re.search(r"(\d+(\.\d+)?)", feedback)
    gpt_score = float(match.group(1)) if match else 0

    # for inserting the session created into the database 

    database.insert_session(text_emotion, voice_stats, gpt_score, gpt_feedback=feedback)

    # load all the previous sessions
    df = database.load_sessions()

    st.divider()
    st.subheader("📊 Your Progress Over Time")

    if not df.empty:
        st.dataframe(df)

        summary = analysis.generate_report(df)
        st.markdown(summary)

        fig_trend = px.line(
            df.sort_values("timestamp"),
            x="timestamp",
            y="gpt_score",
            markers=True,
            title="Interview Performance Trend",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. Score", round(df["gpt_score"].mean(), 2))
        with col2:
            st.metric("Avg. Energy", round(df["energy"].mean(), 3) if "energy" in df else 0)
        with col3:
            st.metric("Avg. Tempo", round(df["tempo"].mean(), 2) if "tempo" in df else 0)
    else:
        st.info("No previous interview sessions yet. Record one to begin tracking progress!")

    # cleanup temps
    for temp in [temp_path_raw, "temp_uploaded_audio.wav"]:
        if os.path.exists(temp):
            os.remove(temp)

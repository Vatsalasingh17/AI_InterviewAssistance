# modules/database.py
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

DB_PATH = Path("sessions.json") # defines the file that will store all the records of the session in json format.

def init_db(path: str | Path = DB_PATH): # it makes sure that db file always exists before writing into it.
    p = Path(path)
    if not p.exists():
        p.write_text("[]")

def insert_session(text_emotion: str, audio_stats: dict, gpt_score: float, gpt_feedback: str = ""):


    init_db(DB_PATH) #makes sure that session.json exists before opening it.
    with DB_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "text_emotion": text_emotion,
        "pitch": audio_stats.get("pitch", 0),
        "energy": audio_stats.get("energy", 0),
        "tempo": audio_stats.get("tempo", 0),
        "jitter": audio_stats.get("jitter", 0),
        "shimmer": audio_stats.get("shimmer", 0),
        "pauses": audio_stats.get("pauses", 0),
        "gpt_feedback": gpt_feedback,
        "gpt_score": float(gpt_score)
    }

    data.append(record)
    with DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_sessions(path: str | Path = DB_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    
    # if file exist but it is empty then return empty data frame.
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return pd.DataFrame()
    

    df = pd.DataFrame(data) #convert the json list (data) to panda dataframe (df).


    # normalize types
    df["gpt_score"] = pd.to_numeric(df["gpt_score"], errors="coerce").fillna(0) #converting gpt scores to numeric,invalid values become NaN and repalces with zero.
    df["timestamp"] = pd.to_datetime(df["timestamp"])  #converting the timestamp string to panda datatime object.


    # keep a consistent column order
    cols = ["timestamp", "text_emotion", "pitch", "energy", "tempo", "jitter", "shimmer", "pauses", "gpt_feedback", "gpt_score"]
    return df.loc[:, [c for c in cols if c in df.columns]]

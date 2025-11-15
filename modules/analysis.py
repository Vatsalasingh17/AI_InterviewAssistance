# modules/analysis.py
import pandas as pd
import numpy as np

def calculate_trends(df: pd.DataFrame):
    """Return summary stats and simple linear trend on gpt_score."""
    if df is None or df.empty:
        return {
            "average_score": 0,
            "trend": "No data",
            "score_variance": 0,
            "score_volatility": 0
        }

    if len(df) < 2:
        return {
            "average_score": float(df["gpt_score"].mean()),
            "trend": "Not enough data",
            "score_variance": 0,
            "score_volatility": 0
        }

    slope = np.polyfit(range(len(df)), df["gpt_score"], 1)[0]  # polyfit function return two values [slope,intercept]  [0] gives us slope.
    trend = "Improving" if slope > 0 else ("Declining" if slope < 0 else "Stable")
    variance = float(np.var(df["gpt_score"]))   # variance means how much your scores are spread out if gpt scores are close together then low variance and if gpt scores are far away then high variance
    volatility = float(np.std(df["gpt_score"])) # (volatility) standard deviation which is the square root of variance.

    return {
        "average_score": round(df["gpt_score"].mean(), 2),
        "trend": trend,
        "score_variance": round(variance, 3),
        "score_volatility": round(volatility, 3)
    }

def generate_report(df: pd.DataFrame) -> str:
    trends = calculate_trends(df)

    def safe_avg(column):
        return round(df[column].mean(), 4) if column in df and not df[column].isna().all() else 0  #“If the column exists AND it is not completely empty,return the average of that column (rounded to 4 decimals).Otherwise,return 0.”

    report = f"""
**Performance Summary**

Scores
- Average Score: {trends['average_score']}
- Trend: {trends['trend']}
- Score Variance: {trends['score_variance']}
- Score Volatility: {trends['score_volatility']}

Audio Features
- Avg Pitch: {safe_avg("pitch")}
- Avg Energy: {safe_avg("energy")}
- Avg Tempo: {safe_avg("tempo")}
- Avg Jitter: {safe_avg("jitter")}
- Avg Shimmer: {safe_avg("shimmer")}
- Avg Pauses: {safe_avg("pauses")}

Other
- Sessions Analyzed: {len(df)}
"""
    return report

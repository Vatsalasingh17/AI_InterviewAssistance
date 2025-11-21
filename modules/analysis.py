# modules/analysis.py
import pandas as pd
import numpy as np

def calculate_trends(df: pd.DataFrame):
    """
    Compute summary statistics and a simple linear trend for the 'gpt_score' column.

    Returns a dictionary containing:
    - average_score: Mean GPT score
    - trend: 'Improving', 'Declining', or 'Stable' based on slope of linear fit
    - score_variance: Variance of GPT scores
    - score_volatility: Standard deviation of GPT scores

    Handles edge cases such as:
    - Empty DataFrame
    - DataFrames with fewer than two rows (insufficient for slope calculation)
    """
    # Handle empty or missing data
    if df is None or df.empty:
        return {
            "average_score": 0,
            "trend": "No data",
            "score_variance": 0,
            "score_volatility": 0
        }

    # If only a single score exists, slope cannot be computed
    if len(df) < 2:
        return {
            "average_score": float(df["gpt_score"].mean()),
            "trend": "Not enough data",
            "score_variance": 0,
            "score_volatility": 0
        }

    # Fit a linear regression line to determine score trend
    # polyfit returns [slope, intercept]; [0] extracts slope
    slope = np.polyfit(range(len(df)), df["gpt_score"], 1)[0]

    # Interpret slope direction
    trend = "Improving" if slope > 0 else ("Declining" if slope < 0 else "Stable")

    # Variance and standard deviation measure score spread/volatility
    variance = float(np.var(df["gpt_score"]))
    volatility = float(np.std(df["gpt_score"]))

    return {
        "average_score": round(df["gpt_score"].mean(), 2),
        "trend": trend,
        "score_variance": round(variance, 3),
        "score_volatility": round(volatility, 3)
    }

def generate_report(df: pd.DataFrame) -> str:
    """
    Generate a formatted performance report including:
    - Score trends (from calculate_trends)
    - Average values of audio features if available

    Missing or fully-empty columns are handled safely by safe_avg().
    """
    trends = calculate_trends(df)

    def safe_avg(column):
        """
        Safely compute the average of a column:
        - Returns 0 if the column does not exist or contains only NaN values.
        - Otherwise, returns the mean rounded to 4 decimal places.
        """
        return round(df[column].mean(), 4) if column in df and not df[column].isna().all() else 0

    # Build the multi-section report
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

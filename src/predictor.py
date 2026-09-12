"""Recursive multi-output forecasts produced by the saved Random Forest."""
from pathlib import Path
import joblib
import pandas as pd
from data_generator import INTERVAL_MINUTES, METRICS
from feature_engineering import add_features


def forecast(history: pd.DataFrame, model_path: Path, hours: int) -> pd.DataFrame:
    artifact = joblib.load(model_path)
    working = history[["timestamp", *METRICS]].copy().reset_index(drop=True)
    working["timestamp"] = pd.to_datetime(working["timestamp"])
    future = []
    for _ in range(hours * (60 // INTERVAL_MINUTES)):
        timestamp = working["timestamp"].iloc[-1] + pd.Timedelta(minutes=INTERVAL_MINUTES)
        candidate = pd.concat([working, pd.DataFrame([{"timestamp": timestamp, **{m: 0.0 for m in METRICS}}])], ignore_index=True)
        values = artifact["model"].predict(add_features(candidate).iloc[[-1]][artifact["feature_columns"]])[0]
        record = {"timestamp": timestamp, **dict(zip(METRICS, values))}
        working = pd.concat([working, pd.DataFrame([record])], ignore_index=True)
        future.append(record)
    return pd.DataFrame(future)

"""Lag and calendar feature construction for autoregressive forecasting."""
import pandas as pd
from data_generator import METRICS

LAGS = (1, 2, 3, 4, 8)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    for metric in METRICS:
        for lag in LAGS:
            data[f"{metric}_lag_{lag}"] = data[metric].shift(lag)
    return data


def feature_columns() -> list[str]:
    return ["hour", "day_of_week"] + [f"{metric}_lag_{lag}" for metric in METRICS for lag in LAGS]

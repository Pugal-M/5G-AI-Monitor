"""Train and iteratively forecast 5G latency, throughput, and jitter."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from data_generator import METRICS

LAGS = (1, 2, 3, 6, 12)


def _features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().reset_index(drop=True)
    for metric in METRICS:
        for lag in LAGS:
            frame[f"{metric}_lag_{lag}"] = frame[metric].shift(lag)
    return frame


def train(data: pd.DataFrame, path: Path) -> float:
    framed = _features(data).dropna()
    feature_columns = [c for c in framed if "_lag_" in c]
    split = int(len(framed) * 0.8)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(framed[feature_columns].iloc[:split], framed[METRICS].iloc[:split])
    prediction = model.predict(framed[feature_columns].iloc[split:])
    mae = mean_absolute_error(framed[METRICS].iloc[split:], prediction)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feature_columns}, path)
    return mae


def forecast(history: pd.DataFrame, path: Path, hours: int) -> pd.DataFrame:
    artifact = joblib.load(path)
    model, columns = artifact["model"], artifact["features"]
    result = history[METRICS].copy().reset_index(drop=True)
    forecasts = []
    for _ in range(hours * 12):
        row = _features(result).iloc[-1:][columns]
        values = model.predict(row)[0]
        next_row = dict(zip(METRICS, values))
        result = pd.concat([result, pd.DataFrame([next_row])], ignore_index=True)
        forecasts.append(next_row)
    return pd.DataFrame(forecasts)

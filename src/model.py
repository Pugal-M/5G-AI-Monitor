"""Random Forest model training and chronological evaluation."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_generator import METRICS
from feature_engineering import add_features, feature_columns


def train_model(history: pd.DataFrame, output_path: Path) -> dict:
    featured = add_features(history).dropna().reset_index(drop=True)
    columns = feature_columns()
    split = int(len(featured) * 0.8)
    forest = RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)
    forest.fit(featured[columns].iloc[:split], featured[METRICS].iloc[:split])
    predicted, actual = forest.predict(featured[columns].iloc[split:]), featured[METRICS].iloc[split:]
    metrics = {metric: {"mae": round(float(mean_absolute_error(actual[metric], predicted[:, i])), 3),
                        "rmse": round(float(np.sqrt(mean_squared_error(actual[metric], predicted[:, i]))), 3),
                        "r2": round(float(r2_score(actual[metric], predicted[:, i])), 3)}
               for i, metric in enumerate(METRICS)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": forest, "feature_columns": columns, "metrics": metrics}, output_path)
    return metrics

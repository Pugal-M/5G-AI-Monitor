"""Deterministic threshold checks; the LLM never performs these calculations."""
import pandas as pd

THRESHOLDS = {"latency_ms": 50.0, "throughput_mbps": 250.0, "jitter_ms": 10.0}


def calculate_severity(violation_types: int) -> str:
    return ["LOW", "MEDIUM", "HIGH", "CRITICAL"][min(violation_types, 3)]


def check_thresholds(predictions: pd.DataFrame) -> dict:
    anomalies, flags = [], pd.Series(False, index=predictions.index)
    for metric, condition in [("latency_ms", "greater_than"), ("throughput_mbps", "less_than"), ("jitter_ms", "greater_than")]:
        failed = predictions[metric] > THRESHOLDS[metric] if condition == "greater_than" else predictions[metric] < THRESHOLDS[metric]
        flags |= failed
        for idx in predictions.index[failed]:
            anomalies.append({"timestamp": str(predictions.loc[idx, "timestamp"]), "metric": metric,
                              "value": round(float(predictions.loc[idx, metric]), 2),
                              "threshold": THRESHOLDS[metric], "condition": condition})
    kinds = len({item["metric"] for item in anomalies})
    return {
            "status": "ANOMALY" if anomalies else "NORMAL",
            "severity": calculate_severity(kinds),
            "anomalies": anomalies,
            "anomalous_predictions": int(flags.sum()),
            "maximum_latency": round(float(predictions["latency_ms"].max()), 2),
            "minimum_throughput": round(float(predictions["throughput_mbps"].min()), 2),
            "maximum_jitter": round(float(predictions["jitter_ms"].max()), 2)
        }

def get_network_status(predictions: pd.DataFrame) -> dict:
    return check_thresholds(predictions)

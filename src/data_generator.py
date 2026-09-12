"""Reproducible, correlated 5G telemetry generation at 15-minute intervals."""
from pathlib import Path
import numpy as np
import pandas as pd

INTERVAL_MINUTES = 15
SAMPLES_PER_HOUR = 60 // INTERVAL_MINUTES
METRICS = ["latency_ms", "throughput_mbps", "jitter_ms"]


def generate_normal_data(days: int = 21, seed: int = 42) -> pd.DataFrame:
    """Generate healthy traffic with daily load cycles and correlated metrics."""
    rng = np.random.default_rng(seed)
    count = days * 24 * SAMPLES_PER_HOUR
    timestamp = pd.date_range("2026-01-01", periods=count, freq=f"{INTERVAL_MINUTES}min")
    phase = (np.arange(count) % (24 * SAMPLES_PER_HOUR)) / (24 * SAMPLES_PER_HOUR)
    load = 0.28 + 0.55 * np.maximum(np.sin(2 * np.pi * (phase - 0.25)), 0)
    load += rng.normal(0, 0.035, count)
    frame = pd.DataFrame({
        "timestamp": timestamp,
        "latency_ms": 15 + 24 * load + rng.normal(0, 1.2, count),
        "throughput_mbps": 455 - 145 * load + rng.normal(0, 8, count),
        "jitter_ms": 1.8 + 4.0 * load + rng.normal(0, 0.35, count),
        "is_anomaly": 0, "anomaly_type": "normal",
    })
    frame[METRICS] = frame[METRICS].clip(lower=0).round(3)
    return frame


def generate_anomaly_data(days: int = 21, seed: int = 42) -> pd.DataFrame:
    """Generate traffic containing recurring degradation windows."""
    frame = generate_normal_data(days, seed).copy()
    rng = np.random.default_rng(seed + 1)
    window = 8
    # Daily evening incidents make repeating degradation patterns learnable
    # from lag and calendar features; event types rotate across days.
    starts = range(20 * SAMPLES_PER_HOUR, len(frame) - window, 24 * SAMPLES_PER_HOUR)
    event_types = ["latency_spike", "throughput_degradation", "jitter_spike", "combined"]
    for number, start in enumerate(starts):
        event = "combined" if number == days - 1 else event_types[number % len(event_types)]
        index = frame.index[start:start + window]
        if event in ("latency_spike", "combined"):
            frame.loc[index, "latency_ms"] += rng.uniform(38, 58)
        if event in ("throughput_degradation", "combined"):
            frame.loc[index, "throughput_mbps"] *= rng.uniform(0.40, 0.58)
        if event in ("jitter_spike", "combined"):
            frame.loc[index, "jitter_ms"] += rng.uniform(9, 15)
        frame.loc[index, "is_anomaly"] = 1
        frame.loc[index, "anomaly_type"] = event
    frame[METRICS] = frame[METRICS].round(3)
    return frame


def save_data(output_dir: Path, days: int = 21) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_normal_data(days).to_csv(output_dir / "normal_5g.csv", index=False)
    generate_anomaly_data(days).to_csv(output_dir / "anomaly_5g.csv", index=False)

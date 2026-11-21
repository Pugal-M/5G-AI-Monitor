"""End-to-end 5G pipeline: telemetry -> Random Forest -> AI Harness."""
import argparse
from pathlib import Path

import pandas as pd

from data_generator import save_data
from model import train_model
from predictor import forecast
from agents.ai_harness import AIHarness

ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(scenario: str, regenerate: bool = False, use_ollama: bool = True) -> dict:
    if scenario not in {"normal", "anomaly"}:
        raise ValueError("scenario must be 'normal' or 'anomaly'")

    data_dir = ROOT / "data"
    model_path = ROOT / "models" / "random_forest.pkl"
    normal_data = data_dir / "normal_5g.csv"

    if regenerate or not normal_data.exists():
        save_data(data_dir)

    source = data_dir / ("normal_5g.csv" if scenario == "normal" else "anomaly_5g.csv")
    history = pd.read_csv(source)

    # Train/evaluate against the selected scenario's own chronological history.
    evaluation = train_model(history, model_path)
    horizon = 24 if scenario == "normal" else 2

    if scenario == "anomaly":
        # Forecast while the final injected incident is underway so the model
        # receives the observed degradation in its lag features.
        event_rows = history.index[history["is_anomaly"] == 1]
        if len(event_rows) >= 5:
            final_event = event_rows[event_rows >= event_rows[-1] - 7]
            if len(final_event) >= 5:
                history = history.loc[:final_event[4]]

    predictions = forecast(history, model_path, horizon)
    predictions.to_csv(data_dir / f"predictions_{scenario}.csv", index=False)

    # The harness now owns anomaly detection, RCA routing, LLM explanation,
    # and a trace of every stage. The prediction model remains unchanged.
    orchestration = AIHarness().run(
        predictions,
        forecast_horizon=f"{horizon} hours",
        use_ollama=use_ollama,
    )

    return {
        "predictions": predictions,
        "status": orchestration["status"],
        "evaluation": evaluation,
        "horizon": horizon,
        "agent_text": orchestration["agent_text"],
        "root_cause_analysis": orchestration["root_cause_analysis"],
        "harness_trace": orchestration["harness_trace"],
        "run_id": orchestration["run_id"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 5G AI monitoring pipeline.")
    parser.add_argument("--scenario", choices=("normal", "anomaly"), default="normal")
    parser.add_argument("--generate", action="store_true", help="Regenerate synthetic telemetry.")
    parser.add_argument("--no-ollama", action="store_true", help="Disable Qwen/Ollama explanations; use deterministic evidence only.")
    args = parser.parse_args()

    result = run_pipeline(args.scenario, args.generate, use_ollama=not args.no_ollama)
    print(f"RUN ID: {result['run_id']}")
    print(f"STATUS: {result['status']['status']}")
    print(f"PREDICTION HORIZON: {result['horizon']} HOURS")
    print("MODEL EVALUATION:", result["evaluation"])
    print("NETWORK STATUS:", result["status"])
    print("AI HARNESS TRACE:", result["harness_trace"])
    print(result["agent_text"])

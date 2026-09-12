"""End-to-end pipeline callable by the CLI and Streamlit dashboard."""
import argparse
from pathlib import Path
import pandas as pd
from data_generator import save_data
from model import train_model
from predictor import forecast
from anomaly_agent import check_thresholds
from ollama_agent import explain
from agents.root_cause_agent import RootCauseAgent

ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(scenario: str, regenerate: bool = False, use_ollama: bool = True) -> dict:
    data_dir, model_path = ROOT / "data", ROOT / "models" / "random_forest.pkl"
    if regenerate or not (data_dir / "normal_5g.csv").exists():
        save_data(data_dir)
    source = data_dir / ("normal_5g.csv" if scenario == "normal" else "anomaly_5g.csv")
    history = pd.read_csv(source)
    print("\n==============================")
    print("HISTORY COLUMNS")
    print("==============================")
    print(history.columns.tolist())

    print("\n==============================")
    print("LAST ROW")
    print("==============================")
    print(history.tail(1).to_dict("records"))
    # Each scenario is genuinely modelled from its own chronological history.
    evaluation = train_model(history, model_path)
    horizon = 24 if scenario == "normal" else 2
    if scenario == "anomaly":
        # Forecast while the final injected incident is underway, as a live
        # monitor would, so its lag values carry the observed degradation.
        event_rows = history.index[history["is_anomaly"] == 1]
        final_event = event_rows[event_rows >= event_rows[-1] - 7]
        history = history.loc[:final_event[4]]
    predictions = forecast(history, model_path, horizon)
    predictions.to_csv(data_dir / f"predictions_{scenario}.csv", index=False)
    status = check_thresholds(predictions)
    violated = {item["metric"] for item in status["anomalies"]}
    summary_values = {"latency_ms": status["maximum_latency"], "throughput_mbps": status["minimum_throughput"], "jitter_ms": status["maximum_jitter"]}
    status["forecast_horizon"] = f"{horizon} hours"
    status["metrics"] = {metric: {"predicted": summary_values[metric], "threshold": threshold,
                                  "status": "ANOMALY" if metric in violated else "NORMAL"}
                         for metric, threshold in {"latency_ms": 50, "throughput_mbps": 250, "jitter_ms": 10}.items()}
    agent_text = explain(status, {"forecast_horizon": f"{horizon} hours", **status}) if use_ollama else ""
    rca = RootCauseAgent().analyze(status) if status["status"] == "ANOMALY" else None
    return {"predictions": predictions, "status": status, "evaluation": evaluation, "horizon": horizon,
            "agent_text": agent_text, "root_cause_analysis": rca}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("normal", "anomaly"), default="normal")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(args.scenario, args.generate)
    print(f"STATUS: {result['status']['status']}")
    print(f"PREDICTION HORIZON: {result['horizon']} HOURS")
    print("MODEL EVALUATION:", result["evaluation"])
    print(result["status"])
    print(result["agent_text"])

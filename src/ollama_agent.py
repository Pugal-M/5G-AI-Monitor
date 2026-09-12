"""Ollama reasoning layer; deterministic Python supplies all numerical facts."""
import json
import requests


def explain(status: dict, context: dict, model: str = "qwen2.5:1.5b-instruct") -> str:
    payload = {"status": status, "context": context}
    prompt = ("You are a 5G operations assistant. Use only this JSON. Explain the supplied "
              "deterministic status, user impact, and recommend actions. Do not calculate thresholds "
              "or invent values.\n" + json.dumps(payload))
    try:
        response = requests.post("http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}, timeout=30)
        response.raise_for_status()
        return response.json()["response"].strip()
    except (requests.RequestException, KeyError):
        if status["status"] == "NORMAL":
            return "Network is healthy. Continue routine monitoring of latency, throughput, and jitter."
        return ("Predicted degradation requires operator review. Check cell congestion, radio resource "
                "utilization, retransmissions, and transport/backhaul performance.")

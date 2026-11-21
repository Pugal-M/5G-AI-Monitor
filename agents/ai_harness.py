"""AI Harness: orchestrates the 5G AIOps agents and records each execution step.

The harness is deliberately framework-light: existing ML, RAG and Ollama
components remain independent and can be tested or replaced individually.
"""
from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from agents.anomaly_agent import AnomalyAgent
from agents.root_cause_agent import RootCauseAgent
from src.ollama_agent import explain


class AIHarness:
    """Coordinate anomaly detection, root-cause analysis and explanation."""

    def __init__(self, anomaly_agent=None, root_cause_agent=None, explainer=None):
        self.anomaly_agent = anomaly_agent or AnomalyAgent()
        self.root_cause_agent = root_cause_agent or RootCauseAgent()
        self.explainer = explainer or explain

    @staticmethod
    def _stage(trace, name, operation):
        started = perf_counter()
        try:
            result = operation()
            trace.append({
                "stage": name,
                "status": "COMPLETED",
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            })
            return result
        except Exception as error:
            trace.append({
                "stage": name,
                "status": "FAILED",
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "error": f"{type(error).__name__}: {error}",
            })
            raise

    def run(self, predictions, forecast_horizon: str, use_ollama: bool = True) -> dict:
        """Run the monitoring workflow and return results plus an audit trace."""
        run_id = str(uuid4())
        trace = []

        status = self._stage(
            trace,
            "anomaly_detection",
            lambda: self.anomaly_agent.analyze(predictions),
        )

        # Give RCA the KPI snapshot as well as individual threshold violations.
        status["network_metrics"] = {
            "latency_ms": status.get("maximum_latency"),
            "throughput_mbps": status.get("minimum_throughput"),
            "jitter_ms": status.get("maximum_jitter"),
        }
        status["forecast_horizon"] = forecast_horizon

        if status.get("status") == "ANOMALY":
            try:
                rca = self._stage(
                    trace,
                    "root_cause_analysis",
                    lambda: self.root_cause_agent.analyze(status),
                )
            except Exception as error:
                # An RCA failure must not erase the anomaly detected by the
                # deterministic threshold agent.
                rca = {
                    "status": "INSUFFICIENT EVIDENCE",
                    "reason": f"Root Cause Agent failed: {type(error).__name__}: {error}",
                    "root_causes": [],
                    "recommended_actions": [],
                    "rag": None,
                }
            agent_text = rca.get("qwen_reasoning", "") or (
                "Anomaly detected. Root-cause evidence is unavailable; "
                "review the KPI values and diagnostic checks."
            )
        else:
            rca = None
            agent_text = ""
            trace.append({
                "stage": "root_cause_analysis",
                "status": "SKIPPED",
                "reason": "No anomaly was detected.",
                "duration_ms": 0.0,
            })

        if use_ollama and rca is None:
            agent_text = self._stage(
                trace,
                "llm_explanation",
                lambda: self.explainer(
                    status,
                    {"forecast_horizon": forecast_horizon, **status},
                ),
            )
        elif rca is not None:
            trace.append({
                "stage": "llm_explanation",
                "status": "INCLUDED_IN_ROOT_CAUSE_ANALYSIS",
                "duration_ms": 0.0,
            })
        else:
            trace.append({
                "stage": "llm_explanation",
                "status": "SKIPPED",
                "reason": "Ollama explanation was disabled.",
                "duration_ms": 0.0,
            })

        return {
            "run_id": run_id,
            "status": status,
            "agent_text": agent_text,
            "root_cause_analysis": rca,
            "harness_trace": trace,
        }

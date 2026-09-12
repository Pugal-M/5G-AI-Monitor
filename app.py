"""Streamlit demonstration dashboard."""
import sys
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from main import run_pipeline
from anomaly_agent import THRESHOLDS

st.set_page_config(page_title="5G AI Monitor", layout="wide")
st.title("5G AI Network Operations Center")
scenario = st.selectbox("Scenario", ["normal", "anomaly"], format_func=lambda v: "Normal Traffic" if v == "normal" else "Anomalous Traffic")
if st.button("Generate forecast", type="primary"):
    with st.spinner("Generating data, training Random Forest, and forecasting..."):
        st.session_state.result = run_pipeline(scenario)
if "result" in st.session_state:
    result, forecast, status = st.session_state.result, st.session_state.result["predictions"], st.session_state.result["status"]
    st.header("🟢 NORMAL" if status["status"] == "NORMAL" else "🔴 ANOMALY DETECTED")
    cols = st.columns(4)
    cols[0].metric("Severity", status["severity"])
    cols[1].metric("Anomalous predictions", status["anomalous_predictions"])
    cols[2].metric("Max latency", f"{status['maximum_latency']} ms")
    cols[3].metric("Min throughput", f"{status['minimum_throughput']} Mbps")
    st.subheader("📈 Random Forest Prediction Output")
    st.caption(f"Forecast horizon: {result['horizon']} hours • {len(forecast)} fifteen-minute predictions")
    prediction_cols = st.columns(3)
    prediction_cols[0].metric("Maximum predicted latency", f"{status['maximum_latency']} ms")
    prediction_cols[1].metric("Minimum predicted throughput", f"{status['minimum_throughput']} Mbps")
    prediction_cols[2].metric("Maximum predicted jitter", f"{status['maximum_jitter']} ms")
    st.dataframe(
        forecast.assign(timestamp=forecast["timestamp"].astype(str)).round(2),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Model evaluation (chronological holdout)"):
        st.json(result["evaluation"])
    chart_columns = st.columns(2)
    for index, (metric, title) in enumerate([("latency_ms", "Latency"), ("throughput_mbps", "Throughput"), ("jitter_ms", "Jitter")]):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(forecast["timestamp"], forecast[metric], label="Forecast")
        ax.axhline(THRESHOLDS[metric], color="red", linestyle="--", label="Threshold")
        ax.set_title(title); ax.legend(); ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        with chart_columns[index % 2]:
            st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.subheader("🤖 AI Network Operations Agent Output")
    st.write(result["agent_text"])
    st.caption("Deterministic threshold-agent result sent to Qwen/Ollama:")
    st.json(status)
    rca = result.get("root_cause_analysis")
    if rca:
        st.header("🔎 Root Cause Analysis")
        if rca["status"] == "INSUFFICIENT EVIDENCE":
            st.warning("INSUFFICIENT EVIDENCE: " + rca.get("reason", "No relevant knowledge-base evidence."))
        else:
            for number, cause in enumerate(rca["root_causes"], 1):
                st.subheader(f"Possible Root Cause #{number}: {cause['cause']}")
                st.metric("Confidence", f"{cause['confidence']:.0%}")
                st.write(cause["reason"])
                st.write("**Evidence:** " + ", ".join(cause["evidence"]))
                st.write("**Diagnostic checks:** " + "; ".join(cause["diagnostic_checks"]))
            st.subheader("Recommended Actions")
            st.write(rca["recommended_actions"])
            st.subheader("RAG Retrieval")
            st.write("**Query:** " + rca["rag"]["query"])
            for item in rca["rag"]["results"]:
                with st.expander(f"{item['document']} — similarity {item['score']:.3f}"):
                    st.write(item["content"])
            st.subheader("Qwen RCA reasoning")
            st.write(rca["qwen_reasoning"])

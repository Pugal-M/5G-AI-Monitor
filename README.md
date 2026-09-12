# 5G AI Network Monitoring & Prediction

Local demo: synthetic telemetry → Random Forest forecast → deterministic threshold agent → optional Ollama/Qwen explanation → Streamlit dashboard.

## Setup and run

    cd 5g-ai-monitor
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python src/main.py --scenario normal --generate
    python src/main.py --scenario anomaly
    streamlit run app.py

Ollama is optional; the ML and threshold checks work without it.

    ollama pull qwen2.5:1.5b-instruct
    ollama serve

## Architecture

Every 15 minutes the generator emits correlated latency, throughput, and jitter. Lag values (1, 2, 3, 4, and 8 intervals), hour, and day-of-week feed a multi-output Random Forest regressor. Forecasting is recursive: a predicted interval becomes lag input to the next interval.

The threshold agent applies latency > 50 ms, throughput < 250 Mbps, and jitter > 10 ms. Zero, one, two, or three violated metric types map to LOW, MEDIUM, HIGH, or CRITICAL severity. It returns structured JSON; Qwen only explains those supplied facts and recommends action.

## Root Cause Analysis and local RAG

When the threshold agent returns `ANOMALY`, the separate Root Cause Agent receives its JSON contract, creates a search query, retrieves evidence from `knowledge_base/`, then ranks evidence-supported causes and diagnostic actions. The retrieval stack is local BGE-M3 embeddings plus a FAISS cosine-similarity index; Qwen receives the anomaly JSON and retrieved chunks for its explanation.

Build the index once after installing dependencies or changing the knowledge base:

    python src/rag_ingestion.py

This writes `rag/faiss.index` and `rag/metadata.pkl`. If the index or packages are unavailable, the RCA explicitly returns `INSUFFICIENT EVIDENCE` rather than inventing a cause.

## Outputs

Generated telemetry is in `data/normal_5g.csv` and `data/anomaly_5g.csv`; forecasts are in `data/predictions_normal.csv` and `data/predictions_anomaly.csv`. The trained model is `models/random_forest.pkl`.

Normal mode creates a 24-hour / 96-interval forecast. Anomaly mode forecasts two hours / eight intervals using the active incident's observed lag values.
# 5G-AI-Monitor

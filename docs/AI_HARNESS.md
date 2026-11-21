# AI Harness integration

## What changed

The new `agents/ai_harness.py` module coordinates the existing 5G monitoring components. It does not replace the Random Forest model, BGE-M3, FAISS, or Qwen.

### Runtime flow

```text
Telemetry
   |
Random Forest forecast
   |
AIHarness
   |
AnomalyAgent
   |-- NORMAL --> optional Qwen summary --> dashboard
   |
   '-- ANOMALY --> RootCauseAgent
                       |
                    BGE-M3
                       |
                     FAISS
                       |
                 Retrieved evidence
                       |
                    Qwen
                       |
                RCA report
                       |
                    dashboard
```

### Traceability

Each run returns:
- `run_id`: unique identifier for the pipeline execution.
- `harness_trace`: ordered stages with status and elapsed milliseconds.
- `status`: deterministic threshold result.
- `root_cause_analysis`: evidence-grounded RCA result, or an explicit insufficient-evidence result.

### Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/rag_ingestion.py
python src/main.py --scenario anomaly --generate
streamlit run app.py
```

Omit the RAG ingestion step only if a valid index already exists in `rag/`. The first BGE-M3 model load may download model weights. Ollama must be running locally with `qwen2.5:1.5b-instruct` pulled for LLM explanations.

### Important limitation

The project uses synthetic telemetry. A cause is a hypothesis supported by available metrics and retrieved documents, not proof of a real network fault. Real deployment needs additional live KPIs (such as PRB utilization, SINR, CQI, retransmission rate, and backhaul utilization) and operator validation.

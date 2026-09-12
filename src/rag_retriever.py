"""BGE-M3 embedding and FAISS similarity search."""
from __future__ import annotations
import pickle
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT, MODEL_NAME = Path(__file__).resolve().parents[1], "BAAI/bge-m3"
_model = None


def search(query: str, top_k: int = 5) -> dict:
    global _model
    rag = ROOT / "rag"
    if not (rag / "faiss.index").exists():
        raise FileNotFoundError("RAG index missing. Run: python src/rag_ingestion.py")
    _model = _model or SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(rag / "faiss.index"))
    with (rag / "metadata.pkl").open("rb") as source:
        metadata = pickle.load(source)
    vector = np.asarray(_model.encode([query], normalize_embeddings=True), dtype="float32")
    scores, positions = index.search(vector, min(top_k, len(metadata)))
    results = [{**metadata[position], "score": round(float(score), 3)} for score, position in zip(scores[0], positions[0]) if position >= 0]
    return {"query": query, "results": results}

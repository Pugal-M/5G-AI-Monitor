"""Build a local BGE-M3/FAISS index from the 5G knowledge base."""
from __future__ import annotations
import pickle
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE, RAG_DIR = ROOT / "knowledge_base", ROOT / "rag"
MODEL_NAME = "BAAI/bge-m3"


def chunk_text(text: str, size: int = 700, overlap: int = 100) -> list[str]:
    return [text[start:start + size] for start in range(0, len(text), size - overlap) if text[start:start + size].strip()]


def ingest() -> int:
    documents = []
    for path in sorted(KNOWLEDGE_BASE.glob("*.txt")):
        for number, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            documents.append({"document": path.name, "chunk": number, "content": chunk})
    if not documents:
        raise RuntimeError("No .txt documents found in knowledge_base")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode([item["content"] for item in documents], normalize_embeddings=True, show_progress_bar=True)
    vectors = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    RAG_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(RAG_DIR / "faiss.index"))
    with (RAG_DIR / "metadata.pkl").open("wb") as output:
        pickle.dump(documents, output)
    return len(documents)


if __name__ == "__main__":
    print(f"Indexed {ingest()} knowledge chunks.")

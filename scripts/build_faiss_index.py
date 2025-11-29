# scripts/build_faiss_index.py
import json
from pathlib import Path
import numpy as np
try:
    import faiss
except Exception:
    raise SystemExit("Please install faiss (cpu): pip install faiss-cpu")

# Change EMBEDDING_DIM to match your retriever_api EMBEDDING_DIM
EMBEDDING_DIM = 768
DATA_DIR = Path("storage")
DOCS_PATH = DATA_DIR / "documents.json"
INDEX_PATH = DATA_DIR / "faiss.index"
ID_MAP_PATH = DATA_DIR / "id_map.json"

# Simple DummyEmbeddingModel (same logic used by retriever_api)
def simple_embed(texts, dim=EMBEDDING_DIM):
    arr = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t))
        for j in range(dim):
            arr[i, j] = ((h >> (j % 32)) & 0xFF) / 255.0
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms

def main():
    if not DOCS_PATH.exists():
        raise SystemExit(f"{DOCS_PATH} not found. Create storage/documents.json first.")
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    ids = [d["id"] for d in docs]
    texts = [d.get("text","") for d in docs]
    vecs = simple_embed(texts)
    # Build FAISS index (IndexFlatIP for cosine/dot product with normalized vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vecs.astype("float32"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
    print("Wrote", INDEX_PATH, "and", ID_MAP_PATH)

if __name__ == "__main__":
    main()

# services/storage/reindex.py
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DOCS_PATH = os.path.join(os.getcwd(), "documents.json")   # matches DocumentStore default
FAISS_PATH = os.path.join(os.getcwd(), "faiss.index")
IDMAP_PATH = os.path.join(os.getcwd(), "id_map.json")

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DIM = 768

def reindex_from_documents():
    # Load documents
    if not os.path.exists(DOCS_PATH):
        print("No documents.json found at", DOCS_PATH)
        return

    with open(DOCS_PATH, "r") as f:
        docs = json.load(f)  # dict: doc_id -> {id, filename, text}

    doc_items = list(docs.items())
    if len(doc_items) == 0:
        print("No documents to index.")
        return

    print(f"Rebuilding FAISS index from {len(doc_items)} documents...")

    # Create fresh index
    index = faiss.IndexFlatL2(DIM)
    model = SentenceTransformer(MODEL_NAME)

    id_map = []
    vectors = []

    for doc_id, doc in doc_items:
        text = doc.get("text", "")
        if not text:
            continue
        emb = model.encode([text])[0].astype("float32")
        vectors.append(emb)
        id_map.append(doc_id)

    if len(vectors) == 0:
        print("No vectors generated (empty documents).")
        return

    vectors_np = np.vstack(vectors)
    index.add(vectors_np)

    # Persist index and id_map
    faiss.write_index(index, FAISS_PATH)
    with open(IDMAP_PATH, "w") as f:
        json.dump(id_map, f)

    print("Reindex complete.")
    print("faiss.ntotal =", index.ntotal)
    print("id_map length =", len(id_map))

if __name__ == "__main__":
    reindex_from_documents()

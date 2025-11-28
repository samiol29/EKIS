# services/storage/reindex_embeddings_batch.py
import numpy as np
from .document_store import DocumentStore
from .faiss_index import FAISSIndex
from time import time

#batch size can be increased or decreased depending on the system.
def rebuild_all(batch_size: int = 64, limit: int | None = None, show_progress: bool = True):
    """
    Rebuild FAISS index by embedding documents in batches.

    Args:
        batch_size: number of documents to encode per model.encode() call
        limit: optional integer to only reindex the first N documents (for testing)
        show_progress: whether to print progress info
    """
    doc_store = DocumentStore()
    docs_dict = doc_store.documents  # dict: doc_id -> {id, filename, text}

    # Convert to ordered lists
    items = list(docs_dict.items())
    if limit is not None:
        items = items[:limit]

    doc_ids = [doc_id for doc_id, meta in items]
    texts  = [meta.get("text", "") for doc_id, meta in items]

    if show_progress:
        print(f"Embedding {len(texts)} documents in batches of {batch_size}...")

    fa = FAISSIndex()  # will load model and dim

    model = fa.model
    dim = fa.dim

    all_embs = []
    t0 = time()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if show_progress:
            print(f"  batch {i//batch_size + 1} / {((len(texts)-1)//batch_size)+1} - encoding {len(batch_texts)} items...", end="", flush=True)

        # Use model.encode on the batch (fast)
        batch_embs = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
        # Ensure numpy 2D array
        batch_embs = np.asarray(batch_embs, dtype=np.float32)

        # Normalize each row to unit length
        norms = np.linalg.norm(batch_embs, axis=1, keepdims=True) + 1e-12
        batch_embs = batch_embs / norms

        all_embs.append(batch_embs)

        if show_progress:
            print(" done")

    # Concatenate all embeddings
    if len(all_embs) == 0:
        embeddings = np.zeros((0, dim), dtype=np.float32)
    else:
        embeddings = np.vstack(all_embs).astype(np.float32)

    # Assign into FAISSIndex and rebuild index
    fa.id_map = doc_ids
    fa.embeddings = embeddings
    fa.index = faiss_index = fa.index = __import__("faiss").read_index if False else None  # placeholder to avoid lint; we'll rebuild below

    # Rebuild FAISS IndexFlatIP
    import faiss
    fa.index = faiss.IndexFlatIP(dim)
    if embeddings.shape[0] > 0:
        fa.index.add(embeddings)

    # Persist everything
    fa.save_index()

    t1 = time()
    print(f"Rebuild complete. ntotal={fa.index.ntotal}. Time elapsed: {t1 - t0:.2f}s")
    return fa

if __name__ == "__main__":
    # Example usage: adjust batch_size or limit if you want a quick test
    # from project root:
    # python -m services.storage.reindex_embeddings_batch
    rebuild_all(batch_size=32, limit=None, show_progress=True)

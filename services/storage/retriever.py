# services/storage/retriever.py
import numpy as np
from .faiss_index import FAISSIndex
from .document_store import DocumentStore

# Initialize FAISS + DocumentStore
faiss_manager = FAISSIndex()
doc_store = DocumentStore()


# -------------------------
#  Utility Functions
# -------------------------

def format_api_response(query, results):
    """
    Wrap retriever output into clean JSON structure.
    """
    return {
        "query": query,
        "count": len(results),
        "results": results
    }


def normalize_soft(sim, min_sim=-1.0, max_sim=1.0):
    """
    Convert cosine sim (-1..1) to 0..1 for easier UI representation.
    """
    s = (sim - min_sim) / (max_sim - min_sim)
    return round(max(0.0, min(1.0, s)), 4)


# -------------------------
#  FAISS + Rerank Search
# -------------------------

def search_query(query, k=5):
    """
    Low-level FAISS search.
    Uses FAISSIndex.search which returns cosine similarity scores.
    """

    raw_results = faiss_manager.search(query, k)

    results = []
    for rank, r in enumerate(raw_results):
        doc_id = r["document_id"]
        doc_entry = doc_store.get_document(doc_id)

        if not doc_entry:
            continue

        results.append({
            "rank": rank + 1,
            "raw_score": r["score"],                   # FAISS inner-product (cosine)
            "score": normalize_soft(r["score"]),      # optional normalized score
            "doc_id": doc_id,
            "filename": doc_entry.get("filename"),
            "text": doc_entry.get("text")
        })

    return results


def rerank_results(query, results):
    """
    Semantic reranking using *stored embeddings* from FAISSIndex.
    Fast & consistent: no re-embedding of full text.
    """

    if not results:
        return results

    # normalized query embedding
    q_emb = faiss_manager.embed(query)
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    q_emb = q_emb.astype(np.float32)

    reranked = []
    for r in results:
        doc_id = r["doc_id"]

        # get precomputed unit embedding
        doc_emb = faiss_manager.get_embedding_by_docid(doc_id)

        # fallback: embed snippet if missing
        if doc_emb is None:
            snippet = r["text"][:1200]
            emb = faiss_manager.embed(snippet)
            doc_emb = emb / (np.linalg.norm(emb) + 1e-12)
            doc_emb = doc_emb.astype(np.float32)

        cos_sim = float(np.dot(q_emb, doc_emb))

        item = r.copy()
        item["rerank_score"] = round(cos_sim, 6)
        item["rerank_norm"] = normalize_soft(cos_sim)

        reranked.append(item)

    # Sort by semantic similarity descending
    reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
    return reranked


# -------------------------
#  Filtering
# -------------------------

def apply_filters(results, filename=None, filetype=None, min_score=None, max_results=None):
    """
    Filter results by metadata or score.
    Uses rerank_norm (0..1) for min_score so thresholds are intuitive.
    """
    filtered = results

    if filename:
        filtered = [r for r in filtered if r.get("filename") == filename]

    if filetype:
        filtered = [r for r in filtered if r.get("filename", "").lower().endswith(filetype.lower())]

    # Use rerank_norm (0..1) for min_score filtering (more intuitive)
    if min_score is not None:
        filtered = [r for r in filtered if r.get("rerank_norm", 0) >= float(min_score)]

    if max_results is not None:
        filtered = filtered[:max_results]

    return filtered



# -------------------------
#  High-level Retriever
# -------------------------

def retrieve(
    query,
    k=5,
    filename=None,
    filetype=None,
    min_score=None,
    max_results=None
):
    """
    Full high-level retriever:
    - FAISS vector search
    - Semantic re-ranking
    - Metadata filtering
    - API-style structured response
    """

    if not query or not query.strip():
        return format_api_response(query, [])

    try:
        results = search_query(query, k)
        results = rerank_results(query, results)

        results = apply_filters(
            results,
            filename=filename,
            filetype=filetype,
            min_score=min_score,
            max_results=max_results
        )

        final = [r for r in results if r["text"]]

        return format_api_response(query, final)

    except Exception as e:
        print(f"[Retriever Error] {e}")
        return format_api_response(query, [])

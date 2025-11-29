# services/root_agent/subagents/search_agent.py
import os
from typing import List, Dict, Any

# Local FAISSIndex & DocumentStore (existing code)
from services.storage.faiss_index import FAISSIndex
from services.storage.document_store import DocumentStore

# Optional http client for retriever API
try:
    import httpx
    HTTPX_AVAILABLE = True
except Exception:
    HTTPX_AVAILABLE = False

class SearchAgent:
    """
    Semantic search agent.
    Preference order:
      1) Call local retriever HTTP API (if USE_RETRIEVER_API=true and httpx present)
      2) Fallback: use FAISSIndex directly
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8000/v1/search"):
        self.api_url = os.getenv("RETRIEVER_API_URL", api_url)
        self.use_api = os.getenv("USE_RETRIEVER_API", "true").lower() in ("1", "true", "yes")
        self.index = FAISSIndex()
        self.store = DocumentStore()

    def _enrich_results_from_index_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched = []
        for r in hits:
            # support both formats: faiss-style dicts or API-style results
            doc_id = r.get("document_id") or r.get("id") or r.get("doc_id")
            score = r.get("distance") or r.get("score") or r.get("score", None)
            doc = self.store.get_document(doc_id) if doc_id else None
            excerpt = None
            filename = None
            if doc:
                excerpt = doc.get("text", "")[:500]
                filename = doc.get("filename") or doc.get("id") or None
            enriched.append({
                "document_id": doc_id,
                "filename": filename,
                "score": score,
                "excerpt": excerpt
            })
        return enriched

    def answer_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Synchronous method used by router. Returns dict:
        { "query": ..., "results": [ {document_id, filename, score, excerpt}, ... ] }
        """
        # 1) Try HTTP API if enabled & available
        if self.use_api and HTTPX_AVAILABLE:
            try:
                with httpx.Client(timeout=15.0) as client:
                    body = {"query": query, "top_k": k, "use_rerank": False}
                    resp = client.post(self.api_url, json=body)
                    resp.raise_for_status()
                    j = resp.json()
                    results = j.get("results", [])
                    enriched = self._enrich_results_from_index_hits(results)
                    return {"query": query, "results": enriched}
            except Exception as e:
                # fallback to local index (log but don't crash)
                print(f"[SearchAgent] retriever API failed ({e}), falling back to local FAISSIndex.")

        # 2) Fallback to FAISSIndex search
        try:
            hits = self.index.search(query, k=k)  # assumes FAISSIndex.search returns list of dicts with 'document_id' and 'distance'
            enriched = self._enrich_results_from_index_hits(hits)
            return {"query": query, "results": enriched}
        except Exception as e:
            print(f"[SearchAgent] local index search failed: {e}")
            return {"query": query, "results": []}

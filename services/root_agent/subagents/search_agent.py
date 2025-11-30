# services/root_agent/subagents/search_agent.py
import os
from typing import List, Dict, Any, Optional
import logging

# Local FAISSIndex & DocumentStore (existing code)
from services.storage.faiss_index import FAISSIndex
from services.storage.document_store import DocumentStore

# HTTP client: prefer requests (stable). If not present, try httpx.
try:
    import requests
    HTTP_CLIENT = "requests"
except Exception:
    try:
        import httpx
        HTTP_CLIENT = "httpx"
    except Exception:
        HTTP_CLIENT = None

logger = logging.getLogger("root_agent.search_agent")

class SearchAgent:
    """
    Semantic search agent.
    Priority:
      1) Call local retriever HTTP API (if enabled)
      2) Fallback: use FAISSIndex directly
    """
    def __init__(self, api_url: str = "http://127.0.0.1:8000/v1/search"):
        self.api_url = os.getenv("RETRIEVER_API_URL", api_url)
        self.use_api = os.getenv("USE_RETRIEVER_API", "true").lower() in ("1", "true", "yes")
        self.index = FAISSIndex()
        self.store = DocumentStore()

    def _map_api_result(self, r: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map retriever API result -> unified result shape:
        { document_id, filename, score, excerpt }
        """
        doc_id = r.get("id") or r.get("document_id") or r.get("doc_id")
        score = r.get("score")
        excerpt = r.get("excerpt") or ""
        metadata = r.get("metadata") or {}
        filename = metadata.get("filename") or metadata.get("source") or metadata.get("title") or None

        # If excerpt missing, try to fetch from local DocumentStore as a fallback
        if not excerpt and doc_id:
            doc = self.store.get_document(doc_id)
            if doc:
                excerpt = (doc.get("text") or "")[:500]
            else:
                excerpt = ""

        return {
            "document_id": doc_id,
            "filename": filename,
            "score": score,
            "excerpt": excerpt
        }

    def _enrich_results_from_index_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Legacy function: keep compatibility for FAISSIndex output (local).
        Converts local hits (document_id, distance) into the unified shape.
        """
        enriched = []
        for r in hits:
            # support both formats: faiss-style dicts or API-style results
            doc_id = r.get("document_id") or r.get("id") or r.get("doc_id")
            # score could be 'distance' for some local APIs (convert if present)
            score = r.get("score") or r.get("distance") or None
            doc = self.store.get_document(doc_id) if doc_id else None
            excerpt = None
            filename = None
            if doc:
                excerpt = (doc.get("text") or "")[:500]
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
        This version ensures excerpts are populated by fetching /v1/doc/{doc_id}
        from the retriever API when missing.
        """
        # 1) Try HTTP API if enabled & HTTP client available
        enriched = []
        if self.use_api and HTTP_CLIENT is not None:
            try:
                payload = {"query": query, "top_k": k, "use_rerank": True}
                if HTTP_CLIENT == "requests":
                    resp = requests.post(self.api_url, json=payload, timeout=15)
                    resp.raise_for_status()
                    j = resp.json()
                else:
                    with httpx.Client(timeout=15.0) as client:
                        r = client.post(self.api_url, json=payload)
                        r.raise_for_status()
                        j = r.json()

                results = j.get("results", [])
                # Map results straight from API
                enriched = [self._map_api_result(r) for r in results]

                # If any enriched result lacks an excerpt, try to fetch doc via GET /v1/doc/{id}
                base_url = self.api_url.rsplit("/", 1)[0]  # e.g. http://127.0.0.1:8000
                for r in enriched:
                    if (not r.get("excerpt")) and r.get("document_id"):
                        doc_id = r["document_id"]
                        try:
                            doc_resp = None
                            if HTTP_CLIENT == "requests":
                                doc_resp = requests.get(f"{base_url}/v1/doc/{doc_id}", timeout=8)
                                if doc_resp.status_code == 200:
                                    doc_json = doc_resp.json()
                                else:
                                    doc_json = None
                            else:
                                with httpx.Client(timeout=8.0) as client:
                                    dr = client.get(f"{base_url}/v1/doc/{doc_id}")
                                    if dr.status_code == 200:
                                        doc_json = dr.json()
                                    else:
                                        doc_json = None

                            if doc_json:
                                txt = doc_json.get("text") or ""
                                meta = doc_json.get("metadata") or {}
                                r["excerpt"] = txt[:500] if txt else r.get("excerpt") or ""
                                r["filename"] = meta.get("filename") or meta.get("title") or r.get("filename")
                        except Exception:
                            # ignore doc fetch errors — fallback to local store below
                            pass

                # Fallback to local DocumentStore for any still-missing excerpts
                for r in enriched:
                    if (not r.get("excerpt")) and r.get("document_id"):
                        doc = self.store.get_document(r["document_id"])
                        if doc:
                            r["excerpt"] = (doc.get("text") or "")[:500]
                            r["filename"] = r.get("filename") or doc.get("filename") or doc.get("id")

                return {"query": query, "results": enriched}

            except Exception as e:
                logger.exception("retriever API failed (%s). Falling back to local FAISSIndex.", e)

        # 2) Fallback: local FAISSIndex search
        try:
            hits = self.index.search(query, k=k)  # expects list of dicts e.g. {'document_id':..., 'distance':...}
            enriched = self._enrich_results_from_index_hits(hits)
            # ensure fallback: if excerpts missing, try DocumentStore (already used in _enrich function)
            for r in enriched:
                if (not r.get("excerpt")) and r.get("document_id"):
                    doc = self.store.get_document(r["document_id"])
                    if doc:
                        r["excerpt"] = (doc.get("text") or "")[:500]
                        r["filename"] = r.get("filename") or doc.get("filename") or doc.get("id")
            return {"query": query, "results": enriched}
        except Exception as e:
            logger.exception("Local FAISSIndex search failed: %s", e)
            return {"query": query, "results": []}


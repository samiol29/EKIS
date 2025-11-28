from services.storage.faiss_index import FAISSIndex
from services.storage.document_store import DocumentStore

class SearchAgent:
    """
    Semantic search agent (FAISS + sentence-transformers).
    """

    def __init__(self):
        self.index = FAISSIndex()
        self.store = DocumentStore()

    def answer_query(self, query: str):
        results = self.index.search(query, k=5)

        enriched = []
        for r in results:
            doc = self.store.get_document(r["document_id"])
            if doc:
                enriched.append({
                    "document_id": r["document_id"],
                    "filename": doc["filename"],
                    "distance": r["distance"],
                    "excerpt": doc["text"][:300]  # first 300 chars
                })

        return {
            "query": query,
            "results": enriched
        }

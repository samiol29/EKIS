class DocumentAgent:
    """
    Placeholder document ingestion agent.
    Will be replaced by real PDF/DOCX ingestion + embeddings in Phase 2.
    """

    def __init__(self):
        self.docs = []

    def ingest_document(self, entities: dict):
        doc = {
            "id": len(self.docs) + 1,
            "name": entities.get("filename", "untitled")
        }
        self.docs.append(doc)
        return {"status": "ok", "doc_id": doc["id"]}

    def list_documents(self):
        return {"documents": self.docs}

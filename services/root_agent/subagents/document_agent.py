import pdfplumber
import docx
import os

from services.storage.document_store import DocumentStore
from services.storage.faiss_index import FAISSIndex

class DocumentAgent:
    """
    Handles:
    - File ingestion (PDF / TXT / DOCX)
    - Text extraction
    - Embedding generation (FAISS)
    - Document storage
    """

    def __init__(self):
        self.store = DocumentStore()
        self.index = FAISSIndex()

    def extract_text(self, filepath: str):
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)

        if ext == ".txt":
            return open(filepath, "r", encoding="utf-8").read()

        if ext == ".docx":
            doc = docx.Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)

        return ""  # Unsupported file

    def ingest_document(self, entities: dict):
        filepath = entities.get("filepath")
        filename = entities.get("filename", "unnamed")

        if not filepath or not os.path.exists(filepath):
            return {"error": "Invalid or missing filepath"}

        # extract text
        text = self.extract_text(filepath)

        # store
        doc_id = self.store.add_document(text, filename)

        # add to FAISS index
        self.index.add_document(doc_id, text)

        return {
            "status": "ok",
            "doc_id": doc_id,
            "filename": filename
        }

    def list_documents(self):
        return {"documents": self.store.all_documents()}

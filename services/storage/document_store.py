# services/storage/document_store.py
import uuid
import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DOCS_PATH = Path(os.getenv("DOCUMENTS_JSON", Path("storage") / "documents.json"))

class DocumentStore:
    """
    Simple document store that persists to a JSON file.

    The on-disk format may be either:
      - a mapping of {doc_id: {id, filename, text, metadata...}, ...}
      - a list of documents [{id, text, metadata...}, ...]

    This class normalizes access to always present a dict internally.
    """
    def __init__(self, store_path: Optional[str] = None):
        self.store_path = Path(store_path) if store_path else DEFAULT_DOCS_PATH
        # Ensure parent dir exists
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or create doc store
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Normalize: allow both list and dict on disk
                if isinstance(raw, list):
                    # convert list to dict keyed by id
                    self.documents = {str(d.get("id") or d.get("doc_id") or str(uuid.uuid4())): d for d in raw}
                elif isinstance(raw, dict):
                    # assume mapping doc_id -> doc dict
                    self.documents = raw
                else:
                    logger.warning("Unknown documents.json format; starting with empty store.")
                    self.documents = {}
            except Exception as e:
                logger.exception("Failed to read document store %s: %s", self.store_path, e)
                self.documents = {}
        else:
            self.documents = {}

    def save(self) -> None:
        """
        Persist documents to disk as a list (for compatibility).
        """
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        docs_list = list(self.documents.values())
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(docs_list, f, ensure_ascii=False, indent=2)

    def add_document(self, text: str, filename: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        doc_id = str(uuid.uuid4())
        self.documents[doc_id] = {
            "id": doc_id,
            "filename": filename or "",
            "text": text,
            "metadata": metadata or {}
        }
        self.save()
        return doc_id

def get_document(self, doc_id: str) -> dict:
    # existing lookup (adjust to your implementation)
    doc = self._docs_by_id.get(doc_id)  # or however you store docs
    if not doc:
        return {}
    # ensure text key exists
    text = doc.get("text") or doc.get("content") or ""
    filename = doc.get("filename") or doc.get("id") or None
    metadata = doc.get("metadata") or {}
    return {"id": doc_id, "text": text, "filename": filename, "metadata": metadata}

    def all_documents(self) -> List[Dict[str, Any]]:
        """
        Return a list of document dicts.
        """
        return list(self.documents.values())

# -------------------------
# Module-level helpers used by the API
# -------------------------
def _normalize_doc(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a document dict has at least 'id' and 'text' keys and a metadata dict.
    """
    doc_id = str(d.get("id") or d.get("doc_id") or d.get("filename") or str(uuid.uuid4()))
    return {
        "id": doc_id,
        "text": d.get("text", "") or d.get("content", "") or "",
        "metadata": d.get("metadata", {}) or {}
    }

def load_documents(store_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load documents and return a list of dicts with keys: id, text, metadata.

    Tries in this order:
      1. If a DocumentStore file exists at provided store_path (or DEFAULT_DOCS_PATH), use it.
      2. If not, and the global DEFAULT_DOCS_PATH exists, read that.
      3. Return empty list if nothing found.
    """
    try:
        ds = DocumentStore(store_path) if store_path else DocumentStore()
        docs = ds.all_documents()
        # Normalize each document to minimal shape
        out = []
        for d in docs:
            out.append(_normalize_doc(d))
        return out
    except Exception as e:
        logger.exception("load_documents failed: %s", e)
        return []

# Aliases for backward compatibility
def read_documents(store_path: Optional[str] = None) -> List[Dict[str, Any]]:
    return load_documents(store_path)

def get_documents(store_path: Optional[str] = None) -> List[Dict[str, Any]]:
    return load_documents(store_path)

# If you want a quick programmatic store instance via import:
_default_store = DocumentStore()
def get_store() -> DocumentStore:
    return _default_store

# services/root_agent/nlu.py
from pydantic import BaseModel
from typing import Dict, Any

class IntentResult(BaseModel):
    intent: str
    entities: Dict[str, Any] = {}

def detect_intent(text: str) -> IntentResult:
    t = (text or "").strip()
    lower = t.lower()

    # --- Basic rule-based detection ---
    if "upload" in lower or "add document" in lower:
        return IntentResult(intent="upload_document", entities={"query": t})

    if "search" in lower or "find" in lower or "question" in lower:
        return IntentResult(intent="semantic_query", entities={"query": t})

    if "list documents" in lower or "show docs" in lower:
        return IntentResult(intent="list_documents", entities={"query": t})

    if "products" in lower or "catalog" in lower:
        return IntentResult(intent="list_products", entities={"query": t})

    # fallback to semantic search with original text as query
    return IntentResult(intent="semantic_query", entities={"query": t})

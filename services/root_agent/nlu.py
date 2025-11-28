from pydantic import BaseModel

class IntentResult(BaseModel):
    intent: str
    entities: dict = {}

def detect_intent(text: str) -> IntentResult:
    t = text.lower()

    # --- Basic rule-based detection ---
    if "upload" in t or "add document" in t:
        return IntentResult(intent="upload_document")

    if "search" in t or "find" in t or "question" in t:
        return IntentResult(intent="semantic_query")

    if "list documents" in t or "show docs" in t:
        return IntentResult(intent="list_documents")

    if "products" in t or "catalog" in t:
        return IntentResult(intent="list_products")

    return IntentResult(intent="unknown")

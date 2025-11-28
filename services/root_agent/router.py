from services.root_agent.subagents.document_agent import DocumentAgent
from services.root_agent.subagents.search_agent import SearchAgent
from services.product_catalog_agent.agent import ProductCatalogAgent

doc_agent = DocumentAgent()
search_agent = SearchAgent()
product_agent = ProductCatalogAgent()

def route_intent(intent: str, entities: dict, user_id: str, raw_text: str):
    """
    Routes the detected intent to appropriate sub-agent.
    """

    if intent == "upload_document":
        return doc_agent.ingest_document(entities)

    if intent == "semantic_query":
        return search_agent.answer_query(raw_text)

    if intent == "list_documents":
        return doc_agent.list_documents()

    if intent == "list_products":
        return product_agent.list_products()

    return {"message": f"Unknown or unimplemented intent: {intent}"}

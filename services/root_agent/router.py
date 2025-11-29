# services/root_agent/router.py
from services.root_agent.subagents.document_agent import DocumentAgent
from services.root_agent.subagents.search_agent import SearchAgent
from services.root_agent.subagents.writer_agent import WriterAgent
from services.product_catalog_agent.agent import ProductCatalogAgent

doc_agent = DocumentAgent()
search_agent = SearchAgent()
writer_agent = WriterAgent()
product_agent = ProductCatalogAgent()

def route_intent(intent: str, entities: dict, user_id: str, raw_text: str):
    """
    Routes the detected intent to appropriate sub-agent.
    Synchronous: returns a dict result to the caller (RootAgent.handle).
    """

    if intent == "upload_document":
        return doc_agent.ingest_document(entities)

    if intent == "semantic_query":
        # take query from entities (NLU provides)
        query = entities.get("query") or raw_text
        search_out = search_agent.answer_query(query)

        # FILTER: prefer real excerpts; remove placeholder / empty excerpts
        filtered_results = []
        for r in search_out.get("results", []):
            ex = (r.get("excerpt") or "").strip()
            doc_id = r.get("document_id") or ""
            if ex and ex != doc_id:
                filtered_results.append(r)
        # if filtering removed everything, fallback to original results
        if not filtered_results:
            filtered_results = search_out.get("results", [])

        # now compose answer using filtered list
        written = writer_agent.compose(query, filtered_results)
        # combine into structured response
        return {
            "query": query,
            "search_results": search_out.get("results", []),  # keep full for debugging if needed
            "filtered_results_count": len(filtered_results),
            "answer": written.get("answer")
        }


    if intent == "list_documents":
        return doc_agent.list_documents()

    if intent == "list_products":
        return product_agent.list_products()

    return {"message": f"Unknown or unimplemented intent: {intent}"}

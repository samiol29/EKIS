# services/root_agent/subagents/writer_agent.py
from typing import Dict, Any, List
import textwrap

class WriterAgent:
    """
    Simple writer utility used synchronously by router.
    Converts search results into a human-readable answer string.
    """

    def __init__(self):
        pass

    def compose(self, query: str, results: List[Dict[str, Any]], max_chars: int = 1000) -> Dict[str, Any]:
        # pick top results with non-empty excerpt
        parts = []
        for r in results[:3]:
            ex = r.get("excerpt") or ""
            if ex:
                parts.append(ex.strip())
        if not parts:
            answer = f"Sorry — no relevant passages found for: {query}"
        else:
            joined = "\n\n".join(parts)
            answer = f"Answer (based on retrieved passages):\n\n{joined}"
            answer = textwrap.shorten(answer, width=max_chars, placeholder="...")
        return {"answer": answer, "query": query, "result_count": len(results)}

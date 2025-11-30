# services/root_agent/subagents/writer_agent.py
from typing import Dict, Any, List, Optional
import textwrap

class WriterAgent:
    """
    Simple writer utility used synchronously by router.
    Converts search results into a human-readable answer string.
    """

    def __init__(self):
        pass

    def compose(self, query: str, results: List[Dict[str, Any]], max_chars: int = 1000, memory: Optional[str] = None) -> Dict[str, Any]:
        # pick top results with non-empty excerpt
        parts = []
        for r in results[:3]:
            ex = r.get("excerpt") or ""
            if ex:
                parts.append(ex.strip())
        if not parts:
            answer_body = f"Sorry — no relevant passages found for: {query}"
        else:
            joined = "\n\n".join(parts)
            answer_body = f"Answer (based on retrieved passages):\n\n{joined}"
            # shorten to avoid massive outputs
            answer_body = textwrap.shorten(answer_body, width=max_chars, placeholder="...")

        # Prepend memory context if present (kept short)
        if memory:
            # keep memory to a small prefix (avoid exceeding max_chars too much)
            mem_short = textwrap.shorten(memory, width=200, placeholder="...")
            answer = f"Context: {mem_short}\n\n{answer_body}"
        else:
            answer = answer_body

        return {"answer": answer, "query": query, "result_count": len(results)}

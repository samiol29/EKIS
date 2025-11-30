# services/root_agent/root_agent.py
from uuid import uuid4
import datetime

from services.storage.memory_store import MemoryStore  # new import

from services.root_agent.schemas import UserMessage, AgentResponse
from services.root_agent.nlu import detect_intent
from services.root_agent.router import route_intent

class RootAgent:
    """
    The central orchestrator of EKIS.
    Handles:
    - Session memory (short-term + persistent facts)
    - Intent detection
    - Agent routing
    """

    def __init__(self):
        self.sessions = {}  # short-term memory: last 10 messages (kept for compatibility)
        # New: persistent memory store (hybrid)
        self.memory = MemoryStore()

    def handle(self, msg: UserMessage) -> AgentResponse:
        request_id = str(uuid4())

        # Update in-memory session list (keeps previous behavior)
        session = msg.session_id
        if session not in self.sessions:
            self.sessions[session] = []
        self.sessions[session].append(msg.text)
        self.sessions[session] = self.sessions[session][-10:]

        # Update persistent memory store (adds message + extracts facts)
        try:
            self.memory.add_message(session, msg.user_id, msg.text)
        except Exception:
            # never fail the agent because memory save failed
            pass

        # Detect intent
        intent_data = detect_intent(msg.text)

        # Build entities with memory context (so router and subagents can use it)
        entities = intent_data.entities.copy() if getattr(intent_data, "entities", None) else {}
        # Inject combined memory context string (short) and user facts separately
        entities["memory"] = self.memory.get_combined_memory(session, msg.user_id)
        entities["user_facts"] = self.memory.get_user_facts(msg.user_id)

        # Route
        result = route_intent(
            intent=intent_data.intent,
            entities=entities,
            user_id=msg.user_id,
            raw_text=msg.text,
        )

        # Build response
        return AgentResponse(
            request_id=request_id,
            intent=intent_data.intent,
            result=result,
            timestamp=str(datetime.datetime.now())
        )

from uuid import uuid4
import datetime

from services.root_agent.schemas import UserMessage, AgentResponse
from services.root_agent.nlu import detect_intent
from services.root_agent.router import route_intent

class RootAgent:
    """
    The central orchestrator of EKIS.
    Handles:
    - Session memory
    - Intent detection
    - Agent routing
    """

    def __init__(self):
        self.sessions = {}  # short-term memory: last 10 messages

    def handle(self, msg: UserMessage) -> AgentResponse:
        request_id = str(uuid4())

        # Update STM
        session = msg.session_id
        if session not in self.sessions:
            self.sessions[session] = []
        self.sessions[session].append(msg.text)
        self.sessions[session] = self.sessions[session][-10:]

        # Detect intent
        intent_data = detect_intent(msg.text)

        # Route
        result = route_intent(
            intent=intent_data.intent,
            entities=intent_data.entities,
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

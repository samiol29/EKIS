from pydantic import BaseModel

class UserMessage(BaseModel):
    user_id: str
    session_id: str
    text: str

class AgentResponse(BaseModel):
    request_id: str
    intent: str
    result: dict
    timestamp: str

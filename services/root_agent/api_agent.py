# services/root_agent/api_agent.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.root_agent.root_agent import RootAgent
from services.root_agent.schemas import UserMessage

app = FastAPI(title="EKIS Root Agent API")

_root_agent = RootAgent()

class AgentQuery(BaseModel):
    session_id: str
    user_id: str
    text: str

@app.post("/v1/agent/query")
async def agent_query(q: AgentQuery):
    """
    Run the full RootAgent flow for a single user query.
    Returns the AgentResponse (as dict).
    """
    try:
        msg = UserMessage(session_id=q.session_id, user_id=q.user_id, text=q.text)
        resp = _root_agent.handle(msg)

        # If resp is a pydantic model, convert it to dict
        try:
            return resp.dict()
        except Exception:
            return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI
from pydantic import BaseModel

from services.root_agent.root_agent import RootAgent
from services.root_agent.schemas import AgentResponse

app = FastAPI(title="EKIS Root Agent API")
root_agent = RootAgent()

class QueryRequest(BaseModel):
    user_id: str
    session_id: str
    text: str

@app.post("/query", response_model=AgentResponse)
def query(req: QueryRequest):
    return root_agent.handle(req)

@app.get("/health")
def health():
    return {"status": "ok"}

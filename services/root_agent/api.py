# services/root_agent/api.py

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from services.root_agent.root_agent import RootAgent
from services.root_agent.schemas import AgentResponse
from services.root_agent.subagents.document_agent import DocumentAgent

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="EKIS Root Agent API")

root_agent = RootAgent()
doc_agent = DocumentAgent()

class QueryRequest(BaseModel):
    user_id: str
    session_id: str
    text: str
    entities: dict | None = None

@app.post("/query", response_model=AgentResponse)
def query(req: QueryRequest):
    return root_agent.handle(req)

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), 
                      user_id: str = "u1", 
                      session_id: str = "s1"):

    filename = file.filename
    dst_path = os.path.join(UPLOAD_DIR, filename)

    # Save file
    try:
        content = await file.read()
        with open(dst_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Ingest document
    entities = {"filepath": dst_path, "filename": filename}
    result = doc_agent.ingest_document(entities)

    return {"status": "ok", "filepath": dst_path, "ingest_result": result}

# -------------------------
# INDEX STATUS ENDPOINT
# -------------------------
@app.get("/index_status")
def index_status():
    from services.storage.faiss_index import FAISSIndex
    from services.storage.document_store import DocumentStore

    idx = FAISSIndex()
    store = DocumentStore()

    return {
        "faiss_vectors": idx.index.ntotal,
        "id_map_length": len(idx.id_map),
        "document_count": len(store.documents)
    }

@app.get("/health")
def health():
    return {"status": "ok"}

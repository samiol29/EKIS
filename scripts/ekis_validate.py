# scripts/ekis_validate.py
import sys
import os
import importlib
import json
from pathlib import Path

print("=== EKIS CAPSTONE: Quick validator ===")
ROOT = Path(".").resolve()
print("Project root:", ROOT)

# Ensure project root is on sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    print("Added project root to sys.path")

# Helper for importing candidate modules
def try_import(candidates):
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            print(f"[OK] Imported module: {name}")
            return name, mod
        except Exception as e:
            print(f"[..] Could not import {name}: {e}")
    return None, None

# 1) FAISS manager import & checks
faiss_candidates = [
    "services.storage.faiss_index",
    "services.faiss_index",
    "storage.faiss_index",
    "faiss_index"
]
faiss_name, faiss_mod = try_import(faiss_candidates)

faiss = None
if faiss_mod:
    print("Inspecting FAISS manager module...")
    # instantiate if FAISSIndex class exists
    FAISSIndex = getattr(faiss_mod, "FAISSIndex", None)
    if FAISSIndex is None:
        print("[WARN] No FAISSIndex class found in the module.")
    else:
        try:
            # allow env overrides if you use them
            kwargs = {}
            for k in ("FAISS_INDEX_PATH", "ID_MAP_PATH", "EMB_PATH", "FAISS_MODEL_NAME", "EMBEDDING_DIM"):
                v = os.getenv(k)
                if v:
                    if k == "EMBEDDING_DIM":
                        kwargs["dim"] = int(v)
                    elif k == "FAISS_MODEL_NAME":
                        kwargs["model_name"] = v
                    else:
                        # map env names to constructor args heuristically
                        if "INDEX" in k:
                            kwargs["index_path"] = v
                        elif "ID_MAP" in k:
                            kwargs["id_map_path"] = v
                        elif "EMB" in k:
                            kwargs["emb_path"] = v
            print("Instantiating FAISSIndex with kwargs:", kwargs)
            manager = FAISSIndex(**kwargs) if kwargs else FAISSIndex()
            print("[OK] FAISSIndex instantiated.")
            # inspect internals
            idx = getattr(manager, "index", None)
            id_map = getattr(manager, "id_map", None)
            emb_file = getattr(manager, "emb_path", None) or getattr(manager, "embeddings", None)
            print(" - index object:", type(idx))
            try:
                ntotal = idx.ntotal if idx is not None else None
            except Exception:
                ntotal = None
            print(" - faiss.ntotal:", ntotal)
            if id_map is not None:
                print(" - id_map length:", len(id_map))
            else:
                print(" - id_map: not present")
            # check embeddings file if path provided
            if isinstance(emb_file, str) and Path(emb_file).exists():
                print(f" - embeddings file exists: {emb_file}")
            elif getattr(manager, "embeddings", None) is not None:
                print(" - embeddings array present in memory, shape:", getattr(manager, 'embeddings').shape)
            else:
                print(" - no persisted embeddings found or manager doesn't expose 'emb_path'/'embeddings' attr")
        except Exception as e:
            print("[ERROR] Failed to instantiate or inspect FAISSIndex:", e)
else:
    print("[WARN] No FAISS module found in candidates. Make sure your FAISS manager lives in services.storage.faiss_index or storage.faiss_index")

# 2) Document store import & checks
doc_candidates = [
    "services.storage.document_store",
    "services.document_store",
    "storage.document_store",
    "document_store"
]
doc_name, doc_mod = try_import(doc_candidates)
documents = []
if doc_mod:
    loader = None
    for fn in ("load_documents", "read_documents", "get_documents"):
        if hasattr(doc_mod, fn):
            loader = getattr(doc_mod, fn)
            break
    if loader:
        try:
            docs = loader()
            print(f"[OK] Loaded documents via {doc_name}.{loader.__name__}; count = {len(docs)}")
            documents = docs
        except Exception as e:
            print(f"[ERROR] Calling {doc_name}.{loader.__name__} failed: {e}")
    else:
        print(f"[WARN] {doc_name} has no standard loader functions (load_documents/read_documents/get_documents).")
else:
    # Try fallback file path
    docs_path = Path(os.getenv("DOCUMENTS_JSON", ROOT / "storage" / "documents.json"))
    if docs_path.exists():
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            print(f"[OK] Loaded documents from {docs_path}; count = {len(docs)}")
            documents = docs
        except Exception as e:
            print(f"[ERROR] Failed to read {docs_path}: {e}")
    else:
        print(f"[WARN] No documents found at {docs_path}. Create storage/documents.json or implement storage.document_store.load_documents()")

# 3) Validate id_map vs faiss.ntotal if both available
try:
    if 'manager' in locals() and getattr(manager, "index", None) is not None and getattr(manager, "id_map", None) is not None:
        nt = manager.index.ntotal
        imlen = len(manager.id_map)
        print(f"[CHECK] manager.index.ntotal = {nt}; len(manager.id_map) = {imlen}")
        if nt != imlen:
            print("[WARN] ntotal != id_map length — this often means id_map ordering doesn't match the index. Consider reindexing with your reindex function.")
        else:
            print("[OK] faiss.ntotal matches id_map length.")
except Exception as e:
    print("[INFO] Could not validate manager index/id_map:", e)

# 4) Try to import API and call endpoints via TestClient
print("\n--- API sanity checks ---")
api_candidates = ["api.retriever_api"]
api_name, api_mod = try_import(api_candidates)
if api_mod:
    # try to get FastAPI app
    app_obj = None
    if hasattr(api_mod, "app"):
        app_obj = getattr(api_mod, "app")
    elif hasattr(api_mod, "create_app"):
        try:
            app_obj = api_mod.create_app()
            print("[INFO] Created app via create_app()")
        except Exception as e:
            print("[ERROR] create_app() failed:", e)
    if app_obj is None:
        print("[WARN] Could not obtain FastAPI app object from api.retriever_api (no 'app' or create_app()).")
    else:
        try:
            from fastapi.testclient import TestClient
            client = TestClient(app_obj)
            print("[INFO] TestClient created.")
            # /v1/health
            try:
                r = client.get("/v1/health")
                print("GET /v1/health ->", r.status_code, r.json() if r.headers.get("content-type","").startswith("application/json") else r.text)
            except Exception as e:
                print("GET /v1/health failed:", e)
            # POST /v1/search sample
            try:
                payload = {"query": "test query", "top_k": 3}
                r = client.post("/v1/search", json=payload)
                print("POST /v1/search ->", r.status_code)
                try:
                    print("Response JSON keys:", list(r.json().keys()))
                    # print top result summary
                    if r.status_code == 200:
                        res = r.json()
                        print("took_ms:", res.get("took_ms"))
                        print("results_count:", len(res.get("results", [])))
                        if res.get("results"):
                            print("first result:", res["results"][0])
                except Exception:
                    print("Response text:", r.text[:500])
            except Exception as e:
                print("POST /v1/search failed:", e)
        except Exception as e:
            print("Failed to use TestClient:", e)
else:
    print("[WARN] api.retriever_api not importable. Start uvicorn from project root and ensure api/ package is present.")

# 5) Print relevant env vars
print("\n--- Environment Vars (relevant) ---")
keys = ["FAISS_INDEX_PATH","ID_MAP_PATH","EMB_PATH","DOCUMENTS_JSON","EMBEDDING_DIM","RETRIEVER_API_KEY","EKIS_DATA_DIR"]
for k in keys:
    print(f"{k} = {os.getenv(k)}")

print("\n=== Validator complete ===")

# scripts/rebuild_from_services_docs.py
"""
Rebuild storage/embeddings.npy, storage/id_map.json and storage/faiss.index
from services/storage/documents.json using the project's FAISSIndex manager.
"""

import json
from pathlib import Path
import sys

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

# Locations we will write to
OUT_DIR = ROOT / "storage"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = OUT_DIR / "faiss.index"
ID_MAP_PATH = OUT_DIR / "id_map.json"
EMB_PATH = OUT_DIR / "embeddings.npy"

# Where to read source documents from (your service copy)
SRC_DOCS_PATH = ROOT / "services" / "storage" / "documents.json"

if not SRC_DOCS_PATH.exists():
    print("ERROR: source documents not found at", SRC_DOCS_PATH)
    sys.exit(1)

# Import user's FAISSIndex manager
candidates = [
    "services.storage.faiss_index",
    "services.faiss_index",
    "storage.faiss_index",
    "faiss_index",
]
faiss_mod = None
for name in candidates:
    try:
        faiss_mod = __import__(name, fromlist=["FAISSIndex"])
        break
    except Exception:
        continue

if not faiss_mod or not hasattr(faiss_mod, "FAISSIndex"):
    print("ERROR: Could not import FAISSIndex from any candidate. Check services/storage/faiss_index.py")
    sys.exit(1)

# Load docs
with open(SRC_DOCS_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

# Normalize to dict {id: {text, metadata}}
docs_dict = {}
if isinstance(raw, list):
    for d in raw:
        doc_id = str(d.get("id") or d.get("doc_id") or d.get("filename") or "")
        if not doc_id:
            continue
        docs_dict[doc_id] = {"text": d.get("text", ""), "metadata": d.get("metadata", {})}
elif isinstance(raw, dict):
    for k, v in raw.items():
        doc_id = str(k)
        docs_dict[doc_id] = {"text": v.get("text", ""), "metadata": v.get("metadata", {})}
else:
    print("ERROR: unexpected documents.json format")
    sys.exit(1)

if len(docs_dict) == 0:
    print("ERROR: no documents found in", SRC_DOCS_PATH)
    sys.exit(1)

# Instantiate FAISSIndex pointing at storage paths
FAISSIndex = getattr(faiss_mod, "FAISSIndex")
manager = FAISSIndex(
    index_path=str(FAISS_INDEX_PATH),
    id_map_path=str(ID_MAP_PATH),
    emb_path=str(EMB_PATH),
)

print(f"Using FAISSIndex: model={manager.model_name}, dim={manager.dim}")
print("Recomputing embeddings for", len(docs_dict), "documents...")

# Rebuild (this will persist embeddings.npy, id_map.json and faiss.index)
manager.rebuild_from_documents(docs_dict)

print("Rebuild complete.")
print("Wrote:", FAISS_INDEX_PATH)
print("Wrote:", ID_MAP_PATH)
print("Wrote:", EMB_PATH)

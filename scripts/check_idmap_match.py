# scripts/check_idmap_match.py
"""
Check storage/documents.json IDs vs FAISS id_map.

Usage:
    python scripts/check_idmap_match.py
"""
import json
import importlib
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

# load docs.json
docs_path = ROOT / "storage" / "documents.json"
if not docs_path.exists():
    raise SystemExit(f"{docs_path} not found. Run generate_docs_template_from_idmap.py first or create documents.json.")

docs = json.load(open(docs_path, "r", encoding="utf-8"))
doc_ids = [d.get("id") for d in docs]

# import FAISS manager
CANDIDATES = ["services.storage.faiss_index", "services.faiss_index", "storage.faiss_index", "faiss_index"]
faiss_mod = None
for name in CANDIDATES:
    try:
        faiss_mod = importlib.import_module(name)
        break
    except Exception:
        continue

if not faiss_mod or not hasattr(faiss_mod, "FAISSIndex"):
    raise SystemExit("Could not import FAISSIndex module.")

manager = faiss_mod.FAISSIndex()
id_map = manager.id_map or []

print("documents.json count:", len(doc_ids))
print("faiss id_map count:", len(id_map))

missing_in_docs = [i for i in id_map if i not in doc_ids]
extra_in_docs = [i for i in doc_ids if i not in id_map]

print("IDs in faiss but missing in documents.json:", missing_in_docs)
print("IDs in documents.json but not in faiss:", extra_in_docs)

if not missing_in_docs:
    print("GOOD: All FAISS ids present in documents.json")
else:
    print("WARN: some FAISS IDs are missing in documents.json — add them or reindex.")

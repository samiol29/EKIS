# scripts/generate_docs_template_from_idmap.py
"""
Generate storage/documents.json template from FAISSIndex.id_map.

Usage:
    python scripts/generate_docs_template_from_idmap.py
"""
import json
import importlib
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

CANDIDATES = [
    "services.storage.faiss_index",
    "services.faiss_index",
    "storage.faiss_index",
    "faiss_index"
]

faiss_mod = None
for name in CANDIDATES:
    try:
        faiss_mod = importlib.import_module(name)
        print("Using FAISS manager module:", name)
        break
    except Exception:
        continue

if not faiss_mod or not hasattr(faiss_mod, "FAISSIndex"):
    raise SystemExit("Could not import FAISSIndex. Ensure services/storage/faiss_index.py or storage/faiss_index.py is importable.")

# instantiate manager (no kwargs by default)
manager = faiss_mod.FAISSIndex()
id_map = getattr(manager, "id_map", None)
if not id_map:
    raise SystemExit("FAISS manager has no id_map or it's empty. Cannot generate template.")

out_dir = ROOT / "storage"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "documents.json"

docs = []
for doc_id in id_map:
    docs.append({
        "id": doc_id,
        "text": "",          # <-- fill this manually
        "metadata": {}       # <-- add metadata if needed
    })

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(docs, f, ensure_ascii=False, indent=2)

print("Wrote template:", out_path)
print("Open the file and fill in `text` and `metadata` for each id, then restart the API.")

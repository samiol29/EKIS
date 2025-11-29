import os
import json
import faiss
import numpy as np

BASE = os.getcwd()

paths = {
    "FAISS_INDEX_PATH": "storage/faiss.index",
    "ID_MAP_PATH": "storage/id_map.json",
    "DOCUMENTS_PATH": "storage/documents.json",
    "EMBEDDINGS_PATH": "storage/embeddings.npy",
}

print("\n=== STORAGE PATH VERIFICATION ===\n")
print("Working directory:", BASE, "\n")

for name, rel_path in paths.items():
    abs_path = os.path.join(BASE, rel_path)
    print(f"{name}: {abs_path}")
    print("  Exists:", os.path.exists(abs_path))
    if not os.path.exists(abs_path):
        print()
        continue

    if name == "FAISS_INDEX_PATH":
        try:
            idx = faiss.read_index(abs_path)
            print("  FAISS load: OK")
            print("  ntotal:", idx.ntotal)
            try:
                print("  dimension:", idx.d)
            except:
                print("  dimension: (unavailable)")
        except Exception as e:
            print("  FAISS load: ERROR", e)

    elif name == "ID_MAP_PATH":
        try:
            with open(abs_path, "r") as f:
                id_map = json.load(f)
            print("  type:", type(id_map).__name__)
            print("  length:", len(id_map))
        except Exception as e:
            print("  ID map load: ERROR", e)

    elif name == "DOCUMENTS_PATH":
        try:
            with open(abs_path, "r") as f:
                docs = json.load(f)
            print("  num docs:", len(docs))
            print("  first doc ID:", docs[0].get("id"))
        except Exception as e:
            print("  documents load: ERROR", e)

    elif name == "EMBEDDINGS_PATH":
        try:
            arr = np.load(abs_path)
            print("  embeddings shape:", arr.shape)
        except Exception as e:
            print("  embeddings load: ERROR", e)

    print()

print("=== DONE ===")

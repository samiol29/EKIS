#!/usr/bin/env python3
"""
Fix missing embeddings and rebuild FAISS using only sentence-transformers (local).

Usage:
  python fix_missing_embeddings_st_only.py

Requirements:
  pip install sentence-transformers faiss-cpu
"""
import os, json, time, shutil, sys
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
STORAGE = os.path.join(ROOT, "storage")
FAISS_INDEX_PATH = os.path.join(STORAGE, "faiss.index")
DOCS_PATH = os.path.join(STORAGE, "documents.json")
IDMAP_PATH = os.path.join(STORAGE, "id_map.json")
EMB_PATH = os.path.join(STORAGE, "embeddings.npy")
BACKUP_DIR = os.path.join(STORAGE, f"backups_st_fix_{int(time.time())}")

# local-only embedding backend
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERROR: sentence-transformers not installed.")
    print("Install with: pip install sentence-transformers")
    sys.exit(2)

try:
    import faiss
except Exception as e:
    print("ERROR: faiss not installed or failed to import:", e)
    print("Install with: pip install faiss-cpu")
    sys.exit(2)

MODEL_NAME = "all-MiniLM-L6-v2"

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def backup_paths(paths):
    os.makedirs(BACKUP_DIR, exist_ok=True)
    saved = []
    for p in paths:
        if os.path.exists(p):
            dst = os.path.join(BACKUP_DIR, os.path.basename(p) + f".bak_{int(time.time())}")
            shutil.copy(p, dst)
            saved.append(dst)
    return saved

def build_doc_map(docs):
    # Build map doc_id -> doc dict (best-effort)
    m = {}
    if isinstance(docs, list):
        # try common id keys
        for d in docs:
            if not isinstance(d, dict):
                continue
            for key in ("id", "doc_id", "document_id", "uuid"):
                if key in d:
                    m[d[key]] = d
                    break
            else:
                # fallback: look for any string value matching a doc id (later)
                pass
    elif isinstance(docs, dict):
        m = docs.copy()
    return m

def extract_text_from_doc(d):
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        # prefer these fields
        for candidate in ("text", "content", "body", "doc_text", "document"):
            if candidate in d and isinstance(d[candidate], str):
                return d[candidate]
        # otherwise join short string fields
        parts = []
        for v in d.values():
            if isinstance(v, str) and len(v) < 5000:
                parts.append(v)
        if parts:
            return " ".join(parts)
        # fallback to JSON string
        try:
            return json.dumps(d, ensure_ascii=False)
        except:
            return str(d)
    return str(d)

def main():
    print("=== Local fix: sentence-transformers only ===")
    # check required files
    for req in [DOCS_PATH, IDMAP_PATH]:
        if not os.path.exists(req):
            print("Required file missing:", req)
            sys.exit(1)

    docs = load_json(DOCS_PATH)
    id_map = load_json(IDMAP_PATH)

    # derive ordered doc_ids from id_map (supports list or dict with numeric keys)
    if isinstance(id_map, list):
        doc_ids = id_map
    elif isinstance(id_map, dict):
        # prefer numeric-string keys mapping to doc_id
        numeric_keys = [k for k in id_map.keys() if k.isdigit()]
        if numeric_keys:
            # build ordered list by numeric index
            max_i = max(int(k) for k in numeric_keys)
            doc_ids = [id_map.get(str(i)) for i in range(max_i + 1)]
            # filter None
            doc_ids = [d for d in doc_ids if d is not None]
        else:
            # fallback: treat dict keys as doc_ids
            doc_ids = list(id_map.keys())
    else:
        print("Unrecognized id_map format:", type(id_map))
        sys.exit(1)

    n_docs = len(doc_ids)
    print("Documents to index (from id_map):", n_docs)

    existing_embs = None
    if os.path.exists(EMB_PATH):
        try:
            existing_embs = np.load(EMB_PATH)
            print("Found existing embeddings.npy shape:", existing_embs.shape)
        except Exception as e:
            print("Failed to load embeddings.npy:", e)
            existing_embs = None

    existing_count = existing_embs.shape[0] if existing_embs is not None else 0

    # determine missing indices
    missing_indices = []
    if existing_embs is None:
        missing_indices = list(range(n_docs))
    else:
        if existing_count < n_docs:
            missing_indices = list(range(existing_count, n_docs))
        elif existing_count > n_docs:
            print("NOTE: More embeddings than doc ids. Will truncate embeddings to match doc count.")
        else:
            print("Embeddings count matches doc count — will rebuild index only if needed.")

    # Build doc map for text lookup
    doc_map = build_doc_map(docs)

    model = SentenceTransformer(MODEL_NAME)
    if missing_indices:
        texts_to_embed = []
        for mi in missing_indices:
            did = doc_ids[mi]
            d = doc_map.get(did)
            if not d:
                # If docs is a list, try aligning by position as fallback
                if isinstance(docs, list) and mi < len(docs):
                    d = docs[mi]
            txt = extract_text_from_doc(d) if d is not None else str(did)
            texts_to_embed.append(txt)
        print(f"Computing {len(texts_to_embed)} embeddings locally with {MODEL_NAME}...")
        new_embs = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
        new_embs = new_embs.astype("float32")
        if existing_embs is None:
            merged = new_embs
        else:
            if existing_embs.shape[0] < n_docs:
                merged = np.vstack([existing_embs, new_embs])
            else:
                merged = existing_embs[:n_docs]
        print("Merged embeddings shape:", merged.shape)
        # backup and write
        backed = backup_paths([EMB_PATH, FAISS_INDEX_PATH, IDMAP_PATH])
        print("Backed up files:", backed)
        np.save(EMB_PATH, merged.astype("float32"))
        print("Wrote new embeddings.npy ->", EMB_PATH)
    else:
        # no missing; but if existing_count > n_docs, truncate
        if existing_embs is not None and existing_embs.shape[0] > n_docs:
            print("Truncating embeddings to match doc count.")
            merged = existing_embs[:n_docs]
            backed = backup_paths([EMB_PATH, FAISS_INDEX_PATH, IDMAP_PATH])
            np.save(EMB_PATH, merged.astype("float32"))
            print("Truncated and wrote embeddings.npy ->", EMB_PATH)
        else:
            print("No embeddings to compute. Using existing embeddings.npy")

    # final rebuild
    if not os.path.exists(EMB_PATH):
        print("ERROR: embeddings.npy missing after attempted fix. Aborting.")
        sys.exit(1)

    arr = np.load(EMB_PATH).astype("float32")
    if arr.ndim != 2:
        print("ERROR: embeddings.npy must be 2D. Found shape:", arr.shape)
        sys.exit(1)

    # normalize rows for cosine with IndexFlatIP
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arrn = arr / norms
    d = arrn.shape[1]
    print("Building FAISS IndexFlatIP with dim", d)
    idx = faiss.IndexFlatIP(d)
    idx.add(arrn)
    # backup and write
    backed = backup_paths([FAISS_INDEX_PATH, IDMAP_PATH])
    faiss.write_index(idx, FAISS_INDEX_PATH)
    print("Wrote faiss.index with ntotal:", idx.ntotal)
    print("Backups saved at:", BACKUP_DIR)
    print("DONE — run `python phase4_validator.py` to confirm status.")

if __name__ == "__main__":
    main()

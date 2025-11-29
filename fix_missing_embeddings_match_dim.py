#!/usr/bin/env python3
"""
Fix missing embeddings by computing them with a local model that matches existing embedding dimension.

- If embeddings.npy exists, reads its second dimension (target_dim).
- Chooses a sentence-transformers model that produces target_dim embeddings.
  Default mapping: 768 -> "all-mpnet-base-v2", 384 -> "all-MiniLM-L6-v2".
- Computes embeddings for missing indices and merges them.
- Rebuilds FAISS IndexFlatIP (normalizes vectors).
- Backs up originals to storage/backups_fix_dim_<ts>.

Usage:
  python fix_missing_embeddings_match_dim.py
"""
import os, json, time, shutil, sys
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
STORAGE = os.path.join(ROOT, "storage")
FAISS_INDEX_PATH = os.path.join(STORAGE, "faiss.index")
DOCS_PATH = os.path.join(STORAGE, "documents.json")
IDMAP_PATH = os.path.join(STORAGE, "id_map.json")
EMB_PATH = os.path.join(STORAGE, "embeddings.npy")
BACKUP_DIR = os.path.join(STORAGE, f"backups_fix_dim_{int(time.time())}")

# mapping of target dim -> recommended sentence-transformers model
DIM_TO_MODEL = {
    768: "all-mpnet-base-v2",
    384: "all-MiniLM-L6-v2",
    # add more if needed
}

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
    m = {}
    if isinstance(docs, list):
        for d in docs:
            if not isinstance(d, dict):
                continue
            for key in ("id", "doc_id", "document_id", "uuid"):
                if key in d:
                    m[d[key]] = d
                    break
    elif isinstance(docs, dict):
        m = docs.copy()
    return m

def extract_text_from_doc(d):
    if d is None:
        return ""
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        for candidate in ("text", "content", "body", "doc_text", "document"):
            if candidate in d and isinstance(d[candidate], str):
                return d[candidate]
        parts = []
        for v in d.values():
            if isinstance(v, str) and len(v) < 5000:
                parts.append(v)
        if parts:
            return " ".join(parts)
        try:
            return json.dumps(d, ensure_ascii=False)
        except:
            return str(d)
    return str(d)

# attempt to import sentence-transformers and faiss
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERROR: sentence-transformers not installed. pip install sentence-transformers")
    sys.exit(2)

try:
    import faiss
except Exception as e:
    print("ERROR: faiss not installed. pip install faiss-cpu")
    sys.exit(2)

def main():
    print("=== Fix missing embeddings matching target dim ===")
    for req in [DOCS_PATH, IDMAP_PATH]:
        if not os.path.exists(req):
            print("Required file missing:", req)
            sys.exit(1)

    docs = load_json(DOCS_PATH)
    id_map = load_json(IDMAP_PATH)

    # derive doc_ids ordered
    if isinstance(id_map, list):
        doc_ids = id_map
    elif isinstance(id_map, dict):
        numeric_keys = [k for k in id_map.keys() if k.isdigit()]
        if numeric_keys:
            max_i = max(int(k) for k in numeric_keys)
            doc_ids = [id_map.get(str(i)) for i in range(max_i + 1)]
            doc_ids = [d for d in doc_ids if d is not None]
        else:
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
    target_dim = None
    if existing_embs is not None:
        if existing_embs.ndim != 2:
            print("Embeddings must be 2D. Found shape:", existing_embs.shape)
            sys.exit(1)
        target_dim = existing_embs.shape[1]
        print("Target embedding dim inferred:", target_dim)
    else:
        # No existing embeddings: choose a default model dim 768
        target_dim = 768
        print("No existing embeddings present; defaulting target_dim to", target_dim)

    # pick model for this dim
    model_name = DIM_TO_MODEL.get(target_dim)
    if model_name is None:
        print(f"No default local model for target dim {target_dim}. Please update DIM_TO_MODEL mapping in script.")
        sys.exit(1)
    print("Selected local model:", model_name)

    existing_count = existing_embs.shape[0] if existing_embs is not None else 0
    missing_indices = []
    if existing_embs is None:
        missing_indices = list(range(n_docs))
    else:
        if existing_count < n_docs:
            missing_indices = list(range(existing_count, n_docs))
        elif existing_count > n_docs:
            print("More embeddings than doc ids; will truncate to match doc count.")
        else:
            print("Embeddings count matches doc count — only rebuilding index after potential fixes.")

    # map docs for text lookup
    doc_map = build_doc_map(docs)
    model = SentenceTransformer(model_name)

    if missing_indices:
        texts_to_embed = []
        for mi in missing_indices:
            did = doc_ids[mi]
            d = doc_map.get(did)
            if not d:
                if isinstance(docs, list) and mi < len(docs):
                    d = docs[mi]
            txt = extract_text_from_doc(d) if d is not None else str(did)
            texts_to_embed.append(txt)
        print(f"Computing {len(texts_to_embed)} embeddings locally with {model_name}...")
        new_embs = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
        new_embs = np.array(new_embs, dtype="float32")
        # verify shape matches target_dim
        if new_embs.ndim != 2 or new_embs.shape[1] != target_dim:
            print("ERROR: produced embeddings dim", new_embs.shape, "does not match target dim", target_dim)
            print("Aborting. Consider using a different model that produces the required dim.")
            sys.exit(1)
        # merge
        if existing_embs is None:
            merged = new_embs
        else:
            merged = np.vstack([existing_embs[:existing_count], new_embs])
        print("Merged embeddings shape:", merged.shape)
        backed = backup_paths([EMB_PATH, FAISS_INDEX_PATH, IDMAP_PATH])
        print("Backed up:", backed)
        np.save(EMB_PATH, merged.astype("float32"))
        print("Wrote new embeddings.npy ->", EMB_PATH)
    else:
        # maybe truncate if too long
        if existing_embs is not None and existing_embs.shape[0] > n_docs:
            print("Truncating embeddings to match doc count.")
            merged = existing_embs[:n_docs]
            backed = backup_paths([EMB_PATH, FAISS_INDEX_PATH, IDMAP_PATH])
            np.save(EMB_PATH, merged.astype("float32"))
            print("Truncated and wrote embeddings.npy ->", EMB_PATH)
        else:
            print("No missing embeddings to compute.")

    # rebuild index
    arr = np.load(EMB_PATH).astype("float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arrn = arr / norms
    d = arrn.shape[1]
    print("Building IndexFlatIP with dim", d)
    idx = faiss.IndexFlatIP(d)
    idx.add(arrn)
    backed = backup_paths([FAISS_INDEX_PATH, IDMAP_PATH])
    faiss.write_index(idx, FAISS_INDEX_PATH)
    print("Wrote faiss.index with ntotal:", idx.ntotal)
    print("Backups saved at:", BACKUP_DIR)
    print("DONE — run `python phase4_validator.py` to confirm.")

if __name__ == "__main__":
    main()

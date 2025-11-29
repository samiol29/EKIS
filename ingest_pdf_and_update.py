#!/usr/bin/env python3
"""
ingest_pdf_and_update.py

- Extract text from a PDF file
- Find the first placeholder doc in storage/documents.json (where text == id)
- Replace that doc's text with extracted PDF text and add metadata.source_filename
- Compute embedding for that doc with all-mpnet-base-v2 (768-d)
- Replace/append embedding in storage/embeddings.npy to keep alignment with storage/id_map.json
- Rebuild storage/faiss.index (IndexFlatIP) with normalized vectors
- Create backups under storage/backups_ingest_<ts>/

Usage:
  python ingest_pdf_and_update.py "uploads/sample_test_document.pdf"
"""
import sys, json, os, time, shutil
from pathlib import Path

# pdf extraction
from PyPDF2 import PdfReader

# embeddings & faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception:
    faiss = None

ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "storage"
DOCS_PATH = STORAGE / "documents.json"
IDMAP_PATH = STORAGE / "id_map.json"
EMB_PATH = STORAGE / "embeddings.npy"
FAISS_PATH = STORAGE / "faiss.index"

BACKUP_DIR = STORAGE / f"backups_ingest_{int(time.time())}"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-mpnet-base-v2"   # 768-d model to match existing embeddings

def extract_text_from_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    pages = []
    for p in reader.pages:
        try:
            txt = p.extract_text()
        except Exception:
            txt = None
        if txt:
            pages.append(txt)
    return "\n\n".join(pages).strip()

def backup_file(p: Path):
    if p.exists():
        dst = BACKUP_DIR / (p.name + f".bak_{int(time.time())}")
        shutil.copy2(p, dst)
        return dst
    return None

def main(pdf_path_arg: str):
    pdf_path = Path(pdf_path_arg).expanduser()
    print("PDF path:", pdf_path)
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("Warning: extracted text is empty.")
    else:
        print(f"Extracted text length: {len(text)} chars")

    # load docs
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"{DOCS_PATH} missing")
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))

    # load id_map
    if not IDMAP_PATH.exists():
        raise FileNotFoundError(f"{IDMAP_PATH} missing")
    id_map = json.loads(IDMAP_PATH.read_text(encoding="utf-8"))
    # id_map is typically list of ids (index -> doc_id)
    if isinstance(id_map, dict):
        # convert dict form that maps str(index)->doc_id into ordered list when possible
        keys = [k for k in id_map.keys() if k.isdigit()]
        if keys:
            max_k = max(int(k) for k in keys)
            id_list = []
            for i in range(max_k+1):
                id_list.append(id_map.get(str(i)))
            id_map_list = id_list
        else:
            id_map_list = list(id_map.values())
    else:
        id_map_list = id_map

    # find placeholder docs where text == id (our current state)
    placeholder_indices = []
    for idx, d in enumerate(docs):
        if isinstance(d, dict):
            if (d.get("text") or "").strip() == (d.get("id") or "").strip():
                placeholder_indices.append(idx)

    if not placeholder_indices:
        print("No placeholder docs (text==id) found. You can specify a target doc id manually.")
        # fallback: we will not modify anything
        return

    # choose first placeholder index (user can modify script to select another)
    target_idx = placeholder_indices[0]
    target_doc = docs[target_idx]
    target_id = target_doc.get("id")
    print("Replacing placeholder at index", target_idx, "doc_id:", target_id)

    # backup documents.json
    b = backup_file(DOCS_PATH)
    print("Backed up documents.json ->", b)

    # update doc text and metadata
    updated_doc = dict(target_doc)
    updated_doc["text"] = text
    md = updated_doc.get("metadata") or {}
    md = dict(md)
    md["source_filename"] = str(pdf_path.name)
    updated_doc["metadata"] = md
    docs[target_idx] = updated_doc

    # write new documents.json
    new_docs_path = STORAGE / "documents_updated_from_pdf.json"
    new_docs_path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote updated documents ->", new_docs_path)

    # Also overwrite canonical documents.json only after backup (optional)
    # Uncomment the next two lines if you want to directly replace documents.json:
    # DOCS_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    # print("Replaced storage/documents.json with updated documents (backup kept).")

    # Now compute embedding for the updated doc using sentence-transformers (768-d)
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    emb = np.array(emb, dtype=np.float32)
    print("Computed embedding shape:", emb.shape)

    # load existing embeddings.npy
    if not EMB_PATH.exists():
        print("embeddings.npy not found at", EMB_PATH, " — creating new embeddings with this single vector aligned to id_map length")
        existing = None
    else:
        existing = np.load(EMB_PATH)
        print("Existing embeddings shape:", existing.shape)

    # Determine target position in embeddings (should match id_map_list)
    # We will try to find index of target_id in id_map_list
    if target_id in id_map_list:
        target_pos = id_map_list.index(target_id)
        print("Found target_id in id_map at position:", target_pos)
    else:
        # fallback: if id_map_list length equals len(docs), assume same order
        if len(id_map_list) == len(docs):
            target_pos = target_idx
            print("id_map doesn't include target_id; assuming doc order. Using position:", target_pos)
        else:
            raise RuntimeError("Cannot determine target document position in id_map. id_map_list length != docs length and id not found.")

    # Ensure embedding dimension matches existing (if exists)
    target_dim = emb.shape[1]
    if existing is not None:
        if existing.ndim != 2:
            raise RuntimeError("existing embeddings must be 2D")
        if existing.shape[1] != target_dim:
            raise RuntimeError(f"Dimension mismatch: existing embeddings dim={existing.shape[1]}, new emb dim={target_dim}. Aborting.")
    else:
        # create empty existing with zeros but of size len(id_map_list)
        existing = np.zeros((len(id_map_list), target_dim), dtype=np.float32)

    # replace or append embedding at target_pos
    if target_pos < existing.shape[0]:
        existing[target_pos] = emb[0]
    elif target_pos == existing.shape[0]:
        # append
        existing = np.vstack([existing, emb])
    else:
        # pad with zeros up to target_pos then set
        pad = np.zeros((target_pos - existing.shape[0], target_dim), dtype=np.float32)
        existing = np.vstack([existing, pad, emb])

    # normalize embeddings and save backup
    b2 = backup_file(EMB_PATH)
    if b2:
        print("Backed up embeddings.npy ->", b2)
    np.save(EMB_PATH, existing.astype(np.float32))
    print("Wrote embeddings.npy with shape:", existing.shape)

    # rebuild faiss index
    if faiss is None:
        print("faiss not available in this environment. Skipping faiss index rebuild. Install faiss-cpu and re-run.")
        return

    # normalize rows
    norms = np.linalg.norm(existing, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arrn = existing / norms
    d = arrn.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(arrn)
    b3 = backup_file(FAISS_PATH)
    if b3:
        print("Backed up faiss.index ->", b3)
    faiss.write_index(idx, str(FAISS_PATH))
    print("Wrote faiss.index with ntotal:", idx.ntotal)

    # finally, if you want to replace canonical documents.json uncommented earlier, we can now do it.
    # For safety, we kept changes in documents_updated_from_pdf.json; if you want to swap:
    # shutil.copy2(DOCS_PATH, BACKUP_DIR / "documents.json.orig")
    # DOCS_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE. backups at:", BACKUP_DIR)
    print("Updated doc id:", target_id, "at position", target_pos)
    print("Wrote updated docs ->", new_docs_path)
    print("If everything looks good, you can replace storage/documents.json with the updated file and restart the server.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_pdf_and_update.py path/to/file.pdf")
        raise SystemExit(1)
    main(sys.argv[1])

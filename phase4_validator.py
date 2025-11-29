#!/usr/bin/env python3
"""
Phase 4 validator & safe repair tool.

Usage:
  python phase4_validator.py              # run diagnostics only
  python phase4_validator.py --repair     # run diagnostics then rebuild index (if embeddings present)

Edit CONFIG below to match your repo paths if different.
"""
import os, sys, json, argparse, time, shutil
import numpy as np

# try to import faiss with helpful error message
try:
    import faiss
except Exception as e:
    print("ERROR: faiss not installed or failed to import:", e)
    print("Install with: pip install faiss-cpu  (or faiss-gpu if you use GPU)")
    sys.exit(2)

# === CONFIG: Change these if your files are in different paths ===
CONFIG = {
    "faiss_index": "storage/faiss.index",
    "documents_json": "storage/documents.json",
    "embeddings_npy": "storage/embeddings.npy",    # optional but required for repair
    "id_map_json": "storage/id_map.json",          # expected alignment: idx -> doc_id or dict mapping
    "backups_dir": "storage/backups_phase4",
    "search_k": 5,
}

def safe_exists(p):
    return os.path.exists(p) and os.path.getsize(p) > 0

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_header(title):
    print("\n" + "="*6 + " " + title + " " + "="*6)

def diag_index(path):
    print_header("FAISS INDEX")
    if not safe_exists(path):
        print("FAISS index NOT FOUND at", path)
        return {"found": False}
    try:
        idx = faiss.read_index(path)
    except Exception as e:
        print("ERROR reading faiss index:", repr(e))
        return {"found": True, "read_error": True, "error": repr(e)}
    out = {"found": True, "read_error": False}
    try:
        ntotal = idx.ntotal
    except Exception:
        ntotal = None
    out["ntotal"] = ntotal
    # vector dimension
    try:
        d = faiss.vector_size(idx)
    except Exception:
        d = getattr(idx, "d", None)
    out["dim"] = d
    out["is_trained"] = getattr(idx, "is_trained", None)
    print("faiss index OK — ntotal:", ntotal, " dim:", d, " is_trained:", out["is_trained"])
    out["index_obj"] = idx
    return out

def diag_documents(path):
    print_header("DOCUMENTS JSON")
    if not safe_exists(path):
        print("documents.json NOT FOUND at", path)
        return {"found": False}
    try:
        docs = read_json(path)
        n = len(docs) if isinstance(docs, list) else len(docs.keys())
        sample = docs[0] if isinstance(docs, list) and docs else "no list docs"
        print("documents.json FOUND — entries:", n)
        return {"found": True, "count": n, "sample_keys": list(sample.keys()) if isinstance(sample, dict) else None}
    except Exception as e:
        print("ERROR reading documents.json:", e)
        return {"found": True, "read_error": True, "error": repr(e)}

def diag_embeddings(path):
    print_header("EMBEDDINGS")
    if not safe_exists(path):
        print("embeddings.npy NOT FOUND at", path)
        return {"found": False}
    try:
        arr = np.load(path)
        print("embeddings.npy FOUND — shape:", arr.shape, " dtype:", arr.dtype)
        norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
        print("first norm:", float(norms[0]) if norms.size>0 else "n/a", " mean norm:", float(np.mean(norms)))
        return {"found": True, "shape": arr.shape, "dtype": str(arr.dtype), "norms_mean": float(np.mean(norms))}
    except Exception as e:
        print("ERROR reading embeddings.npy:", e)
        return {"found": True, "read_error": True, "error": repr(e)}

def diag_idmap(path):
    print_header("ID MAP")
    if not safe_exists(path):
        print("id_map.json NOT FOUND at", path)
        return {"found": False}
    try:
        idmap = read_json(path)
        n = len(idmap) if isinstance(idmap, (list, dict)) else None
        sample = None
        if isinstance(idmap, list):
            sample = idmap[:3]
        elif isinstance(idmap, dict):
            sample = list(idmap.keys())[:3]
        print("id_map.json FOUND — entries:", n, " sample:", sample)
        return {"found": True, "len": n, "type": type(idmap).__name__}
    except Exception as e:
        print("ERROR reading id_map.json:", e)
        return {"found": True, "read_error": True, "error": repr(e)}

def search_sanity(index_obj, idmap, embeddings_path, k):
    print_header("SEARCH SANITY TEST")
    if index_obj is None:
        print("No index object provided — skipping search test")
        return {"skipped": True}
    # prepare a query vector
    if safe_exists(embeddings_path):
        try:
            arr = np.load(embeddings_path).astype("float32")
            q = np.mean(arr, axis=0).astype("float32")
        except Exception as e:
            print("Failed to load embeddings for query synthesis:", e)
            q = None
    else:
        q = None

    if q is None:
        # fallback: random vector of correct dim
        dim = faiss.vector_size(index_obj)
        q = np.random.randn(dim).astype("float32")
    # normalize if index uses inner product and vectors seem normalized (we don't know index metric here)
    qnorm = np.linalg.norm(q)
    if qnorm > 0:
        q = q / qnorm
    try:
        D, I = index_obj.search(q.reshape(1, -1), k)
        print("Search returned indices:", I.tolist(), " scores:", D.tolist())
        # map indices to doc ids where possible
        mapped = []
        for idx in I[0]:
            try:
                if isinstance(idmap, dict):
                    # idmap might map stringified ints to doc ids
                    key = str(int(idx))
                    mapped.append(idmap.get(key, f"missing_map_for_{idx}"))
                elif isinstance(idmap, list):
                    mapped.append(idmap[idx] if idx < len(idmap) else f"out_of_range_{idx}")
                else:
                    mapped.append(f"no_idmap_type")
            except Exception as e:
                mapped.append(f"map_error_{e}")
        print("Mapped doc ids (if id_map available):", mapped)
        return {"ok": True, "D": D.tolist(), "I": I.tolist(), "mapped": mapped}
    except Exception as e:
        print("ERROR during index.search():", repr(e))
        return {"ok": False, "error": repr(e)}

def backup_file(src, backups_dir):
    os.makedirs(backups_dir, exist_ok=True)
    if safe_exists(src):
        dst = os.path.join(backups_dir, os.path.basename(src) + f".bak_{int(time.time())}")
        shutil.copy(src, dst)
        print("Backed up", src, "->", dst)
        return dst
    return None

def rebuild_index(emb_path, out_path, id_map_path, backups_dir):
    print_header("REBUILD INDEX (SAFE MODE)")
    if not safe_exists(emb_path):
        print("Cannot rebuild: embeddings.npy not found at", emb_path)
        return {"rebuilt": False, "reason": "no_embeddings"}
    arr = np.load(emb_path).astype("float32")
    if arr.ndim != 2:
        print("embeddings.npy must be 2D (N, D). found shape:", arr.shape)
        return {"rebuilt": False, "reason": "bad_shape"}
    # normalize rows for cosine with IndexFlatIP
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    arr = arr / norms
    d = arr.shape[1]
    print("Creating IndexFlatIP with dim", d)
    idx = faiss.IndexFlatIP(d)
    idx.add(arr)
    print("Added vectors, ntotal=", idx.ntotal)
    # backup existing
    backup_file(out_path, backups_dir)
    backup_file(id_map_path, backups_dir)
    # write new index
    faiss.write_index(idx, out_path)
    print("Wrote new faiss index to", out_path)
    # ensure id_map length matches
    idmap = None
    if safe_exists(id_map_path):
        try:
            idmap = read_json(id_map_path)
            if isinstance(idmap, list) and len(idmap) != idx.ntotal:
                print("WARNING: id_map length != ntotal. You may need to regenerate id_map to match embeddings order.")
            elif isinstance(idmap, dict):
                print("id_map is dict; ensure keys map correctly to index ids.")
        except:
            print("Could not read id_map to validate length after rebuild.")
    return {"rebuilt": True, "ntotal": idx.ntotal}

def main():
    p = CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", action="store_true", help="If set, attempt to rebuild index using embeddings.npy (backs up existing index first).")
    args = parser.parse_args()

    idx_diag = diag_index(p["faiss_index"])
    docs_diag = diag_documents(p["documents_json"])
    emb_diag = diag_embeddings(p["embeddings_npy"])
    idmap_diag = diag_idmap(p["id_map_json"])

    index_obj = idx_diag.get("index_obj") if idx_diag.get("found") and not idx_diag.get("read_error") else None
    idmap_obj = None
    if idmap_diag.get("found") and not idmap_diag.get("read_error"):
        try:
            idmap_obj = read_json(p["id_map_json"])
        except:
            idmap_obj = None

    # If index found and dim available and embeddings found, check dim alignment
    if idx_diag.get("found") and emb_diag.get("found") and idx_diag.get("dim") and emb_diag.get("shape"):
        emb_dim = emb_diag["shape"][1]
        idx_dim = idx_diag["dim"]
        if emb_dim != idx_dim:
            print_header("DIMENSION MISMATCH")
            print(f"EMBEDDINGS dim = {emb_dim}  vs  FAISS index dim = {idx_dim}")
            print("Recommendation: rebuild index with embeddings (use --repair) or regenerate embeddings with the expected dim.")
        else:
            print("Dimension check PASS")

    # check id_map vs index count
    if idx_diag.get("found") and idmap_diag.get("found"):
        ntotal = idx_diag.get("ntotal")
        idlen = idmap_diag.get("len")
        if ntotal is not None and idlen is not None and ntotal != idlen:
            print_header("ID MAP vs INDEX LENGTH MISMATCH")
            print(f"faiss.ntotal = {ntotal}  vs  id_map length = {idlen}")
            print("Recommendation: ensure id_map aligns with embeddings order or rebuild index so alignment is restored.")
        else:
            print("ID map length check PASS or one of them missing.")

    # run search sanity
    search_result = None
    if index_obj is not None:
        search_result = search_sanity(index_obj, idmap_obj, p["embeddings_npy"], p["search_k"])
        if not search_result.get("ok", True):
            print("Search sanity FAILED:", search_result.get("error"))

    # If repair flag, attempt rebuild
    if args.repair:
        print_header("REPAIR MODE ENABLED")
        backups_dir = p["backups_dir"]
        os.makedirs(backups_dir, exist_ok=True)
        res = rebuild_index(p["embeddings_npy"], p["faiss_index"], p["id_map_json"], backups_dir)
        print("Rebuild result:", res)
        # re-run quick read to confirm
        try:
            idx2 = faiss.read_index(p["faiss_index"])
            print("Post-rebuild index ntotal:", idx2.ntotal, " dim:", faiss.vector_size(idx2))
        except Exception as e:
            print("Failed to read rebuilt index:", e)

    print("\n" + "="*8 + " DIAGNOSTICS COMPLETE " + "="*8)
    # guidance summary
    if not idx_diag.get("found"):
        print("ACTION: FAISS index missing. If you have embeddings.npy and id_map.json, run with --repair to rebuild.")
    elif idx_diag.get("read_error"):
        print("ACTION: faiss index read error. Try restoring from backups or rebuild with --repair.")
    else:
        print("If all PASS but behaviour still wrong, paste outputs for further debugging: the full outputs of this script, plus any retriever_api tracebacks.")
    print("Done.")

if __name__ == '__main__':
    main()

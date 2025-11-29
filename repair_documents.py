# repair_documents.py
import json
from pathlib import Path

SRC = Path("storage/documents.json")
OUT = Path("storage/documents_repaired.json")

if not SRC.exists():
    print("storage/documents.json not found. Aborting.")
    raise SystemExit(1)

docs = json.loads(SRC.read_text(encoding="utf-8"))
repaired = []
for d in docs:
    if not isinstance(d, dict):
        repaired.append({"id": str(d), "text": str(d), "metadata": {}})
        continue

    # pick doc id
    doc_id = d.get("id") or d.get("doc_id") or d.get("uuid") or None
    if not doc_id:
        # fallback: try to find any plausible id value
        for k, v in d.items():
            if isinstance(v, str) and len(v) == 36 and "-" in v:
                doc_id = v
                break
    if not doc_id:
        # create synthetic id (non-destructive)
        doc_id = f"doc_{len(repaired)}"

    # try to find text in common fields
    text = None
    for candidate in ("text", "content", "body", "document", "doc_text"):
        v = d.get(candidate)
        if isinstance(v, str) and v.strip():
            text = v.strip()
            break

    # try nested fields
    if not text:
        nested = d.get("fields") or d.get("data") or None
        if isinstance(nested, dict):
            for k, v in nested.items():
                if isinstance(v, str) and v.strip():
                    text = v.strip()
                    break

    # fallback: join short string fields (avoid huge dumps)
    if not text:
        parts = []
        for k, v in d.items():
            if k in ("id","doc_id","uuid","metadata"): 
                continue
            if isinstance(v, str) and 0 < len(v) < 5000:
                parts.append(v.strip())
        if parts:
            text = " ".join(parts)

    # final fallback: use id (so system has something)
    if not text:
        text = str(doc_id)

    # metadata preservation (small primitives only)
    metadata = {}
    if isinstance(d.get("metadata"), dict):
        metadata = d.get("metadata")
    else:
        for k, v in d.items():
            if k in ("id","doc_id","text","content","body","document","doc_text"):
                continue
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v

    nd = dict(d)
    nd["id"] = doc_id
    nd["text"] = text
    nd["metadata"] = metadata
    repaired.append(nd)

OUT.write_text(json.dumps(repaired, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote repaired documents to:", OUT)

import os
import json
import numpy as np
import faiss
from openai import OpenAI

# -------------------------------------------
# CONFIG
# -------------------------------------------

# Use your ENV variable or paste your key here (recommended: use env var)
client = OpenAI()

BASE_DIR = os.getcwd()
DOC_PATH = os.path.join(BASE_DIR, "storage", "documents.json")
EMB_PATH = os.path.join(BASE_DIR, "storage", "embeddings.npy")
FAISS_PATH = os.path.join(BASE_DIR, "storage", "faiss.index")
ID_MAP_PATH = os.path.join(BASE_DIR, "storage", "id_map.json")

MODEL = "text-embedding-3-small"   # 768-dim

# -------------------------------------------
# LOAD DOCUMENTS
# -------------------------------------------
print("\n=== STEP 1: Loading documents.json ===\n")

if not os.path.exists(DOC_PATH):
    raise FileNotFoundError(f"documents.json not found at {DOC_PATH}")

with open(DOC_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} documents.")
doc_ids = [doc["id"] for doc in documents]

# Extract text (if blank, we fallback to metadata fields)
def extract_text(doc):
    text = doc.get("text", "")
    if text.strip():
        return text
    # Fallback: join metadata values
    meta = doc.get("metadata", {})
    if meta:
        return " ".join(str(v) for v in meta.values())
    # Strong fallback: use ID (to still generate an embedding)
    return doc["id"]

texts = [extract_text(doc) for doc in documents]
print("Sample extracted text:", texts[0], "\n")

# -------------------------------------------
# COMPUTE EMBEDDINGS
# -------------------------------------------
print("=== STEP 2: Generating embeddings via OpenAI ===\n")

embeddings = []

for i, text in enumerate(texts):
    print(f"Generating embedding {i+1}/{len(texts)} ...")
    
    response = client.embeddings.create(
        model=MODEL,
        input=text
    )

    emb = response.data[0].embedding
    embeddings.append(emb)

embeddings = np.array(embeddings, dtype="float32")

print("\nEmbedding matrix shape:", embeddings.shape)  # should be (3, 768)

# -------------------------------------------
# SAVE embeddings.npy
# -------------------------------------------
print("\n=== STEP 3: Saving embeddings.npy ===\n")

np.save(EMB_PATH, embeddings)
print("Saved:", EMB_PATH)

# -------------------------------------------
# REBUILD FAISS INDEX
# -------------------------------------------
print("\n=== STEP 4: Rebuilding FAISS index ===\n")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, FAISS_PATH)
print("FAISS index rebuilt & saved:", FAISS_PATH)
print("FAISS ntotal:", index.ntotal)

# -------------------------------------------
# VALIDATION
# -------------------------------------------
print("\n=== STEP 5: VALIDATION ===\n")

# Validate id_map alignment
print("Validating id_map.json...")

with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    id_map = json.load(f)

if len(id_map) != embeddings.shape[0]:
    print("WARNING: id_map length mismatch!")
else:
    print("id_map matches embeddings count ✔")

# Validate FAISS
idx = faiss.read_index(FAISS_PATH)
print("FAISS loaded ntotal:", idx.ntotal)
print("FAISS dimension:", idx.d)

print("\n=== ALL DONE SUCCESSFULLY ===\n")

# services/storage/reindex_embeddings.py
from .document_store import DocumentStore
from .faiss_index import FAISSIndex

def rebuild_all():
    doc_store = DocumentStore()
    docs = doc_store.documents  # dict: doc_id -> meta
    fa = FAISSIndex()
    print(f"Rebuilding index from {len(docs)} documents...")
    fa.rebuild_from_documents(docs)
    print("Rebuild complete. index ntotal:", fa.index.ntotal)

if __name__ == "__main__":
    rebuild_all()

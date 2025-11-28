# services/storage/faiss_index.py
import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

class FAISSIndex:
    """
    FAISS index manager that:
    - keeps a list `id_map` of doc_ids (index order)
    - stores normalized embeddings in embeddings.npy
    - uses IndexFlatIP (inner product on unit vectors) so inner product == cosine similarity
    - exposes add/search/get_embedding and rebuild_from_documents utilities
    """

    def __init__(
        self,
        index_path="faiss.index",
        id_map_path="id_map.json",
        emb_path="embeddings.npy",
        model_name="sentence-transformers/all-mpnet-base-v2",
        dim=768
    ):
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.emb_path = emb_path
        self.model_name = model_name
        self.dim = dim

        # Load model
        self.model = SentenceTransformer(self.model_name)

        # Load or init id_map (list of doc_ids in vector order)
        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
        else:
            self.id_map = []

        # Load or init embeddings (float32 unit vectors)
        if os.path.exists(self.emb_path):
            self.embeddings = np.load(self.emb_path)
            if self.embeddings.dtype != np.float32:
                self.embeddings = self.embeddings.astype(np.float32)
        else:
            self.embeddings = np.zeros((0, self.dim), dtype=np.float32)

        # Create or load FAISS index (IndexFlatIP for cosine on normalized vectors)
        if os.path.exists(self.index_path) and self.embeddings.shape[0] > 0:
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            if self.embeddings.shape[0] > 0:
                self.index.add(self.embeddings)

    def _normalize(self, vec: np.ndarray):
        vec = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm

    def embed(self, text: str):
        """
        Return raw embedding (not necessarily normalized).
        """
        emb = self.model.encode([text])[0]
        emb = np.asarray(emb, dtype=np.float32)
        return emb

    def save_index(self):
        """
        Persist faiss.index, embeddings.npy, and id_map.json
        """
        faiss.write_index(self.index, self.index_path)
        np.save(self.emb_path, self.embeddings)
        with open(self.id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f)

    def add_document(self, doc_id: str, text: str, persist: bool = True):
        """
        Add a document (or chunk) to the index.
        - doc_id: string id
        - text: text (or chunk) used to build embedding
        - persist: if True, write files to disk after adding
        """
        emb = self.embed(text)
        emb = self._normalize(emb).reshape(1, -1).astype(np.float32)

        # add to faiss index
        self.index.add(emb)

        # append into embeddings array and id_map
        if self.embeddings.shape[0] == 0:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self.id_map.append(doc_id)

        if persist:
            self.save_index()

    def search(self, query: str, k=5):
        """
        Search the FAISS index.
        Returns a list of dicts:
          { "document_id": <doc_id>, "score": <cosine_sim>, "index": <position> }
        Score is inner-product on unit vectors => cosine similarity (higher is better).
        """
        if self.index.ntotal == 0:
            return []

        q_emb = self.embed(query)
        q_emb = self._normalize(q_emb).reshape(1, -1).astype(np.float32)

        sims, indices = self.index.search(q_emb, k)
        sims = sims[0].tolist()
        indices = indices[0].tolist()

        results = []
        for idx, sim in zip(indices, sims):
            if idx < 0 or idx >= len(self.id_map):
                continue
            results.append({
                "document_id": self.id_map[idx],
                "score": float(sim),
                "index": int(idx)
            })
        return results

    def get_embedding_by_docid(self, doc_id: str):
        """
        Return stored normalized embedding (np.ndarray) or None.
        """
        try:
            idx = self.id_map.index(doc_id)
        except ValueError:
            return None
        return self.embeddings[idx]

    def rebuild_from_documents(self, documents_dict):
        """
        Rebuild embeddings, id_map, and FAISS index from a dict of documents:
        documents_dict: { doc_id: { 'id':..., 'filename':..., 'text':... }, ... }

        This will overwrite embeddings.npy, id_map.json and faiss.index.
        """
        self.id_map = []
        embs = []

        for doc_id, meta in documents_dict.items():
            text = meta.get("text", "")
            emb = self.embed(text)
            emb = self._normalize(emb)
            embs.append(emb)
            self.id_map.append(doc_id)

        if len(embs) == 0:
            self.embeddings = np.zeros((0, self.dim), dtype=np.float32)
        else:
            self.embeddings = np.vstack(embs).astype(np.float32)

        # rebuild FAISS index
        self.index = faiss.IndexFlatIP(self.dim)
        if self.embeddings.shape[0] > 0:
            self.index.add(self.embeddings)

        # persist to disk
        self.save_index()

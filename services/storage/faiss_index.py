import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

class FAISSIndex:
    def __init__(self, index_path="faiss.index", id_map_path="id_map.json"):
        self.index_path = index_path
        self.id_map_path = id_map_path

        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.dim = 768

        # Load or create FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(self.dim)

        # Load or create id_map
        if os.path.exists(self.id_map_path):
            with open(self.id_map_path, "r") as f:
                self.id_map = json.load(f)
        else:
            self.id_map = []

    def embed(self, text: str):
        return self.model.encode([text])[0]

    def save_id_map(self):
        with open(self.id_map_path, "w") as f:
            json.dump(self.id_map, f)

    def add_document(self, doc_id: str, text: str):
        emb = self.embed(text)
        emb = np.array([emb]).astype("float32")

        self.index.add(emb)
        self.id_map.append(doc_id)

        # Persist both index + id_map
        faiss.write_index(self.index, self.index_path)
        self.save_id_map()

    def search(self, query: str, k=5):
        if self.index.ntotal == 0:
            return []

        query_emb = self.embed(query)
        query_emb = np.array([query_emb]).astype("float32")

        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            results.append({
                "document_id": self.id_map[idx],
                "distance": float(dist)
            })
        
        return results

# api/retriever_api.py
"""
retriever_api.py

Production-minded FastAPI service to serve your EKIS retriever (Phase 4)
"""
from __future__ import annotations
import os
import json
import time
import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import datetime

import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Try optional dependencies
try:
    import faiss
except Exception:
    faiss = None

try:
    from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
    SEARCH_COUNTER = Counter('ekis_search_requests_total', 'Total number of search requests')
except Exception:
    PROMETHEUS_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False

# Simple LRU cache for search results to avoid repeated reranking
try:
    from cachetools import LRUCache, TTLCache
except Exception:
    LRUCache = None

# Optional aioredis for distributed cache (not required)
try:
    import aioredis
    REDIS_AVAILABLE = True
except Exception:
    aioredis = None
    REDIS_AVAILABLE = False

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("retriever_api")

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = Path(os.getenv("EKIS_DATA_DIR", "./storage"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", DATA_DIR / "faiss.index"))
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_JSON", DATA_DIR / "documents.json"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
API_KEY = os.getenv("RETRIEVER_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "1024"))
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------
# Debugging / Dump options (toggleable)
# ---------------------------
DEBUG_ENABLED = os.getenv("EKIS_DEBUG_ENABLED", "true").lower() in ("1", "true", "yes")
DEBUG_DIR = DATA_DIR / "retriever_debug"
DEBUG_MAX_TEXT_LEN = int(os.getenv("EKIS_DEBUG_MAX_TEXT_LEN", "1024"))
DUMP_RETRIEVAL_PAYLOAD = os.getenv("EKIS_DUMP_RETRIEVAL_PAYLOAD", "true").lower() in ("1", "true", "yes")

if DEBUG_ENABLED:
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def safe_truncate(text: Optional[str], max_len: int = DEBUG_MAX_TEXT_LEN) -> Optional[str]:
    if text is None:
        return None
    t = str(text)
    if len(t) > max_len:
        return t[:max_len] + "..."
    return t

def dump_debug_payload(label: str, payload: Dict[str, Any]):
    if not DEBUG_ENABLED:
        return None
    try:
        ts = int(time.time() * 1000)
        fn = DEBUG_DIR / f"{label}_{ts}.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved debug payload -> %s", str(fn))
        return str(fn)
    except Exception as e:
        logger.exception("Failed to dump debug payload: %s", e)
        return None

# ---------------------------
# Utility types
# ---------------------------
class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    id: str
    score: float
    excerpt: Optional[str]
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(10, gt=0, le=100)
    filters: Optional[Dict[str, Any]] = None
    use_rerank: bool = True

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    took_ms: float

# ---------------------------
# Embedding model abstraction
# ---------------------------
class EmbeddingModel:
    """Abstract embedding model. Implement `embed` to return numpy array (N, D)"""
    async def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

class DummyEmbeddingModel(EmbeddingModel):
    """Fallback embedding: uses a simple hashing-based vector. NOT for production; used for local testing."""
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    async def embed(self, texts: List[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            for j in range(self.dim):
                arr[i, j] = ((h >> (j % 32)) & 0xFF) / 255.0
        # normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

# ---------------------------
# FAISS + Document store loader
# ---------------------------
class RetrieverIndex:
    def __init__(self, index_path: Path, documents_path: Path, embedding_dim: int):
        self.index_path = index_path
        self.documents_path = documents_path
        self.embedding_dim = embedding_dim

        self.index = None
        self.documents: Dict[str, Document] = {}
        self.id_map: List[str] = []  # maps faiss internal ids -> doc_id
        # optional external embeddings array (numpy)
        self._external_embeddings: Optional[np.ndarray] = None

    def load(self):
        logger.info("Loading FAISS index from %s", str(self.index_path))
        if faiss is None:
            logger.warning("faiss not installed. Search will be a linear scan fallback.")
            self.index = None
        else:
            if not self.index_path.exists():
                logger.warning("FAISS index file %s not found.", self.index_path)
                self.index = None
            else:
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    logger.info("FAISS index loaded.")
                except Exception as e:
                    logger.exception("Loading FAISS index failed: %s", e)
                    self.index = None

        # Load documents
        if not self.documents_path.exists():
            raise FileNotFoundError(f"documents.json not found at {self.documents_path}")
        with open(self.documents_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        # expected format: list of {id, text, metadata}
        for d in docs:
            doc = Document(id=d["id"], text=d.get("text", ""), metadata=d.get("metadata", {}))
            self.documents[doc.id] = doc
        # By default, id_map follows documents order unless overridden externally
        self.id_map = list(self.documents.keys())
        logger.info("Loaded %d documents.", len(self.documents))

    def search(self, query_vectors: np.ndarray, top_k: int = 10, filters: Optional[Dict[str, Any]] = None):
        """
        Search the index.
        Priority:
         1) Use self.index (FAISS) if available.
         2) If not, but self._external_embeddings exists (numpy array aligned with id_map),
            compute cosine similarities against that.
         3) Fallback: hashing-based dummy vectors.
        Returns list of (doc_id, score) tuples.
        """
        # 1) FAISS index pathway
        if self.index is not None:
            try:
                D, I = self.index.search(query_vectors.astype(np.float32), top_k)
                results = []
                for i_row in range(I.shape[0]):
                    row_ids = I[i_row]
                    row_scores = D[i_row]
                    for idx, sc in zip(row_ids, row_scores):
                        if idx < 0 or idx >= len(self.id_map):
                            # skip invalid indices returned by faiss (faiss uses -1 for empty slots)
                            continue
                        doc_id = self.id_map[idx]
                        doc = self.documents.get(doc_id)
                        if self._apply_filters(doc, filters):
                            results.append((doc_id, float(sc)))
                return results
            except Exception:
                logger.exception("FAISS search failed; falling back to numpy-based search.")

        # 2) Use external embeddings if present (preferred over hashing fallback)
        if hasattr(self, "_external_embeddings") and getattr(self, "_external_embeddings", None) is not None:
            try:
                doc_embs = self._external_embeddings  # shape (N, D)
                if doc_embs.ndim != 2:
                    raise ValueError("external embeddings must be 2D array")
                # normalize doc_embs if not already normalized
                doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12
                doc_matrix = doc_embs / doc_norms

                # normalize query_vectors
                q = query_vectors.astype(np.float32)
                q_norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
                q = q / q_norms  # shape (Q, D)

                # compute dot product for the first query vector only
                q0 = q[0]
                scores = doc_matrix.dot(q0)
                top_idx = np.argsort(-scores)[:top_k]
                results = []
                for idx in top_idx:
                    if idx < 0 or idx >= len(self.id_map):
                        continue
                    doc_id = self.id_map[idx]
                    doc = self.documents.get(doc_id)
                    if self._apply_filters(doc, filters):
                        results.append((doc_id, float(scores[idx])))
                return results
            except Exception:
                logger.exception("External-embeddings search failed; falling back to hashing-based search.")

        # 3) Hashing-based fallback
        logger.info("FAISS index missing: falling back to simple similarity using dummy vectors")
        vecs = []
        ids = []
        if not self.documents:
            return []
        # Determine dim for hashing fallback
        if hasattr(self, "embedding_dim"):
            dim = self.embedding_dim
        else:
            try:
                dim = int(query_vectors.shape[1])
            except Exception:
                dim = 768
        for doc_id, doc in self.documents.items():
            if not self._apply_filters(doc, filters):
                continue
            ids.append(doc_id)
            h = abs(hash(doc.text or ""))
            v = np.zeros((dim,), dtype=np.float32)
            for j in range(dim):
                v[j] = ((h >> (j % 32)) & 0xFF) / 255.0
            vecs.append(v)
        if len(vecs) == 0:
            return []
        doc_matrix = np.vstack(vecs)
        doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-12
        doc_matrix = doc_matrix / doc_norms
        q = query_vectors[0]
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = doc_matrix.dot(q)
        top_idx = np.argsort(-scores)[:top_k]
        results = [(ids[i], float(scores[i])) for i in top_idx]
        return results

    def _apply_filters(self, doc: Document, filters: Optional[Dict[str, Any]]):
        if not filters:
            return True
        if doc is None:
            return False
        for k, v in filters.items():
            if k not in doc.metadata:
                return False
            if doc.metadata[k] != v:
                return False
        return True

    def get(self, doc_id: str) -> Optional[Document]:
        return self.documents.get(doc_id)

# ---------------------------
# Reranker wrapper
# ---------------------------
class SemanticReranker:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self.model = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(self.model_name)
                logger.info("Loaded cross-encoder model %s", self.model_name)
            except Exception as e:
                logger.exception("Failed to load cross-encoder: %s", e)
                self.model = None

    def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        if self.model is None:
            # fallback: sort by original score
            return sorted(candidates, key=lambda x: x.score, reverse=True)
        pairs = [[query, (c.excerpt or c.metadata.get('title') or '')] for c in candidates]
        scores = self.model.predict(pairs)
        for i, s in enumerate(scores):
            candidates[i].score = float(s)
        return sorted(candidates, key=lambda x: x.score, reverse=True)

# ---------------------------
# Caching
# ---------------------------
class SimpleCache:
    def __init__(self, max_items: int = CACHE_MAX_ITEMS, ttl: int = CACHE_TTL_SECONDS):
        if LRUCache is not None:
            self.cache = TTLCache(maxsize=max_items, ttl=ttl)
        else:
            self.cache = {}
        self.ttl = ttl

    async def get(self, key: str):
        if LRUCache is not None:
            return self.cache.get(key)
        else:
            entry = self.cache.get(key)
            if not entry:
                return None
            ts, val = entry
            if time.time() - ts > self.ttl:
                self.cache.pop(key, None)
                return None
            return val

    async def set(self, key: str, value: Any):
        if LRUCache is not None:
            self.cache[key] = value
        else:
            self.cache[key] = (time.time(), value)

# ---------------------------
# API key middleware
# ---------------------------
class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, header_name: str = "authorization"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        if API_KEY is None:
            # no auth configured
            return await call_next(request)
        auth = request.headers.get("Authorization") or request.headers.get("authorization")
        if not auth or not auth.startswith("Bearer "):
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": "Missing API key"})
        token = auth.split(" ", 1)[1]
        if token != API_KEY:
            return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": "Invalid API key"})
        return await call_next(request)

# ---------------------------
# Helper: prepare candidates from raw_results
# ---------------------------
def prepare_candidates_from_raw_results(raw_results: List[tuple], retriever: RetrieverIndex, max_excerpt: int = 300):
    """
    raw_results: list of (doc_id, score)
    returns: list of SearchResult (safe) and a list of candidate texts (for reranker)
    """
    candidates = []
    candidate_texts = []
    for doc_id, score in raw_results:
        doc = retriever.get(doc_id) if doc_id is not None else None
        excerpt = None
        meta = {}
        if doc:
            txt = doc.text or ""
            excerpt = (txt[:max_excerpt] + "...") if len(txt) > max_excerpt else txt
            meta = doc.metadata or {}
            candidate_texts.append(safe_truncate(txt))
        else:
            candidate_texts.append(None)
        candidates.append(SearchResult(id=doc_id or "", score=score, excerpt=excerpt, metadata=meta))
    return candidates, candidate_texts

# ---------------------------
# App factory
# ---------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="EKIS Retriever API", version="1.0")
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.getenv("CORS_ORIGIN", "*")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach resources
    app.state.retriever: Optional[RetrieverIndex] = None
    app.state.embedding_model: Optional[EmbeddingModel] = None
    app.state.reranker: Optional[SemanticReranker] = None
    app.state.cache: Optional[SimpleCache] = None
    app.state.faiss_manager = None
    app.state.faiss_index_object = None

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting EKIS retriever API...")

        retriever = None
        faiss_manager = None
        docs_list = None

        import importlib

        faiss_candidates = ["services.storage.faiss_index", "storage.faiss_index", "services.faiss_index"]
        doc_candidates = ["services.storage.document_store", "storage.document_store", "services.document_store"]

        faiss_mod = None
        doc_mod = None
        for m in faiss_candidates:
            try:
                faiss_mod = importlib.import_module(m)
                logger.info("Imported FAISS module: %s", m)
                break
            except Exception:
                continue

        for m in doc_candidates:
            try:
                doc_mod = importlib.import_module(m)
                logger.info("Imported document module: %s", m)
                break
            except Exception:
                continue

        try:
            # Instantiate user's FAISSIndex if available
            if faiss_mod and hasattr(faiss_mod, "FAISSIndex"):
                kwargs = {}
                if os.getenv("FAISS_INDEX_PATH"):
                    kwargs["index_path"] = os.getenv("FAISS_INDEX_PATH")
                if os.getenv("ID_MAP_PATH"):
                    kwargs["id_map_path"] = os.getenv("ID_MAP_PATH")
                if os.getenv("EMB_PATH"):
                    kwargs["emb_path"] = os.getenv("EMB_PATH")
                if os.getenv("FAISS_MODEL_NAME"):
                    kwargs["model_name"] = os.getenv("FAISS_MODEL_NAME")
                if os.getenv("EMBEDDING_DIM"):
                    kwargs["dim"] = int(os.getenv("EMBEDDING_DIM"))

                try:
                    faiss_manager = faiss_mod.FAISSIndex(**kwargs) if kwargs else faiss_mod.FAISSIndex()
                    logger.info("Instantiated FAISSIndex from %s", faiss_mod.__name__)
                except Exception as e:
                    logger.exception("Failed to instantiate FAISSIndex: %s", e)
                    faiss_manager = None

            # Load documents using document_store module if available
            if doc_mod:
                for fn in ("load_documents", "read_documents", "get_documents"):
                    if hasattr(doc_mod, fn):
                        try:
                            docs_list = getattr(doc_mod, fn)()
                            logger.info("Loaded documents via %s.%s (count=%d)", doc_mod.__name__, fn, len(docs_list) if docs_list is not None else 0)
                            break
                        except Exception as e:
                            logger.exception("Error calling %s.%s: %s", doc_mod.__name__, fn, e)
                            docs_list = None

            # Fallback: try reading storage/documents.json
            if not docs_list:
                # treat empty list same as missing
                docs_path = Path(os.getenv("DOCUMENTS_JSON", Path("storage") / "documents.json"))
                if docs_path.exists():
                    try:
                        with open(docs_path, "r", encoding="utf-8") as f:
                            loaded = json.load(f)
                        # normalize loaded into list
                        if isinstance(loaded, dict):
                            # dict keyed by id -> values
                            loaded_list = list(loaded.values())
                        else:
                            loaded_list = loaded
                        docs_list = loaded_list
                        logger.info("Loaded documents from %s (count=%d)", docs_path, len(docs_list))
                    except Exception as e:
                        logger.exception("Failed to read %s: %s", docs_path, e)
                        docs_list = None

            # If still no docs_list and we have a faiss_manager with id_map, create minimal docs
            if (not docs_list or len(docs_list) == 0) and faiss_manager is not None and getattr(faiss_manager, "id_map", None):
                docs_list = []
                for doc_id in faiss_manager.id_map:
                    # create placeholder doc so retriever can return ids/excerpts
                    docs_list.append({"id": doc_id, "text": "", "metadata": {}})                    
                logger.info("Created minimal documents from faiss_manager.id_map (count=%d)", len(docs_list))

            # Build RetrieverIndex wrapper and attach faiss internals if available
            retriever = RetrieverIndex(FAISS_INDEX_PATH, DOCUMENTS_PATH, EMBEDDING_DIM)

            if faiss_manager is not None:
                try:
                    # Prefer the manager's index object if present
                    if hasattr(faiss_manager, "index") and getattr(faiss_manager, "index") is not None:
                        retriever.index = faiss_manager.index
                    # Ensure id_map alignment: prefer manager.id_map
                    if hasattr(faiss_manager, "id_map") and getattr(faiss_manager, "id_map") is not None:
                        retriever.id_map = faiss_manager.id_map
                    # Attach persisted embeddings if the manager exposed them
                    if hasattr(faiss_manager, "embeddings") and getattr(faiss_manager, "embeddings") is not None:
                        retriever._external_embeddings = faiss_manager.embeddings
                    # Store manager on app.state for stable access
                    app.state.faiss_manager = faiss_manager
                    # Also store a direct reference to the faiss Index on app.state for other code paths
                    try:
                        app.state.faiss_index_object = faiss_manager.index if hasattr(faiss_manager, "index") else None
                    except Exception:
                        app.state.faiss_index_object = None
                    logger.info(
                        "Attached faiss_manager internals: index=%s, id_map_len=%s, embeddings=%s",
                        type(getattr(faiss_manager, "index", None)).__name__ if getattr(faiss_manager, "index", None) is not None else None,
                        len(getattr(faiss_manager, "id_map", [])) if getattr(faiss_manager, "id_map", None) is not None else 0,
                        "present" if getattr(faiss_manager, "embeddings", None) is not None else "absent"
                    )
                except Exception as e:
                    logger.exception("Failed to attach faiss_manager internals: %s", e)

            # --- ensure retriever has a usable FAISS index: try manager.index, else try reading disk index
            if getattr(retriever, "index", None) is None:
                try:
                    # prefer faiss_manager.index if available
                    mgr_idx = getattr(faiss_manager, "index", None)
                    if mgr_idx is not None:
                        retriever.index = mgr_idx
                    elif FAISS_INDEX_PATH.exists() and faiss is not None:
                        logger.info("Attempting to read FAISS index from disk at %s", FAISS_INDEX_PATH)
                        try:
                            retriever.index = faiss.read_index(str(FAISS_INDEX_PATH))
                            logger.info("Loaded faiss index into retriever from disk.")
                        except Exception as e:
                            logger.exception("Failed to read faiss index from disk: %s", e)
                except Exception:
                    logger.exception("Error while attaching fallback FAISS index to retriever.")

            # -----------------------------
            # Populate retriever.documents from docs_list (robust text extraction)
            # -----------------------------
            def _extract_text_from_doc_obj(d):
                """
                Best-effort extraction of human text from a document object.
                Looks for common fields and falls back to joining short string fields.
                """
                if d is None:
                    return ""
                if isinstance(d, str):
                    return d
                # prefer explicit fields
                for candidate in ("text", "content", "body", "document", "doc_text"):
                    if isinstance(d, dict) and candidate in d and isinstance(d[candidate], str) and d[candidate].strip():
                        return d[candidate].strip()
                # sometimes text is nested under 'fields' or similar
                if isinstance(d, dict):
                    for candidate in ("fields", "data"):
                        if candidate in d and isinstance(d[candidate], dict):
                            for k,v in d[candidate].items():
                                if isinstance(v, str) and v.strip():
                                    return v.strip()
                # fallback: join short string values from the dict
                if isinstance(d, dict):
                    parts = []
                    for v in d.values():
                        if isinstance(v, str) and 0 < len(v) < 8000:
                            parts.append(v.strip())
                    if parts:
                        # prefer longer joined string but keep reasonable size
                        joined = " ".join(parts)
                        return joined if len(joined) < 20000 else joined[:20000]
                # last resort: JSON-dump
                try:
                    return json.dumps(d, ensure_ascii=False)
                except Exception:
                    return str(d)

            docs_dict = {}
            if docs_list:
                for d in docs_list:
                    if isinstance(d, Document):
                        doc = d
                    else:
                        # determine doc_id from a few possible keys
                        doc_id = None
                        if isinstance(d, dict):
                            for key in ("id", "doc_id", "uuid", "filename"):
                                v = d.get(key)
                                if v:
                                    doc_id = str(v)
                                    break
                        # if still None, try to derive from content or skip
                        if not doc_id:
                            # try to find any plausible id in dict values
                            if isinstance(d, dict):
                                for k,v in d.items():
                                    if isinstance(v, str) and len(v) == 36 and "-" in v:
                                        doc_id = v
                                        break
                        if not doc_id:
                            # skip documents with no id
                            logger.debug("Skipping document with no id: %s", safe_truncate(str(d)))
                            continue
                        # extract text robustly
                        text = _extract_text_from_doc_obj(d)
                        # metadata: prefer 'metadata' or everything else except text/id
                        metadata = {}
                        if isinstance(d, dict):
                            if "metadata" in d and isinstance(d["metadata"], dict):
                                metadata = d["metadata"]
                            else:
                                # include other fields as metadata, but avoid huge text fields
                                for k,v in d.items():
                                    if k in ("id","doc_id","text","content","body","document","doc_text"):
                                        continue
                                    # only include small primitives
                                    if isinstance(v, (str,int,float,bool)):
                                        metadata[k] = v
                        doc = Document(id=doc_id, text=text, metadata=metadata)
                    docs_dict[doc.id] = doc

            retriever.documents = docs_dict


            # If id_map missing, build from documents order
            if not getattr(retriever, "id_map", None):
                retriever.id_map = list(docs_dict.keys())

            # Validate FAISS ntotal vs id_map if index present
            try:
                if retriever.index is not None:
                    faiss_ntotal = retriever.index.ntotal
                    if faiss_ntotal != len(retriever.id_map):
                        logger.warning("FAISS ntotal (%d) != id_map length (%d). This may indicate a mismatch.", faiss_ntotal, len(retriever.id_map))
                    else:
                        logger.info("FAISS ntotal matches id_map length (%d).", faiss_ntotal)
            except Exception:
                logger.debug("Couldn't validate FAISS ntotal vs id_map (faiss missing or index incompatible).")

            # --- Force-load FAISS index from disk into retriever if possible (ensures TestClient sees it) ---
            try:
                if getattr(retriever, "index", None) is None:
                    # first prefer any manager-provided index
                    mgr_idx = getattr(faiss_manager, "index", None) if 'faiss_manager' in locals() else None
                    if mgr_idx is not None:
                        retriever.index = mgr_idx
                        logger.info("Attached faiss_manager.index to retriever (from memory).")
                    else:
                        # then try reading the canonical disk path
                        if FAISS_INDEX_PATH and Path(FAISS_INDEX_PATH).exists() and faiss is not None:
                            try:
                                retriever.index = faiss.read_index(str(FAISS_INDEX_PATH))
                                logger.info("Loaded FAISS index from disk into retriever: %s", FAISS_INDEX_PATH)
                            except Exception as e:
                                logger.exception("Failed to read FAISS index from disk: %s", e)
                        else:
                            logger.debug("No faiss_manager.index and storage faiss file missing or faiss not available.")
            except Exception:
                logger.exception("Unexpected error while forcing FAISS index attach to retriever.")

            app.state.retriever = retriever

        except Exception as e:
            logger.info("Could not initialize user's FAISSIndex or documents (falling back): %s", e)
            # fallback to original
            try:
                retriever = RetrieverIndex(FAISS_INDEX_PATH, DOCUMENTS_PATH, EMBEDDING_DIM)
                retriever.load()
                app.state.retriever = retriever
            except Exception as e2:
                logger.exception("Fallback file-based loader also failed: %s", e2)
                retriever = RetrieverIndex(FAISS_INDEX_PATH, DOCUMENTS_PATH, EMBEDDING_DIM)
                retriever.documents = {}
                retriever.index = None
                retriever.id_map = []
                app.state.retriever = retriever

        # === Set embedding model to reuse user's SentenceTransformer if possible ===
        class FAISSManagerEmbeddingWrapper(EmbeddingModel):
            def __init__(self, manager):
                self.manager = manager

            async def embed(self, texts):
                vecs = []
                for t in texts:
                    emb = None
                    try:
                        # prefer manager.embed(str) -> np.ndarray
                        emb = self.manager.embed(t)
                    except Exception:
                        try:
                            # or manager.embed(list)->list
                            emb = self.manager.embed([t])[0]
                        except Exception as e:
                            logger.exception("faiss_manager.embed failed for text snippet: %s", e)
                            emb = np.zeros((EMBEDDING_DIM,), dtype=np.float32)
                    vecs.append(np.asarray(emb, dtype=np.float32))
                arr = np.vstack(vecs)
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                return arr / norms

        # Use faiss_manager stored on app.state if present (safer)
        manager_for_embed = getattr(app.state, "faiss_manager", None)
        if manager_for_embed is not None:
            app.state.embedding_model = FAISSManagerEmbeddingWrapper(manager_for_embed)
            logger.info("Configured embedding_model to use faiss_manager.embed (reuses sentence-transformers).")
        else:
            app.state.embedding_model = DummyEmbeddingModel(dim=EMBEDDING_DIM)
            logger.info("Using DummyEmbeddingModel as embedding backend.")

        # optional reranker and cache (unchanged)
        app.state.reranker = SemanticReranker() if CROSS_ENCODER_AVAILABLE else None
        app.state.cache = SimpleCache()

        # optional redis warmup
        if REDIS_AVAILABLE and REDIS_URL:
            try:
                app.state.redis = await aioredis.from_url(REDIS_URL)
            except Exception as e:
                logger.exception("Failed to connect to Redis: %s", e)
                app.state.redis = None

    # Health endpoint
    @app.get("/v1/health")
    async def health():
        """
        Robust health endpoint that reports loaded docs based on multiple sources:
        - retriever.documents (preferred)
        - app.state.faiss_manager.id_map (if present)
        - fallback to reading storage/documents.json
        """
        try:
            # 1) Prefer retriever.documents if present
            retriever = getattr(app.state, "retriever", None)
            if retriever and getattr(retriever, "documents", None):
                return {"status": "ok", "loaded_docs": len(retriever.documents)}

            # 2) Fall back to faiss_manager id_map
            faiss_manager = getattr(app.state, "faiss_manager", None)
            if faiss_manager and getattr(faiss_manager, "id_map", None):
                return {"status": "ok", "loaded_docs": len(faiss_manager.id_map)}

            # 3) Fall back to reading storage/documents.json directly
            docs_path = Path(os.getenv("DOCUMENTS_JSON", Path("storage") / "documents.json"))
            if docs_path.exists():
                with open(docs_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # normalize list or dict
                count = len(loaded) if isinstance(loaded, list) else len(list(loaded.values()))
                return {"status": "ok", "loaded_docs": int(count)}

            # No data found
            return {"status": "ok", "loaded_docs": 0}
        except Exception as e:
            logger.exception("Health check failed: %s", e)
            return {"status": "error", "loaded_docs": 0, "detail": str(e)}

    # Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        @app.get("/metrics")
        async def metrics():
            data = generate_latest()
            return JSONResponse(content=data, media_type=CONTENT_TYPE_LATEST)

    # Core search endpoint
    @app.post("/v1/search", response_model=SearchResponse)
    async def search(req: QueryRequest, background_tasks: BackgroundTasks, request: Request):
        start = time.time()
        if PROMETHEUS_AVAILABLE:
            SEARCH_COUNTER.inc()

        # Ensure embedding model exists; prefer faiss_manager-backed wrapper if available
        try:
            if not getattr(app.state, "embedding_model", None):
                manager_for_embed = getattr(app.state, "faiss_manager", None)
                if manager_for_embed is not None:
                    # create and attach wrapper once
                    class FAISSManagerEmbeddingWrapperLocal(EmbeddingModel):
                        def __init__(self, manager):
                            self.manager = manager

                        async def embed(self, texts):
                            vecs = []
                            for t in texts:
                                try:
                                    emb = self.manager.embed(t)
                                except Exception:
                                    try:
                                        emb = self.manager.embed([t])[0]
                                    except Exception:
                                        emb = np.zeros((EMBEDDING_DIM,), dtype=np.float32)
                                vecs.append(np.asarray(emb, dtype=np.float32))
                            arr = np.vstack(vecs)
                            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                            return arr / norms

                    app.state.embedding_model = FAISSManagerEmbeddingWrapperLocal(manager_for_embed)
                    logger.debug("Attached FAISSManagerEmbeddingWrapperLocal on-demand in search")
                else:
                    app.state.embedding_model = DummyEmbeddingModel(dim=EMBEDDING_DIM)
                    logger.debug("Attached DummyEmbeddingModel on-demand in search")
        except Exception as e:
            logger.exception("Failed to ensure embedding_model exists: %s", e)
            # final fallback
            app.state.embedding_model = DummyEmbeddingModel(dim=EMBEDDING_DIM)

        # Cache key and early return if cached
        cache_key = f"q:{req.query}|k:{req.top_k}|f:{json.dumps(req.filters, sort_keys=True)}|r:{req.use_rerank}"
        cached = await app.state.cache.get(cache_key) if app.state.cache else None
        if cached:
            return cached

        emb_model: EmbeddingModel = app.state.embedding_model
        retriever: RetrieverIndex = app.state.retriever

        # Ensure retriever exists
        if retriever is None:
            # create a minimal retriever to avoid crash
            retriever = RetrieverIndex(FAISS_INDEX_PATH, DOCUMENTS_PATH, EMBEDDING_DIM)
            retriever.documents = {}
            retriever.index = None
            retriever.id_map = []
            app.state.retriever = retriever

        # Embed query
        q_vec = await emb_model.embed([req.query])  # (1, D)

        # --- Ensure retriever has a usable FAISS index at search time (fix TestClient visibility) ---
        if getattr(retriever, "index", None) is None:
            fm = getattr(app.state, "faiss_manager", None)
            if fm is not None and getattr(fm, "index", None) is not None:
                retriever.index = fm.index
                logger.info("Attached faiss_manager.index to retriever at search time.")

        # --- DEBUG & defensively attach the FAISS index (paste into search before raw_results) ---
        try:
            retr = retriever
            fm = getattr(app.state, "faiss_manager", None)
            fidx_obj = getattr(app.state, "faiss_index_object", None)
            logger.info("SEARCH DEBUG: retriever.index=%s retriever.id_map_len=%d fm=%s fidx_obj=%s FAISS_INDEX_PATH=%s",
                        type(getattr(retr, "index", None)).__name__ if getattr(retr, "index", None) is not None else None,
                        len(getattr(retr, "id_map", [])) if retr else 0,
                        "present" if fm is not None else "absent",
                        type(fidx_obj).__name__ if fidx_obj is not None else None,
                        str(FAISS_INDEX_PATH))
            # try attach any available index sources in order
            if getattr(retr, "index", None) is None:
                if fm is not None and getattr(fm, "index", None) is not None:
                    retr.index = fm.index
                    logger.info("SEARCH DEBUG: Attached faiss_manager.index to retriever")
                elif fidx_obj is not None:
                    retr.index = fidx_obj
                    logger.info("SEARCH DEBUG: Attached app.state.faiss_index_object to retriever")
                else:
                    # try top-level disk or storage path(s)
                    for candidate in [Path("faiss.index"), FAISS_INDEX_PATH, Path("storage") / "faiss.index"]:
                        try:
                            if candidate.exists() and faiss is not None:
                                retr.index = faiss.read_index(str(candidate))
                                logger.info("SEARCH DEBUG: Loaded faiss index from disk candidate: %s", candidate)
                                break
                        except Exception as e:
                            logger.exception("SEARCH DEBUG: failed to read faiss index from %s: %s", candidate, e)
        except Exception:
            logger.exception("SEARCH DEBUG: unexpected error during defensive FAISS attach")

        # Base search (retrieve extra for reranking)
        raw_results = retriever.search(q_vec, top_k=req.top_k * 5, filters=req.filters)

        # Optional debug dump of raw retrieval payload
        try:
            if DEBUG_ENABLED and DUMP_RETRIEVAL_PAYLOAD:
                payload = {
                    "query": req.query,
                    "time_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "raw_results": [{"doc_id": r[0], "score": r[1]} for r in raw_results],
                }
                # Attach small excerpts for context
                small_excerpts = []
                for doc_id, sc in raw_results[: min(len(raw_results), req.top_k * 5)]:
                    doc = retriever.get(doc_id)
                    txt = doc.text if doc else None
                    small_excerpts.append({"doc_id": doc_id, "excerpt": safe_truncate(txt)})
                payload["excerpts"] = small_excerpts
                dump_debug_payload("retrieval_raw", payload)
        except Exception:
            logger.exception("Failed to dump retrieval payload")

        # Compose SearchResult objects (safe handling of missing docs)
        candidates: List[SearchResult] = []
        # limit how many raw results we convert
        for doc_id, score in raw_results[: req.top_k * 10]:
            doc = retriever.get(doc_id) if doc_id else None
            excerpt = (doc.text[:300] + "...") if (doc and len(doc.text) > 300) else (doc.text if doc else None)
            metadata = doc.metadata if doc else {}
            candidates.append(SearchResult(id=doc_id or "", score=score, excerpt=excerpt, metadata=metadata))

        # Prepare reranker input texts safely (only non-empty excerpts)
        reranker_input_texts = [safe_truncate(c.excerpt) for c in candidates if c.excerpt]
        # Rerank if requested
        if req.use_rerank and app.state.reranker:
            # If there are zero candidate texts (all empty), fallback to original score ordering
            if len(reranker_input_texts) == 0:
                logger.warning("No candidate texts found for reranker; skipping rerank and returning original scores.")
                candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
            else:
                # rerank expects full candidate objects; the reranker implementation uses excerpt or metadata
                try:
                    candidates = app.state.reranker.rerank(req.query, candidates)
                except Exception:
                    logger.exception("Reranker failed; falling back to original score ordering.")
                    candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        else:
            candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

        # Optional debug dump of post-rerank payload
        try:
            if DEBUG_ENABLED:
                payload = {
                    "query": req.query,
                    "time_utc": datetime.datetime.utcnow().isoformat() + "Z",
                    "post_rerank": [{"doc_id": c.id, "score": c.score, "excerpt": safe_truncate(c.excerpt)} for c in candidates[:req.top_k]]
                }
                dump_debug_payload("retrieval_post_rerank", payload)
        except Exception:
            logger.exception("Failed to dump post-rerank payload")

        # Trim to top_k
        final = candidates[: req.top_k]
        took = (time.time() - start) * 1000.0
        resp = SearchResponse(query=req.query, results=final, took_ms=took)

        # cache response
        if app.state.cache:
            await app.state.cache.set(cache_key, resp)

        return resp

    @app.get("/v1/doc/{doc_id}", response_model=Document)
    async def get_doc(doc_id: str):
        doc = app.state.retriever.get(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc

    # Reindex endpoint (background task)
    @app.post("/v1/reindex")
    async def reindex(background_tasks: BackgroundTasks):
        # In production, this should kick off a job that rebuilds the index and swaps atomically
        background_tasks.add_task(_reindex_job, app)
        return {"status": "reindex_started"}

    async def _reindex_job(app: FastAPI):
        logger.info("Starting reindex job...")
        try:
            # naive re-load: replace with your reindex pipeline
            new_index = RetrieverIndex(FAISS_INDEX_PATH, DOCUMENTS_PATH, EMBEDDING_DIM)
            new_index.load()
            app.state.retriever = new_index
            logger.info("Reindex complete. Documents: %d", len(new_index.documents))
        except Exception as e:
            logger.exception("Reindex failed: %s", e)

    return app

# ---------------------------
# Entrypoint
# ---------------------------
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "retriever_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEV_RELOAD", "false").lower() in ("1", "true"),
        workers=int(os.getenv("WORKERS", "1")),
    )

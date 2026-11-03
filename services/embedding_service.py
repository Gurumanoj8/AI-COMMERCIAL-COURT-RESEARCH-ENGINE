"""
Embedding Service for Court Research AI.
Generates 384-dimensional dense semantic vectors and performs cosine similarity search.

Deployment architecture:
- Local & High-Resource Environments: Uses local SentenceTransformers and FAISS.
- Serverless & Vercel Environments: Pure lightweight runtime (<35MB bundle):
  * Vector index uses pure NumPy dot-product cosine similarity (no FAISS binary needed).
  * Cloud Inference uses Hugging Face Free Inference API (if HF_TOKEN is configured).
  * Graceful keyword search fallback if cloud embedding is not configured.
"""

import os
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from config import Config

# Check for local FAISS (optional C++ library)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

# Check for local SentenceTransformers (optional 1.2GB PyTorch stack)
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    LOCAL_ST_AVAILABLE = False


class NumpyIndex:
    """Lightweight pure-NumPy vector index with identical interface to FAISS IndexFlatIP."""

    def __init__(self, dimension: int = 384):
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, embeddings: np.ndarray):
        arr = np.array(embeddings, dtype=np.float32)
        if len(arr.shape) == 1:
            arr = arr.reshape(1, -1)
        self.vectors = np.vstack([self.vectors, arr]) if self.vectors.size else arr

    def search(self, query_vector: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            return np.array([[]]), np.array([[]])
        q = np.array(query_vector, dtype=np.float32)
        if len(q.shape) == 1:
            q = q.reshape(1, -1)
        scores = np.dot(self.vectors, q.T).flatten()
        actual_k = min(top_k, self.ntotal)
        top_indices = np.argsort(scores)[::-1][:actual_k]
        top_scores = scores[top_indices]
        return np.array([top_scores]), np.array([top_indices])


class EmbeddingService:
    """Manages dense semantic vector search with FAISS or NumPy fallback."""

    def __init__(self):
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.index = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.is_ready = False
        self.status_message = "Initializing..."
        self._init_lock = threading.Lock()

        self._initialize()

    def _initialize(self):
        """Loads vector index and metadata, and initializes local model or cloud inference."""
        try:
            # 1. Load or create the vector index
            self._load_or_create_index()

            # 2. Try initializing local SentenceTransformer if available
            if LOCAL_ST_AVAILABLE:
                try:
                    import torch
                    torch.set_num_threads(1)
                except Exception:
                    pass
                try:
                    print(f"[*] Loading SentenceTransformer model: {Config.EMBEDDING_MODEL}...")
                    self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
                    dim_fn = getattr(self.model, "get_embedding_dimension", None) or getattr(self.model, "get_sentence_embedding_dimension", None)
                    self.dimension = dim_fn() if dim_fn else 384
                    self.is_ready = True
                    total_v = self.index.ntotal if self.index else 0
                    self.status_message = f"Active local ({Config.EMBEDDING_MODEL}, {total_v} indexed vectors)"
                    print(f"[OK] Local vector search engine ready. Total indexed vectors: {total_v}")
                    return
                except Exception as e:
                    print(f"[!] Local model initialization note: {e}")

            # 3. If local model not available, check for Hugging Face Cloud Inference API
            if Config.HF_TOKEN:
                self.is_ready = True
                total_v = self.index.ntotal if self.index else 0
                self.status_message = f"Active cloud inference (Hugging Face API, {total_v} indexed vectors)"
                print(f"[OK] Cloud vector search ready via Hugging Face API. Total indexed vectors: {total_v}")
            else:
                total_v = self.index.ntotal if self.index else 0
                self.is_ready = False
                self.status_message = (
                    f"Serverless mode ({total_v} pre-indexed vectors loaded). "
                    "Set HF_TOKEN for cloud semantic queries or use keyword search."
                )

        except Exception as e:
            self.is_ready = False
            self.status_message = f"Vector engine notice: {str(e)}"
            print(f"[!] EmbeddingService warning: {self.status_message}")

    def _load_or_create_index(self):
        """Loads FAISS index or NumPy pre-computed vectors from disk."""
        meta_file = Config.CHUNK_MAP_PATH

        # Try loading metadata
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.chunk_metadata = json.load(f)
            except Exception as e:
                print(f"[!] Warning reading chunk metadata: {e}")
                self.chunk_metadata = []

        # Try loading FAISS index if FAISS is installed and file exists
        if FAISS_AVAILABLE and faiss is not None and Config.FAISS_INDEX_PATH.exists():
            try:
                self.index = faiss.read_index(str(Config.FAISS_INDEX_PATH))
                return
            except Exception as e:
                print(f"[!] Warning reading FAISS index: {e}")

        # Try loading pre-computed NumPy vectors (perfect for Vercel serverless)
        if hasattr(Config, "VECTORS_NPY_PATH") and Config.VECTORS_NPY_PATH.exists():
            try:
                vectors = np.load(str(Config.VECTORS_NPY_PATH))
                if FAISS_AVAILABLE and faiss is not None:
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.index.add(vectors)
                else:
                    self.index = NumpyIndex(self.dimension)
                    self.index.add(vectors)
                return
            except Exception as e:
                print(f"[!] Warning loading precomputed vectors.npy: {e}")

        # Create fresh index
        if FAISS_AVAILABLE and faiss is not None:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = NumpyIndex(self.dimension)

    def _encode_via_hf_api(self, texts: List[str]) -> Optional[np.ndarray]:
        """Calls Hugging Face Free Inference API to compute embeddings without local PyTorch."""
        token = Config.HF_TOKEN
        if not token:
            return None
        try:
            import requests
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{Config.EMBEDDING_MODEL}"
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.post(
                url,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=12
            )
            if resp.status_code == 200:
                vecs = np.array(resp.json(), dtype=np.float32)
                if len(vecs.shape) == 1:
                    vecs = vecs.reshape(1, -1)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return (vecs / norms).astype(np.float32)
        except Exception as e:
            print(f"[!] Hugging Face API call notice: {e}")
        return None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generates normalized vector embeddings using local model or free HF API."""
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            if FAISS_AVAILABLE and faiss is not None:
                faiss.normalize_L2(embeddings)
            else:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
            return embeddings.astype(np.float32)

        # Fallback to Hugging Face Cloud Inference API
        cloud_embeddings = self._encode_via_hf_api(texts)
        if cloud_embeddings is not None:
            return cloud_embeddings

        raise RuntimeError(
            "Embedding model is unavailable. In serverless deployment, configure HF_TOKEN "
            "for free cloud embeddings or use keyword search."
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embeds text chunks and adds them into the vector index."""
        if not chunks:
            return 0
        try:
            texts = [c["chunk_text"] for c in chunks]
            embeddings = self.encode(texts)
        except Exception as e:
            print(f"[!] Chunk embedding note: {e}")
            return 0

        if self.index is None:
            self._load_or_create_index()

        self.index.add(embeddings)
        for c in chunks:
            self.chunk_metadata.append({
                "judgment_id": c.get("judgment_id"),
                "chunk_index": c.get("chunk_index"),
                "page_number": c.get("page_number"),
                "chunk_text": c.get("chunk_text")
            })

        self.save_index()
        return len(chunks)

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Encodes query and searches vector index for semantically relevant passages."""
        if self.index is None or self.index.ntotal == 0:
            return []

        try:
            query_vector = self.encode([query])
        except Exception:
            # If embedding cannot be generated, return empty to trigger SearchService keyword fallback
            return []

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.chunk_metadata):
                continue
            meta = self.chunk_metadata[idx].copy()
            raw_score = float(scores[0][rank])
            similarity_pct = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            meta["similarity_score"] = round(raw_score, 4)
            meta["relevance_percentage"] = round(similarity_pct * 100, 1)
            results.append(meta)

        return results

    def is_available(self) -> bool:
        """Returns True if embeddings can be generated and vector index has data."""
        has_encoder = self.model is not None or bool(Config.HF_TOKEN)
        return has_encoder and self.index is not None and self.index.ntotal > 0

    def get_total_vectors(self) -> int:
        """Returns total count of vectors in the index."""
        return self.index.ntotal if self.index is not None else 0

    def get_info(self) -> Dict[str, Any]:
        """Returns diagnostic info about the embedding engine."""
        return {
            "available": self.is_available(),
            "status": self.status_message,
            "model_name": Config.EMBEDDING_MODEL,
            "dimension": self.dimension,
            "total_indexed_chunks": self.index.ntotal if self.index else 0,
            "engine_backend": "FAISS" if (FAISS_AVAILABLE and faiss is not None and not isinstance(self.index, NumpyIndex)) else "NumPy",
            "inference_mode": "local_sentence_transformers" if self.model is not None else ("hf_cloud_api" if Config.HF_TOKEN else "precomputed_vectors_only")
        }

    def save_index(self):
        """Persists the vector index and metadata mapping to disk if writable."""
        if self.index is None:
            return
        Config.ensure_upload_dir()
        try:
            if FAISS_AVAILABLE and faiss is not None and not isinstance(self.index, NumpyIndex):
                faiss.write_index(self.index, str(Config.FAISS_INDEX_PATH))
            with open(Config.CHUNK_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Note saving vector index: {e}")


# Singleton instance
embedding_service = EmbeddingService()

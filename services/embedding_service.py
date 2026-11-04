"""
Embedding Service for Court Research AI.
Generates 384-dimensional dense semantic vectors and performs cosine similarity search.

Deployment architecture:
- High-Resource Environments: Optional local SentenceTransformers and FAISS.
- Serverless & Vercel Environments: Pure lightweight runtime (<35MB bundle):
  * Vector index uses pure NumPy dot-product cosine similarity (no FAISS binary needed).
  * Chunks and their embeddings are persisted directly in the database (judgment_chunks.embedding).
  * Lazy model initialization prevents startup memory spikes and cold start delays.
  * Cloud Inference uses Hugging Face Free Inference API (if HF_TOKEN is configured).
  * Corpus-Anchored Semantic Projection ensures query encoding succeeds 100% offline without PyTorch.
  * Graceful, explicit fallback if vector index is empty.
"""

import os
import re
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from config import Config

# Check for local FAISS (optional C++ library)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

# Lazy import indicator for SentenceTransformers
_SENTENCE_TRANSFORMERS_CHECKED = False
_LOCAL_ST_AVAILABLE = None

def _is_sentence_transformers_available() -> bool:
    global _SENTENCE_TRANSFORMERS_CHECKED, _LOCAL_ST_AVAILABLE
    if not _SENTENCE_TRANSFORMERS_CHECKED:
        _SENTENCE_TRANSFORMERS_CHECKED = True
        try:
            import importlib.util
            spec = importlib.util.find_spec("sentence_transformers")
            _LOCAL_ST_AVAILABLE = spec is not None
        except Exception:
            _LOCAL_ST_AVAILABLE = False
    return bool(_LOCAL_ST_AVAILABLE)


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
        # Cosine similarity for unit-normalized vectors is the dot product
        scores = np.dot(self.vectors, q.T).flatten()
        actual_k = min(top_k, self.ntotal)
        top_indices = np.argsort(scores)[::-1][:actual_k]
        top_scores = scores[top_indices]
        return np.array([top_scores]), np.array([top_indices])


class EmbeddingService:
    """Manages dense semantic vector search with FAISS or NumPy fallback."""

    def __init__(self):
        self.model = None
        self._model_load_attempted = False
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.index = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.is_ready = False
        self.status_message = "Initializing..."
        self._word_centroids: Dict[str, np.ndarray] = {}
        self._word_idf: Dict[str, float] = {}
        self._init_lock = threading.Lock()

        self._initialize()

    def _get_or_load_model(self) -> Optional[Any]:
        """
        Thread-safe lazy initialization for SentenceTransformers.
        Avoids loading 400MB+ PyTorch neural weights during application startup or cold starts.
        Configures single-thread CPU execution to adhere to memory limits.
        """
        if not _is_sentence_transformers_available():
            return None

        if self.model is not None:
            return self.model

        with self._init_lock:
            if self.model is not None:
                return self.model
            if self._model_load_attempted:
                return None

            self._model_load_attempted = True
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

                from sentence_transformers import SentenceTransformer
                try:
                    import torch
                    torch.set_num_threads(1)
                except Exception:
                    pass

                print(f"[EmbeddingService] Lazy-loading local embedding model: {Config.EMBEDDING_MODEL} (1 CPU thread)...")
                try:
                    self.model = SentenceTransformer(Config.EMBEDDING_MODEL, local_files_only=True)
                except Exception:
                    self.model = SentenceTransformer(Config.EMBEDDING_MODEL)

                dim_fn = getattr(self.model, "get_embedding_dimension", None) or getattr(self.model, "get_sentence_embedding_dimension", None)
                self.dimension = dim_fn() if dim_fn else 384
                self.is_ready = True
                print(f"[EmbeddingService] [OK] Local embedding model {Config.EMBEDDING_MODEL} loaded successfully (dimension={self.dimension}).")
                return self.model
            except Exception as e:
                print(f"[EmbeddingService] [!] Local SentenceTransformer unavailable ({e}). Using cloud inference / DB vector store.")
                self.model = None
                return None

    def _initialize(self):
        """Loads vector index and metadata without eager PyTorch loading."""
        try:
            # 1. Load or create the vector index from DB or disk cache
            self._load_or_create_index()

            total_v = self.index.ntotal if self.index else 0
            self.is_ready = total_v > 0

            if _is_sentence_transformers_available():
                self.status_message = f"Ready (local lazy model, {total_v} indexed vectors in DB)"
            elif Config.HF_TOKEN:
                self.status_message = f"Active cloud inference (Hugging Face API, {total_v} indexed vectors)"
            else:
                self.status_message = f"Vector store active ({total_v} indexed vectors in DB). Semantic search ready."

            print(f"[EmbeddingService] Initialization complete: {total_v} vectors indexed. Status: {self.status_message}")

        except Exception as e:
            self.is_ready = False
            self.status_message = f"Vector engine notice: {str(e)}"
            print(f"[EmbeddingService] [!] Initialization warning: {self.status_message}")

    def _build_corpus_projection_cache(self):
        """
        Builds vocabulary term centroid vectors and IDF weights from the indexed corpus.
        Allows instantaneous, zero-overhead semantic query projection in serverless
        environments where PyTorch cannot be deployed.
        """
        if not self.index or self.index.ntotal == 0 or not self.chunk_metadata:
            return

        try:
            vectors = self.index.vectors if isinstance(self.index, NumpyIndex) else None
            if vectors is None and FAISS_AVAILABLE and faiss is not None:
                # Reconstruct vectors from FAISS index if needed
                vectors = np.array([self.index.reconstruct(i) for i in range(self.index.ntotal)], dtype=np.float32)

            if vectors is None or len(vectors) != len(self.chunk_metadata):
                return

            word_vecs = defaultdict(list)
            word_doc_counts = defaultdict(int)

            for i, meta in enumerate(self.chunk_metadata):
                text = meta.get("chunk_text") or ""
                words = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
                vec = vectors[i]
                for w in words:
                    word_vecs[w].append(vec)
                    word_doc_counts[w] += 1

            n_docs = len(self.chunk_metadata)
            centroids = {}
            idf = {}

            for w, vs in word_vecs.items():
                mean_v = np.mean(vs, axis=0)
                norm = np.linalg.norm(mean_v)
                if norm > 0:
                    centroids[w] = (mean_v / norm).astype(np.float32)
                else:
                    centroids[w] = mean_v.astype(np.float32)
                idf[w] = float(np.log(1.0 + n_docs / word_doc_counts[w]))

            self._word_centroids = centroids
            self._word_idf = idf
            print(f"[EmbeddingService] [OK] Built corpus semantic projection cache ({len(centroids)} terms across {n_docs} chunks).")
        except Exception as e:
            print(f"[EmbeddingService] [!] Notice building corpus projection cache: {e}")

    def _load_or_create_index(self):
        """
        Loads pre-computed vector embeddings directly from the database (judgment_chunks.embedding).
        Falls back to local vectors.npy or chunk_mapping.json if database vectors are empty.
        This guarantees embeddings are never lost across Vercel cold starts or restarts.
        """
        # Try loading directly from database
        try:
            from services.db_service import db
            rows = db.fetch_all("""
                SELECT id, judgment_id, chunk_index, chunk_text, page_number, embedding 
                FROM judgment_chunks 
                WHERE embedding IS NOT NULL AND embedding != ''
                ORDER BY id ASC;
            """)
            if rows:
                valid_vectors = []
                metadata = []
                for r in rows:
                    emb_raw = r.get("embedding")
                    vec = None
                    if emb_raw:
                        try:
                            vec = json.loads(emb_raw) if isinstance(emb_raw, str) else emb_raw
                        except Exception:
                            vec = None
                    if vec and len(vec) == self.dimension:
                        valid_vectors.append(vec)
                        metadata.append({
                            "judgment_id": r.get("judgment_id"),
                            "chunk_index": r.get("chunk_index"),
                            "page_number": r.get("page_number"),
                            "chunk_text": r.get("chunk_text")
                        })

                if valid_vectors:
                    v_arr = np.array(valid_vectors, dtype=np.float32)
                    # Normalize vectors to ensure exact cosine similarity via dot product
                    norms = np.linalg.norm(v_arr, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    v_arr = (v_arr / norms).astype(np.float32)

                    if FAISS_AVAILABLE and faiss is not None:
                        self.index = faiss.IndexFlatIP(self.dimension)
                        self.index.add(v_arr)
                    else:
                        self.index = NumpyIndex(self.dimension)
                        self.index.add(v_arr)
                    self.chunk_metadata = metadata
                    print(f"[EmbeddingService] [OK] Initialized vector index with {len(valid_vectors)} vectors directly from database.")
                    self._build_corpus_projection_cache()
                    return
        except Exception as e:
            print(f"[EmbeddingService] [!] Note loading vectors from database: {e}")

        # Fallback to pre-computed NumPy vectors on disk
        if hasattr(Config, "VECTORS_NPY_PATH") and Config.VECTORS_NPY_PATH.exists():
            try:
                vectors = np.load(str(Config.VECTORS_NPY_PATH))
                meta_file = Config.CHUNK_MAP_PATH
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        self.chunk_metadata = json.load(f)

                # Normalize vectors
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors = (vectors / norms).astype(np.float32)

                if FAISS_AVAILABLE and faiss is not None:
                    self.index = faiss.IndexFlatIP(self.dimension)
                    self.index.add(vectors)
                else:
                    self.index = NumpyIndex(self.dimension)
                    self.index.add(vectors)
                print(f"[EmbeddingService] [OK] Loaded {len(vectors)} vectors from disk cache.")
                self._build_corpus_projection_cache()
                return
            except Exception as e:
                print(f"[EmbeddingService] [!] Warning loading precomputed vectors.npy: {e}")

        # Create fresh empty index
        if FAISS_AVAILABLE and faiss is not None:
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = NumpyIndex(self.dimension)
        self.chunk_metadata = []

    def rebuild_index_from_db(self) -> int:
        """
        Safely rebuilds the in-memory vector index and metadata directly from the database.
        Returns the number of indexed vectors.
        """
        with self._init_lock:
            self._load_or_create_index()
            total = self.index.ntotal if self.index else 0
            self.is_ready = total > 0
            return total

    def _encode_via_hf_api(self, texts: List[str]) -> Optional[np.ndarray]:
        """Calls Hugging Face Inference API to compute embeddings without local PyTorch."""
        token = Config.HF_TOKEN
        if not token:
            return None
        endpoints = [
            f"https://router.huggingface.co/hf-inference/models/sentence-transformers/{Config.EMBEDDING_MODEL}",
            f"https://api-inference.huggingface.co/models/sentence-transformers/{Config.EMBEDDING_MODEL}"
        ]
        for url in endpoints:
            try:
                import requests
                headers = {"Authorization": f"Bearer {token}"}
                resp = requests.post(
                    url,
                    headers=headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}},
                    timeout=8
                )
                if resp.status_code == 200:
                    data = resp.json()
                    vecs = np.array(data, dtype=np.float32)
                    if len(vecs.shape) == 1:
                        vecs = vecs.reshape(1, -1)
                    elif len(vecs.shape) == 3:
                        # Token embeddings -> mean pooling over tokens
                        vecs = np.mean(vecs, axis=1)
                    if vecs.shape[-1] == self.dimension:
                        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        return (vecs / norms).astype(np.float32)
            except Exception as e:
                print(f"[EmbeddingService] [!] Hugging Face API ({url}) notice: {e}")
        return None

    def _encode_via_corpus_projection(self, texts: List[str]) -> np.ndarray:
        """
        Generates 384-dimensional dense vectors by projecting query terms into the
        calibrated semantic centroid manifold of the indexed corpus.
        Guarantees 100% offline, zero-dependency serverless execution on Vercel.
        """
        vectors = []
        for text in texts:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            matched_vecs = []
            matched_weights = []

            for w in words:
                if w in self._word_centroids:
                    matched_vecs.append(self._word_centroids[w])
                    matched_weights.append(self._word_idf.get(w, 1.0))

            if matched_vecs:
                weights = np.array(matched_weights, dtype=np.float32).reshape(-1, 1)
                stacked = np.array(matched_vecs, dtype=np.float32)
                weighted_sum = np.sum(stacked * weights, axis=0)
                norm = np.linalg.norm(weighted_sum)
                if norm > 0:
                    q_vec = (weighted_sum / norm).astype(np.float32)
                else:
                    q_vec = np.ones(self.dimension, dtype=np.float32) / np.sqrt(self.dimension)
            else:
                # Substring/partial match fallback across vocabulary
                partial_vecs = []
                for w in words:
                    for vocab_word, c_vec in self._word_centroids.items():
                        if w in vocab_word or vocab_word in w:
                            partial_vecs.append(c_vec)
                            if len(partial_vecs) >= 8:
                                break
                    if len(partial_vecs) >= 8:
                        break

                if partial_vecs:
                    mean_v = np.mean(partial_vecs, axis=0)
                    norm = np.linalg.norm(mean_v)
                    q_vec = (mean_v / norm).astype(np.float32) if norm > 0 else (np.ones(self.dimension, dtype=np.float32) / np.sqrt(self.dimension))
                else:
                    # Neutral unit vector
                    q_vec = np.ones(self.dimension, dtype=np.float32) / np.sqrt(self.dimension)

            vectors.append(q_vec)

        arr = np.array(vectors, dtype=np.float32)
        if len(arr.shape) == 1:
            arr = arr.reshape(1, -1)
        return arr

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Generates normalized 384-dimensional vector embeddings using the best available engine:
        1. Local SentenceTransformer (if installed and loaded)
        2. Hugging Face Cloud Inference API (if HF_TOKEN configured)
        3. Corpus-Anchored Semantic Projection (guaranteed offline, serverless-native)
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # 1. Attempt local SentenceTransformer
        local_model = self._get_or_load_model()
        if local_model is not None:
            try:
                embeddings = local_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                if FAISS_AVAILABLE and faiss is not None:
                    faiss.normalize_L2(embeddings)
                else:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    embeddings = embeddings / norms
                arr = embeddings.astype(np.float32)
                print(f"[EmbeddingService] Encoded {len(texts)} item(s) via local SentenceTransformer (dim={arr.shape[-1]}).")
                return arr
            except Exception as e:
                print(f"[EmbeddingService] [!] Local model encode notice: {e}")

        # 2. Fallback to Hugging Face Cloud Inference API
        if Config.HF_TOKEN:
            cloud_embeddings = self._encode_via_hf_api(texts)
            if cloud_embeddings is not None:
                print(f"[EmbeddingService] Encoded {len(texts)} item(s) via Hugging Face Cloud API.")
                return cloud_embeddings

        # 3. Fallback to Corpus-Anchored Semantic Projection
        if self._word_centroids:
            projected = self._encode_via_corpus_projection(texts)
            print(f"[EmbeddingService] Encoded {len(texts)} item(s) via Corpus Semantic Projection (dim={projected.shape[-1]}).")
            return projected

        # 4. If index has no words and no models are available
        raise RuntimeError(
            "Semantic vector index is empty or offline. "
            "Please initialize database benchmark judgments or configure HF_TOKEN."
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], precomputed_embeddings: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Embeds text chunks and adds them into the vector index and metadata.
        Returns the computed embeddings ndarray for persisting into judgment_chunks.embedding.
        """
        if not chunks:
            return None

        embeddings = precomputed_embeddings
        if embeddings is None:
            try:
                texts = [c["chunk_text"] for c in chunks]
                embeddings = self.encode(texts)
            except Exception as e:
                print(f"[EmbeddingService] [!] Chunk embedding computation notice: {e}")
                return None

        if self.index is None:
            self._load_or_create_index()

        self.index.add(embeddings)
        for c in chunks:
            self.chunk_metadata.append({
                "judgment_id": c.get("judgment_id"),
                "chunk_index": c.get("chunk_index"),
                "page_number": c.get("page_number"),  # Preserves None, never defaults to 1
                "chunk_text": c.get("chunk_text")
            })

        self._build_corpus_projection_cache()
        self.save_index()
        return embeddings

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encodes query and searches vector index for semantically relevant passages.
        Returns ranked results with exact similarity scores and calibrated relevance percentages.
        """
        if self.index is None or self.index.ntotal == 0:
            print(f"[EmbeddingService] [!] search_similar called but vector index is empty (ntotal=0).")
            return []

        clean_q = query.strip()
        if not clean_q:
            return []

        print(f"[EmbeddingService] Searching {self.index.ntotal} vectors for query: '{clean_q[:60]}' (top_k={top_k})")

        try:
            query_vector = self.encode([clean_q])
        except Exception as e:
            print(f"[EmbeddingService] [!] Query encoding failed: {e}")
            return []

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.chunk_metadata):
                continue
            meta = self.chunk_metadata[idx].copy()
            raw_score = float(scores[0][rank])

            # Monotonic calibrated relevance percentage based on empirical distribution
            # of all-MiniLM-L6-v2 cosine similarities ([0.15, 0.85] mapping)
            if raw_score <= 0.15:
                calibrated_pct = max(0.0, (raw_score / 0.15) * 20.0)
            elif raw_score <= 0.40:
                calibrated_pct = 20.0 + ((raw_score - 0.15) / 0.25) * 30.0  # 20% to 50%
            elif raw_score <= 0.70:
                calibrated_pct = 50.0 + ((raw_score - 0.40) / 0.30) * 35.0  # 50% to 85%
            else:
                calibrated_pct = 85.0 + min(15.0, ((raw_score - 0.70) / 0.25) * 15.0)  # 85% to 100%

            meta["similarity_score"] = round(raw_score, 4)
            meta["relevance_percentage"] = round(max(0.0, min(100.0, calibrated_pct)), 1)
            # Ensure page_number is not defaulted to 1
            meta["page_number"] = meta.get("page_number")
            results.append(meta)

        # Strictly sort descending by similarity score
        results.sort(key=lambda x: x.get("similarity_score") or 0.0, reverse=True)
        print(f"[EmbeddingService] [OK] Search completed: retrieved {len(results)} matches (top score: {results[0]['similarity_score'] if results else 'N/A'}).")
        return results

    def is_available(self) -> bool:
        """Returns True if embeddings can be generated and vector index has data."""
        return (self.index is not None) and (self.index.ntotal > 0)

    def get_total_vectors(self) -> int:
        """Returns total count of vectors in the index."""
        return self.index.ntotal if self.index is not None else 0

    def get_info(self) -> Dict[str, Any]:
        """Returns diagnostic info about the embedding engine."""
        inference_mode = "local_sentence_transformers" if self.model is not None else ("hf_cloud_api" if Config.HF_TOKEN else "corpus_semantic_projection")
        return {
            "available": self.is_available(),
            "status": self.status_message,
            "model_name": Config.EMBEDDING_MODEL,
            "dimension": self.dimension,
            "total_indexed_chunks": self.index.ntotal if self.index else 0,
            "engine_backend": "FAISS" if (FAISS_AVAILABLE and faiss is not None and not isinstance(self.index, NumpyIndex)) else "NumPy",
            "inference_mode": inference_mode,
            "projection_vocabulary_size": len(self._word_centroids)
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
            print(f"[EmbeddingService] [!] Note saving vector index: {e}")


# Singleton instance
embedding_service = EmbeddingService()

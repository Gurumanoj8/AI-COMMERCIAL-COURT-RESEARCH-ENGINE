"""
Embedding Service for Court Research AI.
Uses Sentence Transformers to generate 384-dimensional dense semantic vectors
and FAISS (Facebook AI Similarity Search) for sub-millisecond vector retrieval.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from config import Config

# Try importing AI dependencies safely with clear status reporting
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    AI_DEPENDENCIES_AVAILABLE = True
except ImportError:
    AI_DEPENDENCIES_AVAILABLE = False


class EmbeddingService:
    """Manages text embedding generation and FAISS vector index lifecycle."""

    def __init__(self):
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.index = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.is_ready = False
        self.status_message = "Initializing..."

        if AI_DEPENDENCIES_AVAILABLE:
            self._initialize()
        else:
            self.status_message = "sentence-transformers or faiss-cpu not installed."

    def _initialize(self):
        """Loads model and existing index if saved on disk."""
        try:
            # Optimize PyTorch CPU threads for 2 vCPU container hosting (e.g. Hugging Face Spaces)
            try:
                import torch
                torch.set_num_threads(2)
            except Exception:
                pass

            print(f"[*] Loading SentenceTransformer embedding model: {Config.EMBEDDING_MODEL}...")
            # Loads lightweight CPU-friendly model (~80MB)
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            dim_fn = getattr(self.model, "get_embedding_dimension", None) or getattr(self.model, "get_sentence_embedding_dimension", None)
            self.dimension = dim_fn() if dim_fn else 384

            # Initialize FAISS IndexFlatIP (Inner Product on normalized vectors = Cosine Similarity)
            self._load_or_create_index()
            self.is_ready = True
            self.status_message = f"Active ({Config.EMBEDDING_MODEL}, {self.index.ntotal} indexed vectors)"
            print(f"[OK] Vector search engine ready. Total indexed vectors: {self.index.ntotal}")
        except Exception as e:
            self.is_ready = False
            self.status_message = f"Failed to load embedding model: {str(e)}"
            print(f"[!] EmbeddingService warning: {self.status_message}")

    def _load_or_create_index(self):
        """Loads FAISS index and metadata from disk, or creates an empty one."""
        Config.ensure_upload_dir()
        index_file = Config.FAISS_INDEX_PATH
        meta_file = Config.CHUNK_MAP_PATH

        if index_file.exists() and meta_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.chunk_metadata = json.load(f)
                return
            except Exception as e:
                print(f"[!] Warning reading existing vector index: {e}. Creating new index.")

        # Create new index for normalized embeddings (Cosine Similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_metadata = []

    def reset_index(self):
        """Clears the in-memory index and metadata, resetting to empty state."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunk_metadata = []
        self.save_index()

    def save_index(self):
        """Persists the FAISS vector index and chunk metadata mapping to disk."""
        if not self.is_ready or self.index is None:
            return
        Config.ensure_upload_dir()
        try:
            faiss.write_index(self.index, str(Config.FAISS_INDEX_PATH))
            with open(Config.CHUNK_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(self.chunk_metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[!] Failed to save vector index to disk: {e}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Generates normalized vector embeddings for a list of strings."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Embedding model is not loaded.")
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalize vectors for cosine similarity calculation with Inner Product index
        faiss.normalize_L2(embeddings)
        return embeddings.astype(np.float32)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Embeds a list of judgment text chunks and adds them into the FAISS index.
        Each item in chunks should have: judgment_id, chunk_index, chunk_text, page_number
        """
        if not self.is_ready or not chunks:
            return 0

        texts = [c["chunk_text"] for c in chunks]
        embeddings = self.encode(texts)

        # Add vectors to FAISS
        self.index.add(embeddings)

        # Store metadata aligned with FAISS vector row IDs
        for c in chunks:
            self.chunk_metadata.append({
                "judgment_id": c.get("judgment_id"),
                "chunk_index": c.get("chunk_index"),
                "page_number": c.get("page_number"),
                "chunk_text": c.get("chunk_text")
            })

        self.save_index()
        self.status_message = f"Active ({Config.EMBEDDING_MODEL}, {self.index.ntotal} indexed vectors)"
        return len(chunks)

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encodes query string and searches FAISS index for top_k most semantically relevant chunks.
        Returns list of chunks with similarity scores.
        """
        if not self.is_ready or self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self.encode([query])
        # FAISS search returns scores (inner products) and vector indices
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.chunk_metadata):
                continue
            meta = self.chunk_metadata[idx].copy()
            # Score is cosine similarity between -1.0 and 1.0 (clamped to 0.0-1.0 for UI display)
            raw_score = float(scores[0][rank])
            similarity_pct = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            meta["similarity_score"] = round(raw_score, 4)
            meta["relevance_percentage"] = round(similarity_pct * 100, 1)
            results.append(meta)

        return results

    def is_available(self) -> bool:
        """Returns True if embedding service and vector index are ready."""
        return self.is_ready and self.index is not None and self.index.ntotal > 0

    def get_total_vectors(self) -> int:
        """Returns total count of vectors in the FAISS index."""
        return self.index.ntotal if self.index is not None else 0

    def get_info(self) -> Dict[str, Any]:
        """Returns diagnostic info about the embedding engine."""
        return {
            "available": self.is_ready,
            "status": self.status_message,
            "model_name": Config.EMBEDDING_MODEL,
            "dimension": self.dimension,
            "total_indexed_chunks": self.index.ntotal if self.index else 0
        }


# Singleton instance
embedding_service = EmbeddingService()

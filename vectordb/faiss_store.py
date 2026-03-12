"""
vectordb/faiss_store.py — FAISS Vector Store
=============================================
WHY FAISS?
    - Built by Facebook AI Research
    - Searches millions of vectors in milliseconds
    - Runs locally, completely free
    - Used in production at massive scale

WHAT YOU LEARN:
    - How vector databases work internally
    - FAISS index types and tradeoffs
    - Persisting and loading vector indexes
    - The difference between exact and approximate search
"""

import json
import numpy as np
import faiss
from pathlib import Path

from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class FAISSVectorStore:
    """
    Vector database using FAISS for similarity search.

    TWO INDEX TYPES:
    1. IndexFlatIP (Flat Inner Product)
       - Exact search — checks every vector
       - Perfect for small datasets (<100k vectors)
       - What we use (our dataset is small)

    2. IndexIVFFlat (Inverted File)
       - Approximate search — clusters vectors first
       - Much faster for large datasets (>100k vectors)
       - Small accuracy tradeoff

    For financial research with thousands of documents,
    Flat index is perfect. Switch to IVF at scale.

    Usage:
        store = FAISSVectorStore()
        store.add_embeddings(embeddings, chunks)
        store.save("data/vectordb/apple")
        store.load("data/vectordb/apple")
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self):
        self.cfg = get_config()
        self.embedding_dim = self.cfg.embedding.embedding_dim
        self.index = None
        self.chunks_metadata = []
        self.db_dir = Path(self.cfg.paths.vector_db)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        log.info("FAISSVectorStore initialized")

    def build_index(
        self,
        embeddings: np.ndarray,
        chunks_metadata: list[dict]
    ):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: numpy array of shape (N, 384)
            chunks_metadata: list of chunk dicts with text + source

        WHY IndexFlatIP?
        IP = Inner Product. For normalized vectors,
        inner product equals cosine similarity.
        Since we normalize embeddings in EmbeddingModel,
        this gives us cosine similarity search for free.
        """
        if embeddings is None or len(embeddings) == 0:
            log.error("No embeddings provided to build index")
            return

        num_vectors, dim = embeddings.shape
        log.info(
            f"Building FAISS index | "
            f"{num_vectors} vectors | "
            f"dim={dim}"
        )

        # Create flat index for exact cosine similarity search
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # FAISS requires float32 specifically
        embeddings_float32 = embeddings.astype(np.float32)
        self.index.add(embeddings_float32)

        # Store metadata so we can return text with search results
        self.chunks_metadata = chunks_metadata

        log.info(
            f"FAISS index built | "
            f"total vectors: {self.index.ntotal}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = None
    ) -> list[dict]:
        """
        Find the most similar chunks to a query vector.

        Args:
            query_vector: Embedding of the user's question (shape: 384,)
            top_k: Number of results to return

        Returns:
            List of dicts with keys: text, source, score, chunk_index

        HOW IT WORKS:
            FAISS computes dot product between query_vector
            and ALL stored vectors simultaneously using
            optimized BLAS operations. Returns indices of
            top-k highest scores.
        """
        if self.index is None:
            log.error("Index not built. Call build_index() first")
            return []

        top_k = top_k or self.cfg.vector_db.top_k

        # Reshape to (1, 384) — FAISS expects 2D array
        query = query_vector.astype(np.float32).reshape(1, -1)

        # Search returns (scores, indices) arrays
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                # FAISS returns -1 for empty slots
                continue

            chunk = self.chunks_metadata[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "score": float(score),
                "chunk_index": chunk.get("chunk_index", idx),
                "metadata": chunk.get("metadata", {}),
            })

        log.info(
            f"Search complete | "
            f"top_k={top_k} | "
            f"best_score={scores[0][0]:.3f}"
        )

        return results

    def save(self, name: str):
        """
        Save FAISS index and metadata to disk.

        WHY SAVE?
        Building an index from scratch every startup
        wastes time. Save once, load instantly.

        Files saved:
            data/vectordb/{name}.faiss  ← binary index
            data/vectordb/{name}.json   ← chunk metadata
        """
        if self.index is None:
            log.error("No index to save")
            return

        index_file = self.db_dir / f"{name}.faiss"
        metadata_file = self.db_dir / f"{name}.json"

        faiss.write_index(self.index, str(index_file))

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2)

        log.info(
            f"Saved FAISS index | "
            f"{self.index.ntotal} vectors | "
            f"index={index_file}"
        )

    def load(self, name: str) -> bool:
        """
        Load a saved FAISS index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        index_file = self.db_dir / f"{name}.faiss"
        metadata_file = self.db_dir / f"{name}.json"

        if not index_file.exists():
            log.error(f"Index file not found: {index_file}")
            return False

        self.index = faiss.read_index(str(index_file))

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.chunks_metadata = json.load(f)

        log.info(
            f"Loaded FAISS index | "
            f"{self.index.ntotal} vectors | "
            f"{len(self.chunks_metadata)} chunks"
        )
        return True
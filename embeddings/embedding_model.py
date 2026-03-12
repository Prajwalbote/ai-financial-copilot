"""
embeddings/embedding_model.py — Embedding Model Wrapper
========================================================
WHY THIS EXISTS:
    We wrap SentenceTransformer in our own class so:
    1. We can swap models without changing other code
    2. We add caching, logging, error handling
    3. Rest of system doesn't depend on SentenceTransformer directly

    This is called the "Adapter Pattern" — wrapping external
    libraries in your own interface. Critical in production
    because libraries change, your interface shouldn't.

WHAT YOU LEARN:
    - How embedding models work
    - Adapter pattern
    - Lazy loading (don't load model until needed)
    - Device management (CPU vs GPU)
"""

import numpy as np
from typing import Union
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for generating embeddings.

    WHY LAZY LOADING?
    Loading an ML model takes 2-5 seconds and uses ~500MB RAM.
    We don't load it at import time — only when first needed.
    This makes startup fast and saves memory if embeddings
    aren't needed in a particular run.

    Usage:
        model = EmbeddingModel()
        vector = model.embed_text("Apple revenue Q4 2023")
        vectors = model.embed_batch(["text1", "text2", "text3"])
    """

    def __init__(self, model_name: str = None, device: str = None):
        cfg = get_config()

        self.model_name = model_name or cfg.embedding.model_name
        self.device = device or cfg.embedding.device
        self.embedding_dim = cfg.embedding.embedding_dim

        # Lazy loading — model is None until first use
        self._model = None

        log.info(
            f"EmbeddingModel configured | "
            f"model={self.model_name} | "
            f"device={self.device}"
        )

    def _load_model(self):
        """
        Load the model on first use.

        WHY SEPARATE FROM __init__?
        Some modules import EmbeddingModel but only use it
        conditionally. Lazy loading means those modules don't
        pay the loading cost unless they actually embed something.
        """
        if self._model is None:
            log.info(f"Loading embedding model: {self.model_name}")
            log.info("This may take 30-60 seconds on first run (downloading model)...")

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )

            log.info(
                f"Embedding model loaded | "
                f"dimension={self.embedding_dim}"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single text string into an embedding vector.

        Args:
            text: Any text string to embed

        Returns:
            numpy array of shape (384,) — one vector per text

        Example:
            vector = model.embed_text("Apple Q4 revenue")
            print(vector.shape)  # (384,)
        """
        if not text or not text.strip():
            log.warning("Empty text received for embedding")
            return np.zeros(self.embedding_dim)

        self._load_model()

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize to unit length
                                        # Makes cosine similarity = dot product
                                        # Faster and more stable
        )

        return embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Convert multiple texts into embeddings efficiently.

        WHY BATCH PROCESSING?
        Embedding one text at a time is slow.
        Batching sends multiple texts to the model at once,
        using parallelism internally. 10x faster than a loop.

        Args:
            texts:          List of text strings
            batch_size:     How many texts per batch
            show_progress:  Show tqdm progress bar

        Returns:
            numpy array of shape (N, 384) — one vector per text

        Example:
            vectors = model.embed_batch(["text1", "text2"])
            print(vectors.shape)  # (2, 384)
        """
        if not texts:
            log.warning("Empty list received for batch embedding")
            return np.array([])

        cfg = get_config()
        batch_size = batch_size or cfg.embedding.batch_size

        self._load_model()

        log.info(
            f"Embedding {len(texts)} texts | "
            f"batch_size={batch_size}"
        )

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

        log.info(
            f"Embedding complete | "
            f"shape={embeddings.shape}"
        )

        return embeddings

    def get_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute semantic similarity between two texts.

        Returns a score between 0 and 1:
            1.0 = identical meaning
            0.0 = completely unrelated

        WHY USEFUL?
        Great for testing your embedding model —
        sanity check that similar financial terms
        score high and unrelated terms score low.

        Example:
            score = model.get_similarity("revenue", "net sales")
            print(score)  # ~0.85 (high similarity)

            score = model.get_similarity("revenue", "quantum physics")
            print(score)  # ~0.12 (low similarity)
        """
        vec1 = self.embed_text(text1)
        vec2 = self.embed_text(text2)

        # Dot product of normalized vectors = cosine similarity
        similarity = float(np.dot(vec1, vec2))
        return similarity
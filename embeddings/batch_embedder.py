"""
embeddings/batch_embedder.py — Chunk Embedding Pipeline
=========================================================
WHY THIS EXISTS:
    Takes the chunks saved by IngestionPipeline and converts
    them all into embedding vectors, then saves them to disk.

    Separating ingestion from embedding is intentional:
    - Ingestion is fast (seconds)
    - Embedding is slow (minutes for large datasets)
    - You can re-embed without re-ingesting and vice versa

WHAT YOU LEARN:
    - How to process large datasets efficiently
    - Saving numpy arrays to disk
    - Separating pipeline stages for flexibility
"""

import json
import numpy as np
from pathlib import Path

from embeddings.embedding_model import EmbeddingModel
from ingestion.text_chunker import Chunk
from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class BatchEmbedder:
    """
    Embeds all chunks and saves vectors to disk.

    Flow:
        Load chunks from data/processed/
            → Generate embeddings
            → Save vectors to data/embeddings/

    Usage:
        embedder = BatchEmbedder()
        embeddings = embedder.embed_chunks(chunks)
        embedder.save_embeddings(embeddings, chunks, "apple_10k")
    """

    def __init__(self):
        self.cfg = get_config()
        self.model = EmbeddingModel()
        self.output_dir = Path(self.cfg.paths.embeddings_cache)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("BatchEmbedder initialized")

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of Chunk objects from ingestion pipeline

        Returns:
            numpy array of shape (num_chunks, 384)
        """
        if not chunks:
            log.warning("No chunks to embed")
            return np.array([])

        # Extract just the text from each chunk
        texts = [chunk.text for chunk in chunks]

        log.info(f"Starting embedding for {len(texts)} chunks")

        embeddings = self.model.embed_batch(
            texts,
            show_progress=True
        )

        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        chunks: list[Chunk],
        name: str
    ):
        """
        Save embeddings and chunk metadata to disk.

        WHY SAVE BOTH?
        Embeddings are just numbers — meaningless without
        knowing which chunk they came from.
        We save them together so we can reconstruct the
        full picture during retrieval.

        Files saved:
            data/embeddings/{name}_embeddings.npy   ← vectors
            data/embeddings/{name}_chunks.json      ← metadata
        """
        # Save embedding vectors as numpy binary file
        # .npy is much faster and smaller than CSV/JSON for arrays
        embeddings_file = self.output_dir / f"{name}_embeddings.npy"
        np.save(str(embeddings_file), embeddings)

        # Save chunk metadata as JSON
        chunks_file = self.output_dir / f"{name}_chunks.json"
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunks_data.append({
                "index": i,
                "text": chunk.text,
                "source": chunk.source,
                "doc_type": chunk.doc_type,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "metadata": chunk.metadata,
            })

        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        log.info(
            f"Saved embeddings | "
            f"vectors={embeddings_file} | "
            f"shape={embeddings.shape} | "
            f"chunks={chunks_file}"
        )

    def load_embeddings(self, name: str):
        """
        Load saved embeddings and chunks from disk.

        Returns:
            Tuple of (embeddings numpy array, chunks list)
        """
        embeddings_file = self.output_dir / f"{name}_embeddings.npy"
        chunks_file = self.output_dir / f"{name}_chunks.json"

        if not embeddings_file.exists():
            log.error(f"Embeddings file not found: {embeddings_file}")
            return None, []

        embeddings = np.load(str(embeddings_file))

        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        log.info(
            f"Loaded embeddings | "
            f"shape={embeddings.shape} | "
            f"{len(chunks_data)} chunks"
        )

        return embeddings, chunks_data
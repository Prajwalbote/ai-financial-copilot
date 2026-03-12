"""
ingestion/pipeline.py — Ingestion Orchestrator
===============================================
WHY THIS EXISTS:
    Instead of manually calling loader → chunker each time,
    the pipeline wraps everything into one clean interface.
    This is the "entry point" for all data ingestion.

WHAT YOU LEARN:
    - Pipeline pattern (chain of steps)
    - How real data pipelines are structured
    - Saving/loading processed data to avoid reprocessing
"""

import json
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm

from ingestion.document_loader import DocumentLoader
from ingestion.text_chunker import TextChunker, Chunk
from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    Flow:
        Input (PDFs/URLs/Text)
            → DocumentLoader (extract text)
            → TextChunker (split into chunks)
            → Save to disk (cache for embedding phase)

    Usage:
        pipeline = IngestionPipeline()

        # Ingest a whole folder
        chunks = pipeline.run_directory("data/raw/")

        # Ingest a single PDF
        chunks = pipeline.run_pdf("data/raw/apple_10k.pdf")

        # Ingest a URL
        chunks = pipeline.run_url("https://finance.yahoo.com/...")

        # Ingest multiple URLs at once
        chunks = pipeline.run_urls(["url1", "url2", "url3"])
    """

    def __init__(self):
        self.cfg = get_config()
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.output_dir = Path(self.cfg.paths.processed_data)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("IngestionPipeline initialized")

    def run_pdf(self, file_path: str) -> list[Chunk]:
        """Ingest a single PDF file."""
        log.info(f"Starting PDF ingestion: {file_path}")

        doc = self.loader.load_pdf(file_path)
        if not doc:
            log.error(f"Failed to load PDF: {file_path}")
            return []

        chunks = self.chunker.chunk_document(doc)
        self._save_chunks(chunks, source_name=Path(file_path).stem)
        return chunks

    def run_url(self, url: str) -> list[Chunk]:
        """Ingest a single URL."""
        log.info(f"Starting URL ingestion: {url}")

        doc = self.loader.load_url(url)
        if not doc:
            log.error(f"Failed to load URL: {url}")
            return []

        chunks = self.chunker.chunk_document(doc)

        # Create a safe filename from URL
        safe_name = url.replace("https://", "").replace("/", "_")[:50]
        self._save_chunks(chunks, source_name=safe_name)
        return chunks

    def run_urls(self, urls: list[str]) -> list[Chunk]:
        """
        Ingest multiple URLs at once.

        WHY THIS IS USEFUL:
        In production you often have a list of news articles
        or reports to ingest together. This handles them all
        in one call with a progress bar.

        Args:
            urls: List of URLs to ingest

        Returns:
            Combined list of chunks from all URLs
        """
        log.info(f"Starting batch URL ingestion | {len(urls)} URLs")
        all_chunks = []

        for url in tqdm(urls, desc="Processing URLs"):
            chunks = self.run_url(url)
            all_chunks.extend(chunks)

        log.info(
            f"Batch URL ingestion complete | "
            f"{len(urls)} URLs | "
            f"{len(all_chunks)} total chunks"
        )
        return all_chunks

    def run_directory(self, directory: str) -> list[Chunk]:
        """
        Ingest all documents from a directory.

        This is the most common production use case —
        point at a folder of reports and ingest them all.
        """
        log.info(f"Starting directory ingestion: {directory}")

        documents = self.loader.load_directory(directory)
        if not documents:
            log.warning(f"No documents found in: {directory}")
            return []

        all_chunks = []

        for doc in tqdm(documents, desc="Processing documents"):
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        self._save_chunks(all_chunks, source_name="batch")

        log.info(
            f"Pipeline complete | "
            f"{len(documents)} documents | "
            f"{len(all_chunks)} total chunks"
        )
        return all_chunks

    def _save_chunks(self, chunks: list[Chunk], source_name: str):
        """
        Save chunks to disk as JSON.

        WHY SAVE TO DISK?
        Embedding generation is slow. By saving chunks here,
        we can run the embedding phase separately without
        re-ingesting documents every time.

        This is called caching — a critical production pattern.
        """
        if not chunks:
            return

        output_file = self.output_dir / f"{source_name}_chunks.json"

        chunks_data = [asdict(chunk) for chunk in chunks]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        log.info(f"Saved {len(chunks)} chunks to {output_file}")

    def load_cached_chunks(self, source_name: str) -> list[Chunk]:
        """
        Load previously processed chunks from disk.

        WHY THIS MATTERS:
        In production, you don't reprocess documents on every
        startup. You process once, cache, and reload the cache.
        Saves hours of processing time.
        """
        cache_file = self.output_dir / f"{source_name}_chunks.json"

        if not cache_file.exists():
            log.warning(f"No cached chunks found: {cache_file}")
            return []

        with open(cache_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        chunks = [Chunk(**chunk) for chunk in chunks_data]
        log.info(f"Loaded {len(chunks)} cached chunks from {cache_file}")
        return chunks
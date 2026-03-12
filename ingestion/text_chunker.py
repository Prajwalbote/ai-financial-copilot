"""
ingestion/text_chunker.py — Document Chunking
==============================================
WHY CHUNKING EXISTS:
    LLMs have token limits. You cannot feed a 200-page PDF
    into an LLM in one shot. We split documents into small
    overlapping chunks so:
    1. Each chunk fits in the LLM context window
    2. We can retrieve ONLY relevant chunks per query
    3. Overlap prevents losing context at chunk boundaries

WHAT YOU LEARN:
    - Token vs character based chunking
    - Why overlap matters
    - How to preserve metadata through the pipeline
"""

from dataclasses import dataclass, field
from typing import Optional
from utils.logger import get_logger
from utils.config import get_config
from ingestion.document_loader import Document

log = get_logger(__name__)


@dataclass
class Chunk:
    """
    Represents a single chunk of text from a document.

    WHY TRACK chunk_index AND total_chunks?
    During retrieval, knowing a chunk is "chunk 3 of 45"
    from "apple_10k.pdf" helps with:
    - Debugging (which part of doc did this come from?)
    - Context assembly (grab neighboring chunks if needed)
    - Citation (tell user exactly where answer came from)
    """
    text: str                    # The actual chunk text
    chunk_index: int             # Position in document (0, 1, 2...)
    total_chunks: int            # Total chunks in this document
    source: str                  # Original file path or URL
    doc_type: str                # "pdf", "html", "text"
    metadata: dict = field(default_factory=dict)  # Inherited from Document


class TextChunker:
    """
    Splits documents into overlapping chunks for RAG.

    The overlap strategy:
        [----chunk 1----]
                  [----chunk 2----]
                            [----chunk 3----]

    The overlapping region ensures sentences split at
    boundaries are fully captured in at least one chunk.

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk_document(doc)
        print(f"Split into {len(chunks)} chunks")
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_length: Optional[int] = None,
    ):
        cfg = get_config()

        # Use config values unless overridden
        self.chunk_size = chunk_size or cfg.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunking.chunk_overlap
        self.min_chunk_length = min_chunk_length or cfg.chunking.min_chunk_length

        log.info(
            f"TextChunker initialized | "
            f"chunk_size={self.chunk_size} | "
            f"overlap={self.chunk_overlap}"
        )

    def chunk_document(self, document: Document) -> list[Chunk]:
        """
        Split a Document into a list of Chunks.

        Args:
            document: Document object from DocumentLoader

        Returns:
            List of Chunk objects ready for embedding
        """
        if not document.content.strip():
            log.warning(f"Empty document received: {document.source}")
            return []

        # Split text into raw chunks
        raw_chunks = self._split_text(document.content)

        # Filter out chunks that are too small to be useful
        # (usually headers, page numbers, or noise)
        raw_chunks = [
            c for c in raw_chunks
            if len(c.strip()) >= self.min_chunk_length
        ]

        total = len(raw_chunks)

        # Wrap each raw text chunk into a Chunk object
        chunks = []
        for idx, text in enumerate(raw_chunks):
            chunk = Chunk(
                text=text.strip(),
                chunk_index=idx,
                total_chunks=total,
                source=document.source,
                doc_type=document.doc_type,
                metadata={
                    **document.metadata,   # Inherit parent document metadata
                    "chunk_index": idx,
                    "total_chunks": total,
                }
            )
            chunks.append(chunk)

        log.info(
            f"Chunked: {document.metadata.get('filename', document.source)} | "
            f"{total} chunks created"
        )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Chunk multiple documents at once.

        Args:
            documents: List of Document objects

        Returns:
            Flat list of all Chunk objects across all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        log.info(
            f"Batch chunking complete | "
            f"{len(documents)} documents → {len(all_chunks)} total chunks"
        )
        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """
        Core splitting logic with overlap.

        HOW IT WORKS:
        We slide a window of size chunk_size across the text,
        moving forward by (chunk_size - overlap) each step.

        Example with chunk_size=20, overlap=5:
        Text:     "The quick brown fox jumps over the lazy dog"
        Chunk 1:  "The quick brown fox" (chars 0-19)
        Chunk 2:  "fox jumps over the l" (chars 15-34)
        Chunk 3:  "the lazy dog" (chars 30-end)

        Args:
            text: Full document text

        Returns:
            List of text chunk strings
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # Don't cut in the middle of a word
            # Move end forward to the next space
            if end < text_length:
                next_space = text.find(' ', end)
                if next_space != -1 and next_space - end < 50:
                    end = next_space

            chunk = text[start:end]
            chunks.append(chunk)

            # Move start forward by (chunk_size - overlap)
            # This creates the overlapping effect
            start += self.chunk_size - self.chunk_overlap

        return chunks
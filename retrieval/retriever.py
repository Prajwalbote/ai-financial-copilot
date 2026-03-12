"""
retrieval/retriever.py — Semantic Retrieval
============================================
WHY THIS EXISTS:
    The retriever is the bridge between the user's question
    and the vector database. It:
    1. Embeds the user's query
    2. Searches the vector store
    3. Filters low-quality results
    4. Returns relevant chunks for the LLM

WHAT YOU LEARN:
    - Full retrieval pipeline
    - Similarity threshold filtering
    - How RAG retrieval actually works
"""

from embeddings.embedding_model import EmbeddingModel
from vectordb.faiss_store import FAISSVectorStore
from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class Retriever:
    """
    Semantic retriever for financial documents.

    Usage:
        retriever = Retriever()
        retriever.load_index("apple_10k")
        results = retriever.retrieve("What is Apple's revenue?")

        for r in results:
            print(r['score'], r['text'][:100])
    """

    def __init__(self):
        self.cfg = get_config()
        self.embedding_model = EmbeddingModel()
        self.vector_store = FAISSVectorStore()
        self.is_loaded = False
        log.info("Retriever initialized")

    def load_index(self, name: str) -> bool:
        """
        Load a saved vector index by name.

        Args:
            name: Index name used when saving (e.g. "apple_10k")

        Returns:
            True if loaded successfully
        """
        success = self.vector_store.load(name)
        if success:
            self.is_loaded = True
            log.info(f"Retriever ready with index: {name}")
        return success

    def build_and_save(
        self,
        embeddings,
        chunks_metadata: list[dict],
        name: str
    ):
        """
        Build index from embeddings and save it.

        Use this when setting up for the first time.
        """
        self.vector_store.build_index(embeddings, chunks_metadata)
        self.vector_store.save(name)
        self.is_loaded = True
        log.info(f"Index built and saved: {name}")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None
    ) -> list[dict]:
        """
        Retrieve most relevant chunks for a query.

        Args:
            query:     User's question in natural language
            top_k:     Max number of chunks to return
            min_score: Minimum similarity score (filter noise)

        Returns:
            List of relevant chunk dicts sorted by relevance

        Full flow:
            "What is Apple revenue?"
                → embed → [0.21, -0.54, ...]
                → FAISS search → top 5 chunks
                → filter by min_score
                → return relevant chunks
        """
        if not self.is_loaded:
            log.error("No index loaded. Call load_index() first")
            return []

        top_k = top_k or self.cfg.retrieval.top_k
        min_score = min_score or self.cfg.retrieval.similarity_threshold

        log.info(f"Retrieving for query: '{query[:60]}...'")

        # Step 1 — Embed the query
        query_vector = self.embedding_model.embed_text(query)

        # Step 2 — Search vector store
        results = self.vector_store.search(query_vector, top_k=top_k)

        # Step 3 — Filter results below similarity threshold
        # This removes irrelevant chunks that happened to be
        # the "least bad" matches even if they're not relevant
        filtered = [r for r in results if r["score"] >= min_score]

        if len(filtered) < len(results):
            log.info(
                f"Filtered {len(results) - len(filtered)} "
                f"low-score chunks (below {min_score})"
            )

        log.info(
            f"Retrieved {len(filtered)} relevant chunks | "
            f"top score: {filtered[0]['score']:.3f}"
            if filtered else "No relevant chunks found"
        )

        return filtered
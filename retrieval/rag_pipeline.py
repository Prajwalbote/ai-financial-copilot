"""
retrieval/rag_pipeline.py — Full RAG Pipeline
=============================================
WHY THIS EXISTS:
    This is the brain of the whole system.
    It connects retrieval + LLM into one clean interface.

    The rest of the system (API, UI) only needs to call
    this one class. They don't need to know about FAISS,
    embeddings, or prompt templates.

    This is called "encapsulation" — hiding complexity
    behind a simple interface.

WHAT YOU LEARN:
    - How to orchestrate multiple AI components
    - The complete RAG loop
    - Different query modes (QA, summary, risk)
"""

from llm.base_llm import FinancialLLM
from llm.prompt_templates import PromptTemplates
from retrieval.retriever import Retriever
from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for financial question answering.

    This is the main interface for the entire AI system.
    FastAPI and Streamlit will call THIS class directly.

    Usage:
        rag = RAGPipeline()
        rag.load_index("apple_10k")

        answer = rag.answer("What is Apple's revenue?")
        summary = rag.summarize()
        risks = rag.analyze_risks()
    """

    def __init__(self):
        self.cfg = get_config()
        self.retriever = Retriever()
        self.llm = FinancialLLM()
        self.templates = PromptTemplates()
        self.index_loaded = False
        log.info("RAGPipeline initialized")

    def load_index(self, name: str) -> bool:
        """
        Load a vector index by name.

        Args:
            name: Index name (e.g. "apple_10k")

        Returns:
            True if loaded successfully
        """
        success = self.retriever.load_index(name)
        if success:
            self.index_loaded = True
            log.info(f"RAGPipeline ready with index: {name}")
        return success

    def answer(self, question: str) -> dict:
        """
        Answer a financial question using RAG.

        This is the core method — the complete RAG loop.

        Args:
            question: User's natural language question

        Returns:
            Dict with answer, sources, and retrieved chunks

        WHY RETURN A DICT?
        The API needs the answer text.
        The UI needs the source citations.
        The logs need the chunks for debugging.
        A dict gives everyone what they need.
        """
        if not self.index_loaded:
            return {
                "answer": "No index loaded. Please ingest documents first.",
                "sources": [],
                "chunks": []
            }

        log.info(f"RAG query: '{question}'")

        # Step 1 — Retrieve relevant chunks
        chunks = self.retriever.retrieve(question)

        if not chunks:
            return {
                "answer": "I cannot find relevant information in the documents.",
                "sources": [],
                "chunks": []
            }

        # Step 2 — Format chunks into context
        context = self.templates.format_context(chunks)

        # Step 3 — Build prompt
        prompt = self.templates.qa_prompt(context, question)

        # Step 4 — Generate answer
        answer = self.llm.generate(prompt)

        # Step 5 — Extract unique sources for citation
        sources = list(set(c['source'] for c in chunks))

        result = {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "num_chunks_used": len(chunks)
        }

        log.info(f"RAG answer generated | length={len(answer)}")
        return result

    def summarize(self, question: str = None) -> dict:
        """
        Summarize retrieved document content.

        Args:
            question: Optional focus for the summary

        Returns:
            Dict with summary and sources
        """
        search_query = question or "financial performance summary overview"

        chunks = self.retriever.retrieve(
            search_query,
            top_k=5   # Use more chunks for summaries
        )

        if not chunks:
            return {"summary": "No content found to summarize.", "sources": []}

        context = self.templates.format_context(chunks)
        prompt = self.templates.summarization_prompt(context)
        summary = self.llm.generate(prompt)
        sources = list(set(c['source'] for c in chunks))

        return {
            "summary": summary,
            "sources": sources,
            "chunks_used": len(chunks)
        }

    def analyze_risks(self) -> dict:
        """
        Perform financial risk analysis on loaded documents.

        Returns:
            Dict with risk analysis and sources
        """
        chunks = self.retriever.retrieve(
            "financial risk losses liabilities debt obligations",
            top_k=5
        )

        if not chunks:
            return {"risks": "No risk information found.", "sources": []}

        context = self.templates.format_context(chunks)
        prompt = self.templates.risk_analysis_prompt(context)
        risks = self.llm.generate(prompt)
        sources = list(set(c['source'] for c in chunks))

        return {
            "risks": risks,
            "sources": sources,
            "chunks_used": len(chunks)
        }
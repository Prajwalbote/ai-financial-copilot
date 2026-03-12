"""
llm/prompt_templates.py — Financial Prompt Engineering
=======================================================
WHY PROMPT ENGINEERING MATTERS:
    The same LLM gives completely different quality answers
    depending on how you phrase the prompt.

    Bad prompt:  "Answer this: {question}"
    Good prompt: Structured context + clear instructions
                 + output format specification

WHAT YOU LEARN:
    - How to write effective RAG prompts
    - Why prompt structure affects answer quality
    - Different prompt types for different tasks
"""

from utils.logger import get_logger

log = get_logger(__name__)


class PromptTemplates:
    """
    Collection of prompt templates for financial tasks.

    WHY SEPARATE TEMPLATES FROM LLM CODE?
    Prompts change frequently during development.
    Keeping them separate means you can improve prompts
    without touching LLM or retrieval code.
    This is standard practice in production AI systems.
    """

    @staticmethod
    def qa_prompt(context: str, question: str) -> str:
        """
        Question answering prompt with retrieved context.

        This is the core RAG prompt — most queries use this.

        Args:
            context:  Retrieved chunks joined together
            question: User's question

        Returns:
            Formatted prompt string ready for LLM
        """
        return f"""Read the financial data below and answer the question.
Find exact numbers and figures from the text.
If you see dollar amounts, include them in your answer.

Financial Data:
{context}

Question: {question}

Give a specific answer with numbers if available:"""

    @staticmethod
    def summarization_prompt(context: str) -> str:
        """
        Summarize a financial document or section.

        Used when user asks for a summary rather than
        a specific question.
        """
        return f"""Summarize the key financial numbers from this data.
Include total revenue, costs, income, and any important figures.

Financial Data:
{context}

Key financial summary:"""

    @staticmethod
    def risk_analysis_prompt(context: str) -> str:
        """
        Analyze financial risks from document content.

        Used for risk assessment queries.
        """
        return f"""Analyze financial risks from the data below.
Look for liabilities, losses, debt, and declining metrics.

Financial Data:
{context}

Financial risks identified:"""

    @staticmethod
    def format_context(chunks: list[dict]) -> str:
        """
        Format retrieved chunks into clean context string.

        WHY FORMAT CAREFULLY?
        Raw chunks have no structure. By numbering them
        and adding source info, the LLM can reference
        specific sources in its answer.

        Args:
            chunks: List of chunk dicts from retriever

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'Unknown')
            # Get just the filename from full path
            source_name = source.split('\\')[-1].split('/')[-1]
            text = chunk.get('text', '')
            score = chunk.get('score', 0)

            context_parts.append(
                f"[Source {i}: {source_name} | "
                f"Relevance: {score:.2f}]\n{text}"
            )

        return "\n\n".join(context_parts)
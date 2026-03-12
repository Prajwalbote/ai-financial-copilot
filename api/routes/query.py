"""
api/routes/query.py — RAG Query Endpoints
==========================================
WHY SEPARATE ROUTE FILES?
    Putting all routes in main.py creates a massive file.
    Separating by domain (query, ingest, predict) keeps
    code organized and maintainable.

    This is the Router pattern — standard in FastAPI.
"""

from fastapi import APIRouter, HTTPException
from api.schemas import (
    QuestionRequest,
    SummarizeRequest,
    AnswerResponse,
    SummaryResponse
)
from retrieval.rag_pipeline import RAGPipeline
from utils.cache import get_cache
from utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["RAG Query"])

# Single shared RAGPipeline instance
# Loading the LLM takes 2-3 seconds.
# We load once at startup, reuse for every request.
_rag_pipeline = None


def get_rag_pipeline(index_name: str) -> RAGPipeline:
    """Get or create RAG pipeline with loaded index."""
    global _rag_pipeline

    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()

    if not _rag_pipeline.index_loaded:
        success = _rag_pipeline.load_index(index_name)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Index '{index_name}' not found. "
                       f"Please ingest documents first."
            )

    return _rag_pipeline


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a financial question using RAG with caching.

    Flow:
        1. Check cache for existing answer
        2. If cached → return instantly
        3. If not cached → run RAG pipeline
        4. Cache the result for future requests
        5. Return answer with sources
    """
    log.info(f"Question received: {request.question[:60]}")

    # Step 1 — Check cache first
    # Same question asked twice = instant response
    cache = get_cache()
    cache_key = cache.make_key(
        request.question,
        request.index_name
    )
    cached_result = cache.get(cache_key)

    if cached_result:
        log.info("Returning cached response")
        return AnswerResponse(**cached_result)

    # Step 2 — Not cached, run RAG pipeline
    try:
        rag = get_rag_pipeline(request.index_name)
        result = rag.answer(request.question)

        response = AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            num_chunks_used=result.get("num_chunks_used", 0),
            index_name=request.index_name
        )

        # Step 3 — Cache for next time
        cache.set(cache_key, response.model_dump())

        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error answering question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_document(request: SummarizeRequest):
    """
    Summarize financial documents in the index.

    - Retrieves key chunks from vector database
    - Generates concise financial summary
    - Caches result for repeated requests
    """
    log.info(f"Summarize request for index: {request.index_name}")

    # Check cache
    cache = get_cache()
    cache_key = cache.make_key(
        f"summarize:{request.focus or 'general'}",
        request.index_name
    )
    cached_result = cache.get(cache_key)

    if cached_result:
        log.info("Returning cached summary")
        return SummaryResponse(**cached_result)

    try:
        rag = get_rag_pipeline(request.index_name)
        result = rag.summarize(request.focus)

        response = SummaryResponse(
            summary=result["summary"],
            sources=result["sources"],
            chunks_used=result.get("chunks_used", 0)
        )

        # Cache the summary
        cache.set(cache_key, response.model_dump())

        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error summarizing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error summarizing: {str(e)}"
        )


@router.post("/analyze-risks")
async def analyze_risks(request: SummarizeRequest):
    """
    Perform financial risk analysis on documents.

    - Retrieves risk-related chunks
    - Generates structured risk analysis
    - Caches result for repeated requests
    """
    log.info(f"Risk analysis request for: {request.index_name}")

    # Check cache
    cache = get_cache()
    cache_key = cache.make_key(
        "risk_analysis",
        request.index_name
    )
    cached_result = cache.get(cache_key)

    if cached_result:
        log.info("Returning cached risk analysis")
        return cached_result

    try:
        rag = get_rag_pipeline(request.index_name)
        result = rag.analyze_risks()

        # Cache the result
        cache.set(cache_key, result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error analyzing risks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing risks: {str(e)}"
        )


@router.get("/cache-stats")
async def get_cache_stats():
    """
    Get cache performance statistics.

    Useful for monitoring in production.
    Shows hit rate, total entries, etc.
    """
    cache = get_cache()
    return cache.get_stats()
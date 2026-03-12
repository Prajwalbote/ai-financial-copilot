"""
api/schemas.py — Request/Response Models
=========================================
WHY PYDANTIC SCHEMAS?
    FastAPI uses Pydantic to:
    1. Validate incoming requests automatically
    2. Document the API (auto-generates /docs)
    3. Serialize responses to JSON

    If a request is missing a required field or has
    wrong type, FastAPI returns a clear error automatically.
    No manual validation code needed.

WHAT YOU LEARN:
    - Pydantic data models
    - Request/response separation
    - API contract design
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Models ───────────────────────────────────────────

class QuestionRequest(BaseModel):
    """Request body for asking a financial question."""
    question: str = Field(
        ...,  # ... means required
        min_length=5,
        max_length=500,
        description="Financial question to answer",
        example="What is Apple's total revenue for FY2023?"
    )
    index_name: str = Field(
        default="apple_10k",
        description="Name of the vector index to search"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,  # greater than or equal to 1
        le=20, # less than or equal to 20
        description="Number of chunks to retrieve"
    )


class SummarizeRequest(BaseModel):
    """Request body for document summarization."""
    index_name: str = Field(
        default="apple_10k",
        description="Name of the vector index to summarize"
    )
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus area for summary"
    )


class IngestRequest(BaseModel):
    """Request body for URL ingestion."""
    url: str = Field(
        ...,
        description="URL to ingest",
        example="https://finance.yahoo.com/news/..."
    )
    index_name: str = Field(
        default="documents",
        description="Index name to add documents to"
    )


# ── Response Models ──────────────────────────────────────────

class AnswerResponse(BaseModel):
    """Response from RAG question answering."""
    answer: str
    sources: list[str]
    num_chunks_used: int
    index_name: str


class SummaryResponse(BaseModel):
    """Response from summarization."""
    summary: str
    sources: list[str]
    chunks_used: int


class PredictionResponse(BaseModel):
    """Response from stock prediction."""
    ticker: str
    direction: str
    confidence: float
    current_price: float
    indicators: dict


class RiskResponse(BaseModel):
    """Response from risk analysis."""
    ticker: str
    risk_score: float
    risk_level: str
    metrics: dict
    interpretation: str


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    status: str
    chunks_created: int
    source: str


class HealthResponse(BaseModel):
    """Response from health check."""
    status: str
    version: str
    components: dict
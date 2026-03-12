"""
api/main.py — FastAPI Application Entry Point
=============================================
WHY FASTAPI?
    - Automatic API documentation at /docs
    - Built-in request validation via Pydantic
    - Async support for handling multiple requests
    - Fast — one of the fastest Python frameworks
    - Used in production by major tech companies

WHAT YOU LEARN:
    - How to structure a production API
    - Middleware (CORS, logging)
    - Startup/shutdown events
    - Router mounting pattern
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

from api.routes import query, predict, ingest
from api.schemas import HealthResponse
from utils.logger import get_logger
from utils.config import get_config

log = get_logger(__name__)
cfg = get_config()


# ── Create FastAPI App ───────────────────────────────────────
app = FastAPI(
    title="AI Financial Research Copilot",
    description="""
    A RAG-powered financial research assistant.

    Features:
    - Answer questions from financial documents
    - Summarize financial reports
    - Predict stock price direction
    - Analyze financial risk
    - Ingest new documents via URL or PDF upload
    """,
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc"      # ReDoc UI at /redoc
)


# ── CORS Middleware ──────────────────────────────────────────
# WHY CORS?
# Browsers block requests from different origins by default.
# CORS headers tell the browser it's OK for our Streamlit UI
# to call this API even though they run on different ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production: specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging Middleware ───────────────────────────────
@app.middleware("http")
async def log_requests(request, call_next):
    """Log every request with timing information."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    log.info(
        f"{request.method} {request.url.path} | "
        f"status={response.status_code} | "
        f"duration={duration:.3f}s"
    )
    return response


# ── Startup Event ────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Runs when the API server starts.
    Use for loading models, connecting to databases, etc.
    """
    log.info("="*50)
    log.info("AI Financial Research Copilot API starting...")
    log.info(f"Environment: {cfg.project.environment}")
    log.info("API docs available at: http://localhost:8000/docs")
    log.info("="*50)


# ── Shutdown Event ───────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the API server stops."""
    log.info("API shutting down...")


# ── Mount Routers ────────────────────────────────────────────
# Each router handles a group of related endpoints
app.include_router(query.router)
app.include_router(predict.router)
app.include_router(ingest.router)


# ── Root Endpoint ────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    """API root — confirms the service is running."""
    return {
        "message": "AI Financial Research Copilot API",
        "version": "1.0.0",
        "docs": "http://localhost:8000/docs"
    }


# ── Health Check ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    System health check endpoint.

    WHY HEALTH CHECKS?
    Load balancers and monitoring tools ping /health
    to know if the service is running correctly.
    Standard in every production deployment.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "api": "running",
            "vector_db": "ready",
            "llm": "ready",
            "ml_model": "ready"
        }
    )


# ── Run Directly ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=cfg.api.reload,
        workers=cfg.api.workers
    )
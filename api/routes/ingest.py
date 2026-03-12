"""
api/routes/ingest.py — Document Ingestion Endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from api.schemas import IngestRequest, IngestResponse
from ingestion.pipeline import IngestionPipeline
from embeddings.batch_embedder import BatchEmbedder
from retrieval.retriever import Retriever
from utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["Ingestion"])

_pipeline = None
_embedder = None


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


@router.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: IngestRequest):
    """
    Ingest a financial document from a URL.

    - Fetches and processes the URL content
    - Creates chunks and embeddings
    - Adds to vector database
    """
    log.info(f"URL ingestion request: {request.url}")

    try:
        pipeline = get_pipeline()
        chunks = pipeline.run_url(request.url)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract content from: {request.url}"
            )

        # Embed and index the new chunks
        embedder = BatchEmbedder()
        embeddings = embedder.embed_chunks(chunks)
        embedder.save_embeddings(
            embeddings, chunks, request.index_name
        )

        # Build retrieval index
        retriever = Retriever()
        chunks_as_dicts = [
            {
                "text": c.text,
                "source": c.source,
                "doc_type": c.doc_type,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "metadata": c.metadata
            }
            for c in chunks
        ]
        retriever.build_and_save(
            embeddings, chunks_as_dicts, request.index_name
        )

        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            source=request.url
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error ingesting URL: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting URL: {str(e)}"
        )


@router.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    index_name: str = "documents"
):
    """
    Upload and ingest a PDF document.

    - Accepts PDF file upload
    - Processes and indexes automatically
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    log.info(f"PDF upload: {file.filename}")

    try:
        # Save uploaded file temporarily
        upload_path = Path("data/raw") / file.filename
        content = await file.read()

        with open(upload_path, "wb") as f:
            f.write(content)

        # Ingest the saved PDF
        pipeline = get_pipeline()
        chunks = pipeline.run_pdf(str(upload_path))

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from PDF"
            )

        # Embed and index
        embedder = BatchEmbedder()
        embeddings = embedder.embed_chunks(chunks)
        embedder.save_embeddings(embeddings, chunks, index_name)

        retriever = Retriever()
        chunks_as_dicts = [
            {
                "text": c.text,
                "source": c.source,
                "doc_type": c.doc_type,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "metadata": c.metadata
            }
            for c in chunks
        ]
        retriever.build_and_save(embeddings, chunks_as_dicts, index_name)

        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            source=file.filename
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error ingesting PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting PDF: {str(e)}"
        )
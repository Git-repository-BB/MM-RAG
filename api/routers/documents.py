from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from api.dependencies import get_ingestion_service
from api.jobs import job_store
from api.schemas import (
    DocumentSummary,
    IndexStatusResponse,
    IngestJobResponse,
    IngestJobStatusResponse,
    IngestRequest,
    IngestResponse,
    ParseRequest,
    ParseResponse,
)
from exception.custom_exception import DocumentPortalException
from src.services.ingestion_service import IngestionService

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    service: IngestionService = Depends(get_ingestion_service),
) -> list[DocumentSummary]:
    return [DocumentSummary(**item) for item in service.discover_pdfs()]


@router.post("/upload", response_model=DocumentSummary)
async def upload_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentSummary:
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        result = service.save_upload(filename=file.filename or "upload.pdf", data=data)
    except DocumentPortalException as exc:
        raise HTTPException(status_code=500, detail=exc.error_message) from exc

    return DocumentSummary(**result)


@router.post("/parse", response_model=ParseResponse)
def parse_document(
    request: ParseRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> ParseResponse:
    try:
        result = service.parse_document(
            pdf_path=request.pdf_path,
            fingerprint=request.fingerprint,
            force=request.force,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentPortalException as exc:
        raise HTTPException(status_code=500, detail=exc.error_message) from exc

    return ParseResponse(**result)


@router.get("/index-status", response_model=IndexStatusResponse)
def index_status(
    collection_name: str,
    filename: str,
    fingerprint: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> IndexStatusResponse:
    try:
        result = service.check_index_status(
            collection_name=collection_name,
            filename=filename,
            fingerprint=fingerprint,
        )
    except DocumentPortalException as exc:
        raise HTTPException(status_code=500, detail=exc.error_message) from exc

    if result is None:
        return IndexStatusResponse(found=False)

    return IndexStatusResponse(found=True, **result)


@router.post("/ingest", response_model=IngestJobResponse, status_code=202)
def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestJobResponse:
    """
    Start ingestion as a background job and return immediately.

    Embeddings and Qdrant upserts can take a while for large PDFs, so this
    endpoint hands the work off to a background thread (see api/jobs.py) and
    responds with a job_id right away. Poll GET /ingest/{job_id} for status.
    """
    job = job_store.create()

    def run_ingestion() -> dict:
        return service.ingest_document(
            pdf_path=request.pdf_path,
            fingerprint=request.fingerprint,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            replace_existing=request.replace_existing,
            force=request.force,
        )

    background_tasks.add_task(job_store.run, job.job_id, run_ingestion)

    return IngestJobResponse(job_id=job.job_id, status=job.status)


@router.get("/ingest/{job_id}", response_model=IngestJobStatusResponse)
def ingest_job_status(job_id: str) -> IngestJobStatusResponse:
    job = job_store.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    return IngestJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=IngestResponse(**job.result) if job.result is not None else None,
        error=job.error,
    )

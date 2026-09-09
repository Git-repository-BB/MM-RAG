from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_ingestion_service
from api.schemas import (
    DocumentSummary,
    IndexStatusResponse,
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


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    try:
        result = service.ingest_document(
            pdf_path=request.pdf_path,
            fingerprint=request.fingerprint,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            replace_existing=request.replace_existing,
            force=request.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DocumentPortalException as exc:
        raise HTTPException(status_code=500, detail=exc.error_message) from exc

    return IngestResponse(**result)

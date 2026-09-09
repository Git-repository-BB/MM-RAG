from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_retrieval_service
from api.schemas import RetrieveRequest
from exception.custom_exception import DocumentPortalException
from src.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/api/retrieve", tags=["retrieval"])


@router.post("")
def retrieve(
    request: RetrieveRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> list[dict]:
    try:
        return service.retrieve(
            query=request.query,
            collection_name=request.collection_name,
            k=request.k,
            filename=request.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentPortalException as exc:
        raise HTTPException(status_code=500, detail=exc.error_message) from exc

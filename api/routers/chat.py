from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_generation_service
from api.schemas import ChatRequest, ChatResponse
from exception.custom_exception import DocumentPortalException
from src.services.generation_service import GenerationService

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
def ask(
    request: ChatRequest,
    service: GenerationService = Depends(get_generation_service),
) -> ChatResponse:
    try:
        result = service.ask(
            query=request.query,
            collection_name=request.collection_name,
            model_name=request.model_name,
            k=request.k,
            max_images=request.max_images,
            filename=request.filename,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentPortalException as exc:
        raise HTTPException(status_code=500, detail=exc.error_message) from exc

    return ChatResponse(**result)

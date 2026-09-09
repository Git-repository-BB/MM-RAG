from __future__ import annotations

from functools import lru_cache

from src.services.generation_service import GenerationService
from src.services.ingestion_service import IngestionService
from src.services.retrieval_service import RetrievalService


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService()


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    return RetrievalService()


@lru_cache(maxsize=1)
def get_generation_service() -> GenerationService:
    return GenerationService()

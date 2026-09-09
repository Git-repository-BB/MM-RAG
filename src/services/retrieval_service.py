from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from src.retriever import MultimodalQdrantRetriever

logger = CustomLogger().get_logger(__name__)


@lru_cache(maxsize=8)
def _get_retriever(collection_name: str) -> MultimodalQdrantRetriever:
    return MultimodalQdrantRetriever(collection_name=collection_name)


class RetrievalService:
    """
    Retrieval Service.

    Owns similarity search against Qdrant collections. Kept separate from
    generation so the API layer can retrieve raw evidence without invoking
    an LLM (e.g. for debugging or non-generative UIs).
    """

    def retrieve(
        self,
        *,
        query: str,
        collection_name: str,
        k: int = 6,
        filename: str | None = None,
        content_types: Iterable[str] | None = None,
        page_number: int | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> list[dict[str, Any]]:
        try:
            retriever = _get_retriever(collection_name)

            results = retriever.retrieve_with_scores(
                query=query,
                k=k,
                filename=filename,
                content_types=content_types,
                page_number=page_number,
                page_from=page_from,
                page_to=page_to,
            )

            return [
                {
                    "content": document.page_content,
                    "metadata": dict(document.metadata),
                    "score": float(score),
                }
                for document, score in results
            ]
        except DocumentPortalException:
            raise
        except Exception as exc:
            logger.exception("retrieval_service_failed", query=query, error=str(exc))
            raise DocumentPortalException("Failed to retrieve documents", exc) from exc

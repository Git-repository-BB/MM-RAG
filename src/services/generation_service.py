from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from src.generation import MultimodalRAGGenerator

logger = CustomLogger().get_logger(__name__)


@lru_cache(maxsize=8)
def _get_generator(
    collection_name: str,
    model_name: str,
    max_images: int,
) -> MultimodalRAGGenerator:
    return MultimodalRAGGenerator(
        collection_name=collection_name,
        model_name=model_name,
        max_images=max_images,
    )


class GenerationService:
    """
    Generation Service.

    Owns retrieval-augmented answer generation. Delegates retrieval to the
    same Qdrant-backed retriever used by the Retrieval Service, then invokes
    the multimodal LLM with the retrieved text/table/image evidence.
    """

    def ask(
        self,
        *,
        query: str,
        collection_name: str,
        model_name: str = "gpt-4.1-mini",
        k: int = 6,
        max_images: int = 4,
        filename: str | None = None,
    ) -> dict[str, Any]:
        try:
            generator = _get_generator(collection_name, model_name, max_images)
            return generator.answer_question(query=query, k=k, filename=filename)
        except DocumentPortalException:
            raise
        except Exception as exc:
            logger.exception("generation_service_failed", query=query, error=str(exc))
            raise DocumentPortalException("Failed to generate an answer", exc) from exc

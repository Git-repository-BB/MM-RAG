from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from api.routers import chat, documents, retrieval

app = FastAPI(
    title="MM-RAG API",
    description=(
        "API layer for the multimodal RAG pipeline. Frontends (Streamlit, "
        "React/Next.js, etc.) talk to this service instead of calling the "
        "ingestion/retrieval/generation pipeline directly."
    ),
    version="1.0.0",
)

# Local/demo CORS policy. Restrict allow_origins to known frontend origins
# before deploying this API beyond a local development environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(retrieval.router)
app.include_router(chat.router)


@app.get("/health")
def health() -> dict[str, object]:
    qdrant_url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_Cluster_Endpoint")

    return {
        "status": "ok",
        "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "qdrant_url_configured": bool(qdrant_url),
        "qdrant_api_key_configured": bool(os.getenv("QDRANT_API_KEY")),
        "tesseract_path_configured": bool(os.getenv("TESSERACT_PATH")),
        "default_collection_name": os.getenv("QDRANT_COLLECTION_NAME") or "mm-rag-documents",
        "default_chat_model": os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini",
    }

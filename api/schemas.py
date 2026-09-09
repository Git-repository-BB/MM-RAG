from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    filename: str
    path: str
    fingerprint: str
    size_bytes: int


class ParseRequest(BaseModel):
    pdf_path: str
    fingerprint: str
    force: bool = False


class ParseResponse(BaseModel):
    output_dir: str
    restored: bool
    restore_mode: Optional[str] = None
    pages: list[dict[str, Any]]
    images: list[dict[str, Any]]
    tables: list[dict[str, Any]]
    documents_count: int


class IngestRequest(BaseModel):
    pdf_path: str
    fingerprint: str
    collection_name: str
    chunk_size: int = Field(default=2000, ge=200, le=10000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    replace_existing: bool = True
    force: bool = False


class IngestResponse(BaseModel):
    collection_name: str
    loaded_existing: bool
    match_mode: Optional[str] = None
    collection_points: Optional[int] = None
    source_document_count: Optional[int] = None
    input_document_count: Optional[int] = None
    indexed_document_count: Optional[int] = None
    inserted_point_count: Optional[int] = None
    file_sha256: Optional[str] = None


class IndexStatusResponse(BaseModel):
    found: bool
    collection_name: Optional[str] = None
    match_mode: Optional[str] = None
    collection_points: Optional[int] = None


class RetrieveRequest(BaseModel):
    query: str
    collection_name: str
    k: int = Field(default=6, gt=0, le=50)
    filename: Optional[str] = None


class ChatRequest(BaseModel):
    query: str
    collection_name: str
    model_name: str = "gpt-4.1-mini"
    k: int = Field(default=6, gt=0, le=50)
    max_images: int = Field(default=4, ge=0, le=10)
    filename: Optional[str] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict[str, Any]]
    used_images: list[dict[str, Any]]
    retrieval_count: int
    model_name: str
    collection_name: str
    usage: Optional[dict[str, Any]] = None

from __future__ import annotations

import os
import time
from typing import Any, Callable

import requests

API_BASE_URL = (os.getenv("MMRAG_API_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")

_DEFAULT_TIMEOUT = 300


class ApiClientError(Exception):
    """Raised when the MM-RAG API returns an error response."""


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return

    try:
        detail = response.json().get("detail")
    except Exception:
        detail = response.text

    raise ApiClientError(detail or f"API request failed with status {response.status_code}")


def get_health() -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/health", timeout=10)
    _raise_for_status(response)
    return response.json()


def list_documents() -> list[dict[str, Any]]:
    response = requests.get(f"{API_BASE_URL}/api/documents", timeout=_DEFAULT_TIMEOUT)
    _raise_for_status(response)
    return response.json()


def upload_document(filename: str, data: bytes) -> dict[str, Any]:
    files = {"file": (filename, data, "application/pdf")}
    response = requests.post(
        f"{API_BASE_URL}/api/documents/upload",
        files=files,
        timeout=_DEFAULT_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def parse_document(pdf_path: str, fingerprint: str, force: bool = False) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/api/documents/parse",
        json={"pdf_path": pdf_path, "fingerprint": fingerprint, "force": force},
        timeout=_DEFAULT_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def index_status(collection_name: str, filename: str, fingerprint: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/api/documents/index-status",
        params={
            "collection_name": collection_name,
            "filename": filename,
            "fingerprint": fingerprint,
        },
        timeout=_DEFAULT_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def start_ingest_job(
    *,
    pdf_path: str,
    fingerprint: str,
    collection_name: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 120,
    replace_existing: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Kick off ingestion on the API and return immediately with a job_id."""
    response = requests.post(
        f"{API_BASE_URL}/api/documents/ingest",
        json={
            "pdf_path": pdf_path,
            "fingerprint": fingerprint,
            "collection_name": collection_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "replace_existing": replace_existing,
            "force": force,
        },
        timeout=30,
    )
    _raise_for_status(response)
    return response.json()


def get_ingest_job_status(job_id: str) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/api/documents/ingest/{job_id}",
        timeout=30,
    )
    _raise_for_status(response)
    return response.json()


def ingest_document(
    *,
    pdf_path: str,
    fingerprint: str,
    collection_name: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 120,
    replace_existing: bool = True,
    force: bool = False,
    poll_interval: float = 1.5,
    on_poll: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Start an ingestion job and poll it to completion.

    Ingestion runs asynchronously on the API (a background thread per job).
    This helper starts the job, then polls GET /ingest/{job_id} until the job
    reports "completed" or "failed", calling `on_poll` with each status
    payload so callers (e.g. Streamlit) can show live progress.
    """
    job = start_ingest_job(
        pdf_path=pdf_path,
        fingerprint=fingerprint,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        replace_existing=replace_existing,
        force=force,
    )

    while True:
        status = get_ingest_job_status(job["job_id"])

        if on_poll:
            on_poll(status)

        if status["status"] == "completed":
            return status["result"]

        if status["status"] == "failed":
            raise ApiClientError(status.get("error") or "Ingestion job failed")

        time.sleep(poll_interval)


def ask(
    *,
    query: str,
    collection_name: str,
    model_name: str,
    k: int,
    max_images: int,
    filename: str | None = None,
) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/api/chat/ask",
        json={
            "query": query,
            "collection_name": collection_name,
            "model_name": model_name,
            "k": k,
            "max_images": max_images,
            "filename": filename,
        },
        timeout=_DEFAULT_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()

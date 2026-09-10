from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from exception.custom_exception import DocumentPortalException

JobStatus = str  # "pending" | "running" | "completed" | "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore:
    """
    In-memory registry for background (asynchronous) pipeline work.

    A request handler creates a job and schedules `run` as a FastAPI
    BackgroundTask. Starlette executes background tasks in a worker thread
    *after* the response has already been sent, so the HTTP request that
    kicked off ingestion returns immediately with a job_id instead of
    blocking until embeddings/upsert finish. Clients poll the job status
    endpoint until it reports "completed" or "failed".

    This store is per-process and in-memory, which is fine for a single
    local/demo deployment. For multi-worker or multi-process deployments,
    replace it with a shared store (e.g. Redis) so every worker can see
    every job.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(job_id=str(uuid.uuid4()))

        with self._lock:
            self._jobs[job.job_id] = job

        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)

            if job is None:
                return

            for key, value in fields.items():
                setattr(job, key, value)

            job.updated_at = time.time()

    def run(self, job_id: str, func: Callable[[], dict[str, Any]]) -> None:
        """Execute func() on the background thread and record the outcome."""
        self._update(job_id, status="running")

        try:
            result = func()
            self._update(job_id, status="completed", result=result)
        except DocumentPortalException as exc:
            self._update(job_id, status="failed", error=exc.error_message)
        except Exception as exc:  # noqa: BLE001 - persist any failure on the job
            self._update(job_id, status="failed", error=str(exc))


# Process-wide singleton shared by all request handlers.
job_store = JobStore()

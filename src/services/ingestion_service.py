from __future__ import annotations

import hashlib
import json
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import os

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from src.ingestion import MultimodalDocumentIngestion
from src.parsing import ComplexPDFParser

logger = CustomLogger().get_logger(__name__)


DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PARSED_ROOT = DATA_DIR / "parsed_pdf_output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PARSED_ROOT.mkdir(parents=True, exist_ok=True)

PARSED_JSON_FILES = (
    "page_records.json",
    "image_records.json",
    "table_records.json",
)

DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME") or "mm-rag-documents"


@lru_cache(maxsize=4)
def _get_qdrant_client(qdrant_url: str, qdrant_api_key: str | None) -> QdrantClient:
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def _fingerprint_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fingerprint_file(path: Path) -> str:
    sha = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(block)

    return sha.hexdigest()


def _parsed_artifacts_complete(output_dir: Path) -> bool:
    return all((output_dir / filename).exists() for filename in PARSED_JSON_FILES)


def _parse_manifest_path(output_dir: Path) -> Path:
    return output_dir / "parse_manifest.json"


def _write_parse_manifest(output_dir: Path, pdf_path: Path, fingerprint: str) -> None:
    manifest = {
        "filename": pdf_path.name,
        "pdf_path": str(pdf_path.resolve()),
        "sha256": fingerprint,
    }

    _parse_manifest_path(output_dir).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_parse_manifest(output_dir: Path) -> dict[str, Any] | None:
    manifest_path = _parse_manifest_path(output_dir)

    if not manifest_path.exists():
        return None

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _legacy_output_mentions_pdf(output_dir: Path, filename: str) -> bool:
    """
    Old terminal parsing did not create a manifest.

    rag_ready_documents.md contains metadata with the original source path,
    therefore the filename can be used to identify that legacy output.
    """
    rag_file = output_dir / "rag_ready_documents.md"

    if not rag_file.exists():
        return False

    try:
        content = rag_file.read_text(encoding="utf-8", errors="ignore")
        return filename.lower() in content.lower()
    except Exception:
        return False


def _find_existing_parsed_output(
    pdf_path: Path,
    fingerprint: str,
) -> tuple[Path | None, str | None]:
    """
    Detect both supported layouts.

    New layout:  data/parsed_pdf_output/<pdf-stem>/
    Old layout:  data/parsed_pdf_output/
    """
    candidates = [PARSED_ROOT / pdf_path.stem, PARSED_ROOT]

    for output_dir in candidates:
        if not _parsed_artifacts_complete(output_dir):
            continue

        manifest = _read_parse_manifest(output_dir)

        if manifest:
            stored_hash = str(manifest.get("sha256", ""))

            if stored_hash == fingerprint:
                return output_dir, "sha256"

            # A manifest exists and says this is a different file/version.
            continue

        if output_dir != PARSED_ROOT:
            if output_dir.name.lower() == pdf_path.stem.lower():
                return output_dir, "legacy-folder"

        if _legacy_output_mentions_pdf(output_dir, pdf_path.name):
            return output_dir, "legacy-filename"

    return None, None


def _load_parsed_records(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}

    for key, filename in (
        ("pages", "page_records.json"),
        ("images", "image_records.json"),
        ("tables", "table_records.json"),
    ):
        with (output_dir / filename).open("r", encoding="utf-8") as file:
            records[key] = json.load(file)

    return records


def _qdrant_document_filter(*, field: str, value: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))]
    )


def _qdrant_has_points(
    client: QdrantClient,
    collection_name: str,
    query_filter: models.Filter,
) -> bool:
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=query_filter,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    return bool(points)


class IngestionService:
    """
    Ingestion Service.

    Owns document intake, PDF parsing, and Qdrant ingestion. Reuses existing
    parsed artifacts / Qdrant indexes whenever possible so the API layer never
    needs to know about the underlying pipeline modules.
    """

    def discover_pdfs(self) -> list[dict[str, Any]]:
        """Find PDFs already stored under data/ and data/uploads/."""
        discovered: dict[str, Path] = {}

        for folder in (DATA_DIR, UPLOAD_DIR):
            if not folder.exists():
                continue

            for path in folder.glob("*.pdf"):
                try:
                    key = str(path.resolve()).lower()
                except OSError:
                    key = str(path).lower()

                discovered[key] = path

        results: list[dict[str, Any]] = []

        for path in sorted(discovered.values(), key=lambda p: p.name.lower()):
            try:
                fingerprint = _fingerprint_file(path)
            except OSError as exc:
                logger.warning("pdf_fingerprint_failed", path=str(path), error=str(exc))
                continue

            results.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "fingerprint": fingerprint,
                    "size_bytes": path.stat().st_size,
                }
            )

        return results

    def save_upload(self, filename: str, data: bytes) -> dict[str, Any]:
        """Persist uploaded PDF bytes and return its identity."""
        try:
            safe_name = Path(filename).name
            fingerprint = _fingerprint_bytes(data)
            destination = UPLOAD_DIR / safe_name

            if not destination.exists() or _fingerprint_file(destination) != fingerprint:
                destination.write_bytes(data)

            return {
                "filename": safe_name,
                "path": str(destination),
                "fingerprint": fingerprint,
                "size_bytes": len(data),
            }
        except Exception as exc:
            logger.exception("pdf_upload_failed", filename=filename, error=str(exc))
            raise DocumentPortalException("Failed to store uploaded PDF", exc) from exc

    def parse_document(
        self,
        pdf_path: str,
        fingerprint: str,
        force: bool = False,
        tesseract_path: str | None = None,
    ) -> dict[str, Any]:
        """Parse a PDF, or reuse existing parsed artifacts when available."""
        try:
            path = Path(pdf_path)

            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {path}")

            output_dir = PARSED_ROOT / path.stem

            if not force:
                existing_dir, mode = _find_existing_parsed_output(path, fingerprint)

                if existing_dir is not None:
                    records = _load_parsed_records(existing_dir)

                    return {
                        "output_dir": str(existing_dir),
                        "restored": True,
                        "restore_mode": mode,
                        "pages": records["pages"],
                        "images": records["images"],
                        "tables": records["tables"],
                        "documents_count": len(records["pages"])
                        + len(records["images"])
                        + len(records["tables"]),
                    }

            if output_dir.exists():
                shutil.rmtree(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            parser = ComplexPDFParser(
                pdf_path=str(path),
                output_dir=str(output_dir),
                tesseract_path=tesseract_path or os.getenv("TESSERACT_PATH") or None,
            )

            parsed = parser.parse(save_output=True)
            _write_parse_manifest(output_dir=output_dir, pdf_path=path, fingerprint=fingerprint)

            return {
                "output_dir": str(output_dir),
                "restored": False,
                "restore_mode": None,
                "pages": parsed["pages"],
                "images": parsed["images"],
                "tables": parsed["tables"],
                "documents_count": len(parsed["documents"]),
            }

        except DocumentPortalException:
            raise
        except Exception as exc:
            logger.exception("pdf_parse_failed", pdf_path=pdf_path, error=str(exc))
            raise DocumentPortalException("Failed to parse PDF", exc) from exc

    def check_index_status(
        self,
        *,
        collection_name: str,
        filename: str,
        fingerprint: str,
    ) -> dict[str, Any] | None:
        """
        Check whether this PDF has already been indexed, without ingesting.

        New ingestions store metadata.file_sha256, so future runs can match the
        exact PDF bytes. Older terminal-created points fall back to
        metadata.filename for backward compatibility.
        """
        qdrant_url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_Cluster_Endpoint")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            return None

        client = _get_qdrant_client(qdrant_url, qdrant_api_key)

        if not client.collection_exists(collection_name):
            return None

        exact_filter = _qdrant_document_filter(field="metadata.file_sha256", value=fingerprint)

        try:
            exact_hash_match = _qdrant_has_points(client, collection_name, exact_filter)
        except Exception:
            # Older collections may not have a payload index for file_sha256.
            exact_hash_match = False

        if exact_hash_match:
            info = client.get_collection(collection_name)
            return {
                "collection_name": collection_name,
                "loaded_existing": True,
                "match_mode": "sha256",
                "collection_points": getattr(info, "points_count", None),
            }

        legacy_filter = _qdrant_document_filter(field="metadata.filename", value=filename)

        if _qdrant_has_points(client, collection_name, legacy_filter):
            info = client.get_collection(collection_name)
            return {
                "collection_name": collection_name,
                "loaded_existing": True,
                "match_mode": "legacy-filename",
                "collection_points": getattr(info, "points_count", None),
            }

        return None

    def ingest_document(
        self,
        *,
        pdf_path: str,
        fingerprint: str,
        collection_name: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 120,
        replace_existing: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Ingest a parsed PDF into Qdrant, or reuse an existing index."""
        try:
            path = Path(pdf_path)

            if not force:
                existing = self.check_index_status(
                    collection_name=collection_name,
                    filename=path.name,
                    fingerprint=fingerprint,
                )

                if existing is not None:
                    existing["file_sha256"] = fingerprint
                    return existing

            output_dir, _ = _find_existing_parsed_output(path, fingerprint)

            if output_dir is None:
                raise ValueError(
                    "No parsed artifacts found for this PDF. Parse it before ingesting."
                )

            records = _load_parsed_records(output_dir)

            parser = ComplexPDFParser(
                pdf_path=str(path),
                output_dir=str(output_dir),
                tesseract_path=os.getenv("TESSERACT_PATH") or None,
            )
            parser.page_records = records["pages"]
            parser.image_records = records["images"]
            parser.table_records = records["tables"]
            documents = parser.create_langchain_documents()

            for document in documents:
                document.metadata["file_sha256"] = fingerprint

            ingestion = MultimodalDocumentIngestion(
                collection_name=collection_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            result = ingestion.ingest_documents(
                documents=documents,
                replace_existing=replace_existing,
            )
            result["file_sha256"] = fingerprint
            result["loaded_existing"] = False

            return result

        except DocumentPortalException:
            raise
        except Exception as exc:
            logger.exception("pdf_ingestion_failed", pdf_path=pdf_path, error=str(exc))
            raise DocumentPortalException("Failed to ingest PDF into Qdrant", exc) from exc

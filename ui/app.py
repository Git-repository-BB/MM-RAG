# -*- coding: utf-8 -*-
"""
Resume-aware ChatGPT-style Streamlit UI for the MM-RAG project.

This UI is a thin, admin/demo-oriented client. It never touches the parsing,
ingestion, retrieval, or generation pipelines directly. Every pipeline action
goes through the MM-RAG API layer (see `api/main.py`):

    Streamlit UI -> API layer -> Ingestion / Retrieval / Generation services

A dedicated public frontend (e.g. React/Next.js) can talk to the same API.

Behavior:
1. Select an existing project PDF OR upload a new PDF.
2. If parsed artifacts already exist on disk, load them instead of parsing again.
3. If Qdrant already contains the document, reuse the existing index.
4. Only run missing stages.
5. Chat through the API's /api/chat/ask endpoint.

Expected location:
    <project_root>/ui/app.py

Configuration:
    MMRAG_API_BASE_URL - base URL of the running MM-RAG API (default
    http://127.0.0.1:8000). Start the API with:
        uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import api_client
from api_client import ApiClientError
from prompt_library.prompt import CHAT_SUGGESTED_PROMPTS


# ---------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="MM-RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        max-width: 1120px;
        padding-top: 1.6rem;
        padding-bottom: 7rem;
    }

    [data-testid="stSidebar"] {
        min-width: 330px;
        max-width: 380px;
    }

    .mmrag-title {
        font-size: 2.2rem;
        font-weight: 750;
        letter-spacing: -0.04em;
        margin-bottom: 0.15rem;
    }

    .mmrag-subtitle {
        opacity: 0.72;
        margin-bottom: 1.2rem;
    }

    .source-card {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 12px;
        padding: 10px 12px;
        margin: 6px 0;
    }

    .tiny-muted {
        opacity: 0.65;
        font-size: 0.82rem;
    }

    .ready-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.25);
        margin-right: 6px;
        margin-bottom: 8px;
        font-size: .83rem;
    }

    .resume-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: 10px 12px;
        margin: 8px 0;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 14px;
        padding: 10px 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

def init_session_state() -> None:
    defaults = {
        "document_token": None,
        "selected_pdf_path": None,
        "selected_pdf_fingerprint": None,
        "last_uploaded_key": None,
        "parsed_result": None,
        "parsed_file_fingerprint": None,
        "ingestion_result": None,
        "active_collection": None,
        "index_restore_mode": None,
        "index_check_token": None,
        "chat_messages": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_document_state() -> None:
    """Clear all state tied to the previously selected document."""
    st.session_state.parsed_result = None
    st.session_state.parsed_file_fingerprint = None
    st.session_state.ingestion_result = None
    st.session_state.active_collection = None
    st.session_state.index_restore_mode = None
    st.session_state.index_check_token = None
    st.session_state.chat_messages = []


def clear_chat() -> None:
    st.session_state.chat_messages = []


init_session_state()


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def safe_filename(filename: str) -> str:
    return Path(filename).name


def deepest_error_message(exc: BaseException) -> str:
    if isinstance(exc, ApiClientError):
        return str(exc)

    current: BaseException = exc
    visited: set[int] = set()

    while current.__cause__ is not None and id(current) not in visited:
        visited.add(id(current))
        current = current.__cause__

    return f"{type(current).__name__}: {current}"


def current_filename() -> str | None:
    path = st.session_state.selected_pdf_path

    if not path:
        return None

    return Path(path).name


def is_chat_ready() -> bool:
    return bool(
        st.session_state.ingestion_result
        and st.session_state.active_collection
    )


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------

@st.cache_data(ttl=15, show_spinner=False)
def get_health_cached() -> dict[str, Any] | None:
    try:
        return api_client.get_health()
    except Exception:
        return None


def render_config_status(health: dict[str, Any] | None) -> None:
    with st.sidebar.expander("⚙️ Environment", expanded=False):
        if health is None:
            st.error("MM-RAG API is unreachable.")
            st.caption(api_client.API_BASE_URL)
            return

        st.write("✅ API reachable")
        st.caption(api_client.API_BASE_URL)
        st.write(
            f"{'✅' if health.get('openai_api_key_configured') else '❌'} OpenAI API key"
        )
        st.write(
            f"{'✅' if health.get('qdrant_url_configured') else '❌'} Qdrant URL"
        )
        st.write(
            f"{'✅' if health.get('qdrant_api_key_configured') else '⚠️'} Qdrant API key"
        )
        st.write(
            f"{'✅' if health.get('tesseract_path_configured') else 'ℹ️'} Tesseract path"
        )


def render_pipeline_badges() -> None:
    parsed = st.session_state.parsed_result is not None
    indexed = st.session_state.ingestion_result is not None
    ready = is_chat_ready()

    st.markdown(
        (
            f'<span class="ready-pill">'
            f'{"✅" if parsed else "○"} Parsed'
            f'</span>'
            f'<span class="ready-pill">'
            f'{"✅" if indexed else "○"} Indexed'
            f'</span>'
            f'<span class="ready-pill">'
            f'{"✅" if ready else "○"} Chat ready'
            f'</span>'
        ),
        unsafe_allow_html=True,
    )


def render_parse_summary(parsed: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Pages", len(parsed.get("pages", [])))
    col2.metric("Tables", len(parsed.get("tables", [])))
    col3.metric("Images", len(parsed.get("images", [])))
    col4.metric("Documents", parsed.get("documents_count", 0))


def render_document_inspector(parsed: dict[str, Any]) -> None:
    with st.expander("🔎 Document inspector", expanded=False):
        text_tab, table_tab, image_tab = st.tabs(
            ["Text / OCR", "Tables", "Images"]
        )

        with text_tab:
            pages = parsed.get("pages", [])

            if not pages:
                st.info("No page text was produced.")
            else:
                page_options = [
                    int(page.get("page_number", index + 1))
                    for index, page in enumerate(pages)
                ]

                selected_page = st.selectbox(
                    "Preview page",
                    options=page_options,
                    key="preview_page",
                )

                selected_page_record = next(
                    (
                        page
                        for page in pages
                        if int(page.get("page_number", -1)) == selected_page
                    ),
                    None,
                )

                if selected_page_record:
                    st.markdown("**Selectable text**")
                    st.text(
                        str(selected_page_record.get("text", "")).strip()[:12000]
                        or "(no selectable text)"
                    )
                    st.markdown("**OCR text**")
                    st.text(
                        str(selected_page_record.get("ocr_text", "")).strip()[:12000]
                        or "(no OCR text)"
                    )

        with table_tab:
            tables = parsed.get("tables", [])

            if not tables:
                st.info("No tables detected.")
            else:
                labels = [
                    (
                        f"Page {table.get('page_number', '?')}"
                        f" · Table {table.get('table_index', '?')}"
                    )
                    for table in tables
                ]

                index = st.selectbox(
                    "Preview table",
                    options=range(len(tables)),
                    format_func=lambda i: labels[i],
                    key="preview_table",
                )

                table = tables[index]

                if table.get("markdown"):
                    st.markdown(table["markdown"])
                else:
                    st.write(table.get("raw_table"))

        with image_tab:
            images = parsed.get("images", [])

            if not images:
                st.info("No embedded images detected.")
            else:
                labels = [
                    (
                        f"Page {image.get('page_number', '?')}"
                        f" · Image {image.get('image_index', '?')}"
                    )
                    for image in images
                ]

                index = st.selectbox(
                    "Preview image",
                    options=range(len(images)),
                    format_func=lambda i: labels[i],
                    key="preview_image",
                )

                image_record = images[index]
                # Streamlit and the API are assumed to share a filesystem in
                # this local/demo deployment, so image paths returned by the
                # API can be rendered directly.
                image_path = Path(str(image_record.get("image_path", "")))

                if image_path.exists():
                    st.image(
                        str(image_path),
                        caption=labels[index],
                        width="stretch",
                    )

                    ocr_text = str(image_record.get("image_ocr_text", "")).strip()

                    if ocr_text:
                        with st.expander("OCR text"):
                            st.text(ocr_text[:6000])
                else:
                    st.warning(f"Image file not found: {image_path}")


def render_index_status() -> None:
    result = st.session_state.ingestion_result or {}

    if result.get("loaded_existing"):
        mode = result.get("match_mode")

        st.success("Existing Qdrant index loaded — no re-ingestion was performed.")

        if mode == "sha256":
            st.caption("Matched using exact PDF SHA-256.")
        else:
            st.caption(
                "Matched using filename because this index was created by the "
                "older terminal pipeline before SHA metadata was stored."
            )

        points = result.get("collection_points")

        if points is not None:
            st.metric("Collection points", points)

        st.caption(f"Collection: {result.get('collection_name')}")

    else:
        st.success("Document indexed successfully.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Parsed docs", result.get("input_document_count", 0))
        col2.metric("Indexed chunks", result.get("indexed_document_count", 0))
        col3.metric("Qdrant points", result.get("inserted_point_count", 0))


def render_sidebar_document_workspace(selected_pdf_path: str | None) -> None:
    """
    Keep document/pipeline details out of the main chat canvas.

    Everything related to parsing, restored artifacts, document statistics,
    inspection, and Qdrant index state is shown in the sidebar.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Document status")

    with st.sidebar:
        render_pipeline_badges()

    if selected_pdf_path is None:
        st.sidebar.caption("No document selected.")
        return

    st.sidebar.caption(Path(selected_pdf_path).name)

    with st.sidebar.expander("📄 Document details", expanded=False):
        st.caption(selected_pdf_path)

        parsed = st.session_state.parsed_result

        if parsed:
            row1_col1, row1_col2 = st.columns(2)
            row2_col1, row2_col2 = st.columns(2)

            row1_col1.metric("Pages", len(parsed.get("pages", [])))
            row1_col2.metric("Tables", len(parsed.get("tables", [])))
            row2_col1.metric("Images", len(parsed.get("images", [])))
            row2_col2.metric("Documents", parsed.get("documents_count", 0))

            if parsed.get("restored"):
                st.success(
                    "Existing parsed data restored. "
                    "OCR/table/image extraction was skipped."
                )
            else:
                st.success("PDF parsed in this app session.")
        else:
            st.info("No reusable parsed data found yet.")

    if st.session_state.parsed_result:
        with st.sidebar:
            render_document_inspector(st.session_state.parsed_result)

    with st.sidebar.expander("🗄️ Qdrant index status", expanded=False):
        result = st.session_state.ingestion_result or {}

        if not result:
            st.info("No reusable Qdrant index found yet.")
            return

        collection = (
            result.get("collection_name")
            or st.session_state.active_collection
            or "unknown"
        )

        if result.get("loaded_existing"):
            mode = result.get("match_mode")

            st.success("Existing Qdrant index loaded. Embeddings were not created again.")

            if mode == "sha256":
                st.caption("Matched using exact PDF SHA-256.")
            else:
                st.caption("Matched using filename from the older terminal-created index.")

            points = result.get("collection_points")

            if points is not None:
                st.metric("Collection points", points)

        else:
            st.success("Document indexed in this app session.")

            col1, col2 = st.columns(2)
            col1.metric("Chunks", result.get("indexed_document_count", 0))
            col2.metric("Points", result.get("inserted_point_count", 0))

        st.caption(f"Collection: {collection}")


def render_source_cards(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"Sources · {len(sources)} retrieved", expanded=False):
        for source in sources:
            citation = source.get("citation", "Unknown source")
            content_type = source.get("content_type", "unknown")
            score = source.get("score")

            score_text = (
                f"{float(score):.4f}" if isinstance(score, (float, int)) else "n/a"
            )

            st.markdown(
                f"""
<div class="source-card">
    <strong>{citation}</strong><br/>
    <span class="tiny-muted">
        type: {content_type} · similarity: {score_text}
    </span>
</div>
""",
                unsafe_allow_html=True,
            )


def render_used_images(images: list[dict[str, Any]]) -> None:
    # Assumes the Streamlit process and the API share a filesystem (local/demo
    # deployment). A public frontend would instead fetch images over HTTP.
    valid_images = [
        image
        for image in images
        if image.get("image_path") and Path(str(image["image_path"])).exists()
    ]

    if not valid_images:
        return

    with st.expander(f"Retrieved visual evidence · {len(valid_images)}", expanded=False):
        columns = st.columns(min(3, len(valid_images)))

        for index, image in enumerate(valid_images):
            with columns[index % len(columns)]:
                st.image(
                    str(image["image_path"]),
                    caption=image.get("citation", "Retrieved image"),
                    width="stretch",
                )


def render_assistant_message(message: dict[str, Any]) -> None:
    st.markdown(message.get("content", ""))

    render_source_cards(message.get("sources", []))
    render_used_images(message.get("used_images", []))

    metadata = message.get("metadata", {})
    details: list[str] = []

    if metadata.get("model_name"):
        details.append(str(metadata["model_name"]))

    if metadata.get("retrieval_count") is not None:
        details.append(f"{metadata['retrieval_count']} retrieved")

    if details:
        st.caption(" · ".join(details))


def render_chat_history() -> None:
    for message in st.session_state.chat_messages:
        role = message.get("role", "assistant")
        avatar = "🧑‍💻" if role == "user" else "🧠"

        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message.get("content", ""))


# ---------------------------------------------------------------------
# Sidebar: select document
# ---------------------------------------------------------------------

st.sidebar.markdown("## 🧠 MM-RAG")
st.sidebar.caption("Resume-aware document workspace")
api_health = get_health_cached()
render_config_status(api_health)

try:
    existing_pdfs = api_client.list_documents()
except ApiClientError as exc:
    st.sidebar.error("Could not list existing documents from the API.")
    st.sidebar.code(str(exc))
    existing_pdfs = []

source_options = ["Upload new PDF"]

if existing_pdfs:
    source_options.insert(0, "Use existing project PDF")

source_mode = st.sidebar.radio("Document source", options=source_options)

selected_pdf_path: str | None = None
selected_fingerprint: str | None = None

if source_mode == "Use existing project PDF":
    selected_index = st.sidebar.selectbox(
        "Existing PDF",
        options=range(len(existing_pdfs)),
        format_func=lambda i: existing_pdfs[i]["filename"],
    )

    selected_document = existing_pdfs[selected_index]
    selected_pdf_path = selected_document["path"]
    selected_fingerprint = selected_document["fingerprint"]

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        upload_key = (uploaded_file.name, uploaded_file.size)

        if st.session_state.last_uploaded_key != upload_key:
            try:
                uploaded_bytes = uploaded_file.getvalue()
                upload_result = api_client.upload_document(
                    filename=safe_filename(uploaded_file.name),
                    data=uploaded_bytes,
                )
                st.session_state.last_uploaded_key = upload_key
                st.session_state.last_upload_result = upload_result
            except ApiClientError as exc:
                st.sidebar.error("Upload failed.")
                st.sidebar.code(str(exc))

        upload_result = st.session_state.get("last_upload_result")

        if upload_result:
            selected_pdf_path = upload_result["path"]
            selected_fingerprint = upload_result["fingerprint"]

            st.sidebar.caption(
                f"{upload_result['filename']} · "
                f"{upload_result['size_bytes'] / (1024 * 1024):.2f} MB"
            )


# ---------------------------------------------------------------------
# Document changed -> restore local parsed artifacts
# ---------------------------------------------------------------------

if selected_pdf_path is not None and selected_fingerprint is not None:
    document_token = f"{selected_pdf_path}|{selected_fingerprint}"

    if st.session_state.document_token != document_token:
        reset_document_state()

        st.session_state.document_token = document_token
        st.session_state.selected_pdf_path = selected_pdf_path
        st.session_state.selected_pdf_fingerprint = selected_fingerprint

        try:
            parsed = api_client.parse_document(
                pdf_path=selected_pdf_path,
                fingerprint=selected_fingerprint,
                force=False,
            )

            if parsed.get("restored"):
                st.session_state.parsed_result = parsed
                st.session_state.parsed_file_fingerprint = selected_fingerprint
        except ApiClientError:
            # No reusable parsed artifacts yet; the user can trigger parsing.
            pass

    else:
        st.session_state.selected_pdf_path = selected_pdf_path
        st.session_state.selected_pdf_fingerprint = selected_fingerprint

else:
    if st.session_state.document_token is not None:
        st.session_state.document_token = None
        reset_document_state()


# ---------------------------------------------------------------------
# Sidebar: parsing
# ---------------------------------------------------------------------

if selected_pdf_path is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Document pipeline")

    if st.session_state.parsed_result is not None:
        if st.session_state.parsed_result.get("restored"):
            st.sidebar.success("1 · Existing parsed data loaded")
            st.sidebar.caption("No OCR/parsing rerun.")
        else:
            st.sidebar.success("1 · PDF parsed")

        force_parse = st.sidebar.button("Re-parse PDF", use_container_width=True)
    else:
        force_parse = st.sidebar.button(
            "1 · Parse PDF",
            type="primary",
            use_container_width=True,
        )

    if force_parse:
        try:
            with st.status("Parsing PDF...", expanded=True) as status:
                parsed = api_client.parse_document(
                    pdf_path=selected_pdf_path,
                    fingerprint=selected_fingerprint,
                    force=True,
                )

                st.session_state.parsed_result = parsed
                st.session_state.parsed_file_fingerprint = selected_fingerprint

                # Parsed bytes have changed/rebuilt, so re-check the index.
                st.session_state.ingestion_result = None
                st.session_state.active_collection = None
                st.session_state.index_restore_mode = None
                st.session_state.index_check_token = None
                st.session_state.chat_messages = []

                status.update(label="PDF parsed", state="complete", expanded=False)

            st.rerun()

        except ApiClientError as exc:
            st.sidebar.error("PDF parsing failed.")
            st.sidebar.code(str(exc))


# ---------------------------------------------------------------------
# Sidebar: collection + existing-index restore
# ---------------------------------------------------------------------

default_collection = (
    (api_health or {}).get("default_collection_name") or "mm-rag-documents"
)

collection_name = default_collection

if selected_pdf_path is not None:
    collection_name = st.sidebar.text_input(
        "Qdrant collection",
        value=default_collection,
        key="collection_name_input",
    ).strip()

    index_check_token = f"{st.session_state.document_token}|{collection_name}"

    if collection_name and st.session_state.index_check_token != index_check_token:
        # Changing collection must invalidate the previously loaded index state.
        st.session_state.ingestion_result = None
        st.session_state.active_collection = None
        st.session_state.index_restore_mode = None

        try:
            status = api_client.index_status(
                collection_name=collection_name,
                filename=Path(selected_pdf_path).name,
                fingerprint=selected_fingerprint,
            )

            if status.get("found"):
                st.session_state.ingestion_result = {
                    "collection_name": status.get("collection_name"),
                    "loaded_existing": True,
                    "match_mode": status.get("match_mode"),
                    "collection_points": status.get("collection_points"),
                }
                st.session_state.active_collection = collection_name
                st.session_state.index_restore_mode = status.get("match_mode")

        except ApiClientError as exc:
            # API/Qdrant being temporarily unavailable should not prevent
            # local parse inspection.
            st.sidebar.warning("Could not check the existing Qdrant index.")
            st.sidebar.code(str(exc))

        st.session_state.index_check_token = index_check_token


# ---------------------------------------------------------------------
# Sidebar: ingestion only when index missing
# ---------------------------------------------------------------------

if (
    selected_pdf_path is not None
    and st.session_state.parsed_result is not None
):
    if st.session_state.ingestion_result is not None:
        if st.session_state.index_restore_mode is not None:
            st.sidebar.success("2 · Existing Qdrant index loaded")
            st.sidebar.caption("No embeddings/re-ingestion rerun.")
        else:
            st.sidebar.success("2 · Indexed in Qdrant")

        if st.sidebar.button("Re-index document", use_container_width=True):
            st.session_state.ingestion_result = None
            st.session_state.active_collection = None
            st.session_state.index_restore_mode = None
            st.session_state.chat_messages = []
            st.rerun()

    else:
        with st.sidebar.expander("Advanced indexing", expanded=False):
            chunk_size = st.number_input(
                "Chunk size",
                min_value=200,
                max_value=10000,
                value=2000,
                step=100,
                key="chunk_size_input",
            )

            chunk_overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=2000,
                value=120,
                step=20,
                key="chunk_overlap_input",
            )

            replace_existing = st.checkbox(
                "Replace existing document points",
                value=True,
                key="replace_existing_input",
            )

        invalid_chunking = chunk_overlap >= chunk_size

        if invalid_chunking:
            st.sidebar.error("Chunk overlap must be smaller than chunk size.")

        if st.sidebar.button(
            "2 · Ingest to Qdrant",
            type="primary",
            use_container_width=True,
            disabled=invalid_chunking or not collection_name,
        ):
            try:
                with st.status("Indexing document...", expanded=True) as status:
                    started_at = time.monotonic()

                    def _on_poll(job_status: dict[str, Any], _status=status, _started_at=started_at) -> None:
                        elapsed = int(time.monotonic() - _started_at)
                        _status.update(label=f"Indexing document... ({elapsed}s elapsed)")
                        _status.write(f"Job status: {job_status['status']} · {elapsed}s elapsed")

                    result = api_client.ingest_document(
                        pdf_path=selected_pdf_path,
                        fingerprint=selected_fingerprint,
                        collection_name=collection_name,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        replace_existing=replace_existing,
                        force=True,
                        on_poll=_on_poll,
                    )

                    st.session_state.ingestion_result = result
                    st.session_state.active_collection = collection_name
                    st.session_state.index_restore_mode = None
                    st.session_state.chat_messages = []

                    status.update(
                        label="Qdrant indexing complete",
                        state="complete",
                        expanded=False,
                    )

                st.rerun()

            except ApiClientError as exc:
                st.sidebar.error("Qdrant ingestion failed.")
                st.sidebar.code(str(exc))


# ---------------------------------------------------------------------
# Sidebar: chat settings
# ---------------------------------------------------------------------

if is_chat_ready():
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Chat settings")

    chat_model = st.sidebar.text_input(
        "Model",
        value=(api_health or {}).get("default_chat_model") or "gpt-4.1-mini",
        key="chat_model_input",
    )

    top_k = st.sidebar.slider(
        "Top-K retrieval",
        min_value=2,
        max_value=15,
        value=6,
        key="top_k_input",
    )

    max_images = st.sidebar.slider(
        "Max retrieved images",
        min_value=0,
        max_value=8,
        value=4,
        key="max_images_input",
    )

    current_pdf_only = st.sidebar.checkbox(
        "Search only current PDF",
        value=True,
        key="current_pdf_only_input",
    )

    if st.sidebar.button("Clear chat", use_container_width=True):
        clear_chat()
        st.rerun()


# ---------------------------------------------------------------------
# Sidebar: document / parsing / index information
# ---------------------------------------------------------------------

render_sidebar_document_workspace(selected_pdf_path)


# ---------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------

st.markdown(
    '<div class="mmrag-title">Multimodal Document Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    (
        '<div class="mmrag-subtitle">'
        "Ask grounded questions across text, OCR, tables, "
        "and retrieved visual evidence."
        "</div>"
    ),
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Main chat readiness
# ---------------------------------------------------------------------

if selected_pdf_path is None:
    st.markdown("### Start a document conversation")
    st.caption("Choose an existing PDF or upload a new one from the sidebar.")
    st.stop()

if st.session_state.parsed_result is None:
    st.markdown("### Preparing your document")
    st.caption(
        "No reusable parsed artifacts were found. Use **Parse PDF** in the sidebar."
    )
    st.stop()

if st.session_state.ingestion_result is None:
    st.markdown("### Almost ready")
    st.caption(
        "The PDF is parsed. Use **Ingest to Qdrant** in the sidebar "
        "to enable document chat."
    )
    st.stop()


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

if not st.session_state.chat_messages:
    st.markdown("### What would you like to know?")
    st.caption("Ask about policies, contracts, incidents, tables, charts, or diagrams.")

    suggestions = CHAT_SUGGESTED_PROMPTS

    columns = st.columns(2)
    suggested_prompt: str | None = None

    for index, suggestion in enumerate(suggestions):
        with columns[index % 2]:
            if st.button(
                suggestion,
                key=f"suggestion_{index}",
                use_container_width=True,
            ):
                suggested_prompt = suggestion
else:
    suggested_prompt = None


render_chat_history()

typed_prompt = st.chat_input(
    "Ask anything about the document...",
    key="document_chat_input",
)

prompt = suggested_prompt or typed_prompt

if prompt:
    user_message = {"role": "user", "content": str(prompt)}
    st.session_state.chat_messages.append(user_message)

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(str(prompt))

    with st.chat_message("assistant", avatar="🧠"):
        try:
            with st.status("Searching the document...", expanded=False) as status:
                filename_filter = current_filename() if current_pdf_only else None

                result = api_client.ask(
                    query=str(prompt),
                    collection_name=st.session_state.active_collection,
                    model_name=chat_model.strip(),
                    k=int(top_k),
                    max_images=int(max_images),
                    filename=filename_filter,
                )

                status.update(label="Answer ready", state="complete", expanded=False)

            assistant_message = {
                "role": "assistant",
                "content": result.get("answer", "No answer was generated."),
                "sources": result.get("sources", []),
                "used_images": result.get("used_images", []),
                "metadata": {
                    "model_name": result.get("model_name"),
                    "retrieval_count": result.get("retrieval_count"),
                    "usage": result.get("usage"),
                },
            }

            st.session_state.chat_messages.append(assistant_message)
            render_assistant_message(assistant_message)

        except ApiClientError as exc:
            error_text = str(exc)

            st.error("I could not generate an answer.")
            st.code(error_text)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": f"The RAG pipeline returned an error: `{error_text}`",
                    "sources": [],
                    "used_images": [],
                    "metadata": {},
                }
            )

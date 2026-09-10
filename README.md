# Multimodal RAG Full-Stack GenAI Bootcamp

A full-stack Generative AI project for learning how to build a multimodal
Retrieval-Augmented Generation (RAG) application. The project will demonstrate
how documents and other supported content can be ingested, processed, indexed,
retrieved, and supplied to a large language model to produce grounded answers.

> [!NOTE]
> The ingestion, retrieval, and generation pipeline is implemented behind a
> FastAPI service layer. Streamlit is an admin/demo frontend that talks to
> that API instead of calling the pipeline directly — see [Architecture](#architecture).

## Project goals

- Build an end-to-end RAG workflow.
- Process and retrieve information from multimodal source content.
- Generate responses grounded in retrieved context.
- Connect the AI pipeline to a user-facing application.
- Organize the backend, frontend, configuration, and supporting services as a
  maintainable full-stack project.

## Expected RAG workflow

1. Load source documents or other supported content.
2. Extract, clean, and divide the content into useful chunks.
3. Create embeddings that represent those chunks.
4. Store the embeddings and metadata in a vector store.
5. Retrieve the most relevant context for a user's question.
6. Send the question and retrieved context to a language model.
7. Return a grounded answer through the application interface.

## Architecture

```text
Frontend (Streamlit admin/demo UI, or a public React/Next.js app)
        |
        v
   API layer (FastAPI, api/main.py)
        |
        +--> Ingestion Service   (src/services/ingestion_service.py)
        +--> Retrieval Service   (src/services/retrieval_service.py)
        +--> Generation Service  (src/services/generation_service.py)
```

- The **API layer** (`api/`) exposes REST endpoints for document discovery,
  upload, parsing, Qdrant ingestion, retrieval, and chat generation.
- The **services** (`src/services/`) contain the orchestration logic (reuse
  parsed artifacts, reuse existing Qdrant indexes, run only missing stages)
  and wrap the lower-level pipeline modules in `src/` (`parsing.py`,
  `ingestion.py`, `retriever.py`, `generation.py`).
- The **Streamlit app** (`ui/app.py`) is a thin client: it never imports the
  pipeline modules directly, it only calls the API through `ui/api_client.py`.
  Any other frontend (e.g. React/Next.js) can call the same API.

### Asynchronous ingestion

`POST /api/documents/ingest` does not block until embeddings/upsert finish.
It creates a job, schedules the actual ingestion work as a FastAPI
`BackgroundTask` (run on a worker thread via `api/jobs.py`), and immediately
returns `202 Accepted` with a `job_id`:

```json
{ "job_id": "b3f6...", "status": "pending" }
```

Callers poll `GET /api/documents/ingest/{job_id}` until `status` becomes
`"completed"` (with a `result`) or `"failed"` (with an `error`). The
Streamlit UI does this polling for you (`ui/api_client.py:ingest_document`)
and updates the sidebar status text every ~1.5s while indexing runs.

The job store is an in-memory, per-process registry — fine for a single
local/demo API instance. For multi-worker or multi-process deployments,
swap it for a shared store (e.g. Redis) so every worker can see every job.

## Prerequisites

Before setting up the project, install:

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (it can also be installed and managed through `uv`)
- Git

## Python version

This project uses **Python 3.12**.

List the Python versions available to `uv`:

```cmd
uv python list
```

If Python 3.12 is not available, install it with:

```cmd
uv python install 3.12
```

## Setup

### 1. Clone the repository

```cmd
git clone <repository-url>
cd mm-rag-full-stack-genai-bootcamp-1.0
```

Replace `<repository-url>` with the URL of this repository.

### 2. Create the virtual environment

Create a virtual environment named `env` using Python 3.12:

```cmd
uv venv env --python 3.12
```

The general form of the command is:

```cmd
uv venv env --python <python-version>
```

### 3. Activate the virtual environment

On Windows Command Prompt:

```cmd
env\Scripts\activate.bat
```

On macOS or Linux:

```bash
source env/bin/activate
```

After activation, confirm the selected Python version:

```cmd
python --version
```

The output should report Python 3.12.x.

### 4. Install dependencies

Install the dependencies listed in `requirements.txt`:

```cmd
uv pip install -r requirements.txt
```

### 5. Configure environment variables

Store local configuration, API keys, model settings, and service credentials in
the `.env` file. For example:

```dotenv
# Add only the variables required by the implemented services.
# LLM_API_KEY=your_api_key
# MODEL_NAME=your_model_name
```

Never commit real secrets or API keys. The `.env` file and `env` virtual
environment directory are excluded through `.gitignore`.

## Running the application

The API must be running before the Streamlit UI can parse, ingest, or chat.

### 1. Start the API layer

```cmd
uvicorn api.main:app --reload
```

The API is served at `http://127.0.0.1:8000` by default (interactive docs at
`http://127.0.0.1:8000/docs`).

### 2. Start the Streamlit UI

In a second terminal (with the same virtual environment activated):

```cmd
streamlit run ui/app.py
```

If the API is not running on the default host/port, point the UI at it:

```cmd
set MMRAG_API_BASE_URL=http://127.0.0.1:8000
```

## Current project structure

```text
MM-RAG/
|-- .env                    # Local environment variables (not committed)
|-- requirements.txt        # Python dependencies
|-- api/                    # FastAPI layer (routers, schemas, DI)
|   |-- main.py
|   |-- schemas.py
|   |-- dependencies.py
|   `-- routers/            # documents, retrieval, chat endpoints
|-- src/
|   |-- parsing.py          # PDF/OCR/table/image extraction
|   |-- ingestion.py        # Chunking, embeddings, Qdrant upsert
|   |-- retriever.py        # Qdrant similarity search
|   |-- generation.py       # Multimodal RAG answer generation
|   `-- services/           # Ingestion / Retrieval / Generation services
|-- ui/
|   |-- app.py              # Streamlit admin/demo frontend
|   `-- api_client.py       # HTTP client used by the Streamlit app
|-- prompt_library/         # Shared prompt templates
|-- exception/              # Shared custom exception type
|-- logger/                 # Structured logging setup
|-- config/                 # Application configuration
`-- data/                   # Uploaded PDFs and parsed artifacts
```

## Common commands

```cmd
# List available Python installations
uv python list

# Create the Python 3.12 virtual environment
uv venv env --python 3.12

# Activate it in Command Prompt
env\Scripts\activate.bat

# Install project dependencies
uv pip install -r requirements.txt

# Leave the virtual environment
deactivate
```

## Development guidelines

- Use Python 3.12 for consistent local development.
- Activate the virtual environment before running Python commands.
- Add new Python dependencies to `requirements.txt`.
- Keep credentials and machine-specific settings in `.env`.
- Add tests as each application component is implemented.
- Update this README whenever setup or run commands change.

## Implemented components

- PDF ingestion and preprocessing (text, OCR, tables, embedded images)
- Chunking and metadata management
- Embedding generation and Qdrant vector storage
- Semantic retrieval with metadata filters
- Multimodal LLM-based response generation
- FastAPI backend exposing ingestion, retrieval, and chat endpoints
- Streamlit admin/demo frontend consuming the API
- Structured application logging and custom exception handling

## Planned enhancements

- Dedicated public frontend (e.g. React/Next.js) against the same API
- Serving retrieved/inspected images over the API instead of shared local disk
- Authentication/authorization on the API layer
- Automated tests and CI
- Evaluation and observability tooling

## Status

Initial project setup is complete. Feature implementation has not yet been added.

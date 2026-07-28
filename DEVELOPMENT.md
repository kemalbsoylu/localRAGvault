# Architecture & Development Guide

This document covers the technical architecture, containerization strategy, native local development setup, engineering standards, and API reference for `localRAGvault`. It is designed for developers, maintainers, and systems engineers.

---

## Tech Stack & Engine Specifications

* **Text Generation & Embedding Models:** Open-weight models via Ollama (Default: `gemma4:latest` / `embeddinggemma:latest`)
* **Vector Database:** PostgreSQL 16.14 + `pgvector` (via `psycopg` 3 binary driver)
* **Backend Framework:** Python 3.12 (managed by `uv`) + FastAPI + Pydantic v2
* **Frontend Interface:** Streamlit
* **Containerization & Deployment:** Multi-target Docker builds, Docker Compose, GitHub Container Registry (GHCR), and Docker Hub (`kemalbsoylu/localragvault`)
* **CI/CD & Quality Assurance:** Automated GitHub Actions (`ci.yml`, `release.yml`), Ruff, Ty, Pytest, Pytest-cov

---

## Project Structure

```text
localRAGvault/
├── .github/
│   └── workflows/
│       ├── ci.yml           # Automated linter, type-check, and pgvector unit testing pipeline
│       └── release.yml      # Multi-arch image build & automated release publishing pipeline
├── core/
│   ├── api.py               # FastAPI application, lifecycle management, and REST endpoints
│   ├── config.py            # Centralized environment configurations and hyperparameter defaults
│   ├── database.py          # PostgreSQL/pgvector connection pooling, schema init, and atomic queries
│   ├── extractors.py        # Format-specific text extractors (PDF, DOCX, CSV, JSON, Markdown)
│   ├── logging_config.py    # Rotating file and terminal telemetry logging setup
│   ├── schemas.py           # Strict Pydantic data validation and request/response models
│   └── utils.py             # Ollama client wrappers, text chunking physics, and LLM execution
├── ui/
│   └── app.py               # Streamlit frontend featuring multi-turn chat, vault search, and settings
├── tests/
│   ├── conftest.py          # Pytest fixtures, mock embeddings, and isolated DB provisioning
│   ├── test_workspaces.py   # Workspace CRUD, hyperparameter patching, and dimension safeguards
│   ├── test_documents.py    # Document ingestion, clean upserts, batch upload telemetry, and vector search
│   ├── test_chat_threads.py # Multi-turn RAG execution, message history, and thread renaming validation
│   ├── test_extractors.py   # Format-specific extraction engines (PDF, DOCX, CSV, JSON, Markdown)
│   └── test_utils.py        # Chunking physics, overlap boundary errors, and model tag normalization
├── uploads/                 # Mirrored physical storage for ingested files (isolated by workspace ID)
├── logs/                    # System diagnostic logs and rotating error telemetry
├── Dockerfile               # Multi-stage Docker build targeting optimized 'api' and 'ui' containers
├── docker-compose.yml       # Orchestration for vector DB, backend API, and UI services
├── .env.docker              # Internal environment configuration for Docker bridge networking
├── .dockerignore            # Build context exclusion rules
├── pyproject.toml           # Project metadata, dependencies, Ruff linting rules, and Pytest markers
├── .env.example             # Native local environment configuration template
└── README.md                # User quickstart and feature overview
```

---

## Native Local Development Setup (Without Docker)

While end-users run the application via Docker Desktop, developers contributing to core logic often prefer running the application natively for faster live-reloading and easier debugging.

### 1. Prerequisites
To run natively, install these core dependencies on your local OS:
* **[Ollama](https://ollama.com/download)** (v0.32.0+) – Ensure it is running locally on port `11434`.
* **[PostgreSQL](https://www.postgresql.org/download/)** (v16.14) with the `pgvector` extension enabled.
* **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (v0.11.32+) – Python package manager.

### 2. Database Provisioning
Run the following commands in your terminal to provision the local database and enable vector search (replace `postgres` with your local PostgreSQL superuser account if different):
```bash
psql -U postgres -c "CREATE DATABASE localragvault;"
psql -U postgres -d localragvault -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Environment Configuration & Dependency Sync
Copy the native environment template and update it with your local PostgreSQL credentials:
```bash
cp .env.example .env
```
*(Note: Once the empty database is created and your `.env` is configured, the application will automatically build all required internal tables and vector columns upon startup!)*

Install project dependencies using `uv`:
```bash
uv sync
```

### 4. Launching the Split Architecture
Open two separate terminal windows inside the project root:

* **Terminal 1 (Start the Backend API):**
  ```bash
  uv run uvicorn core.api:app --reload --reload-dir core
  ```
  *Starts the REST API server at `http://127.0.0.1:8000` and initializes database schemas.*

* **Terminal 2 (Start the Web Dashboard):**
  ```bash
  uv run streamlit run ui/app.py
  ```
  *Opens the interactive UI at `http://localhost:8501`.*

---

## Maintainability & Architecture Decisions

This project is engineered to be readable, defensible, and easily maintainable by future contributors. Below are the intentional design trade-offs made during development:

### 1. Multi-Target Containerization & Network Isolation
To ensure seamless deployment across different platforms without bloating container images, we utilize a 4-stage build process in `Dockerfile` (`base`, `dependencies`, `api`, and `ui`). By copying compiled environments directly from Astral's `uv` images, build times are minimized and production layers remain pristine. 
Inside `docker-compose.yml`, services are isolated within a private bridge network (`vault-net`). The FastAPI server communicates securely with the `pgvector` container via internal DNS (`DB_HOST=db`), while routing LLM requests safely out of the Docker bridge to the host OS Ollama daemon via `host.docker.internal`.

### 2. Strict Domain & Service Boundaries
Rather than writing monolithic route handlers, the application is separated into isolated domain modules:
* `core/extractors.py`: Pure functions dedicated solely to format-specific file parsing and table structural preservation.
* `core/database.py`: Handles connection pooling, vector registration, and atomic SQL execution without leaking database logic to the API.
* `core/utils.py`: Encapsulates Ollama daemon communication, vector embedding batching, and text chunking physics.
* `core/schemas.py`: Single source of truth for Pydantic data contracts.

### 3. Defensive Type Safety & Pydantic Normalization
All API boundaries are strictly governed by Pydantic v2 schemas. Custom field validators (such as enforcing `:latest` tags on models via `normalize_tag`) and model validators (ensuring `chunk_overlap` is strictly less than `chunk_size`) reject malformed payloads at the API edge before they ever touch the database or LLM engine. Python 3.12+ type annotations are used universally and verified statically via `ty check`.

### 4. Vector Pollution Safeguards (Database Resilience)
A common failure mode in RAG systems occurs when users switch embedding models (e.g., from a 384-dimensional model to a 768-dimensional model) within the same index, causing database crashes. `localRAGvault` solves this by probing the LLM for dimension size upon workspace creation and permanently locking the PostgreSQL table column to that dimension. Subsequent upload attempts with mismatched models are rejected with a clear 400 error.

### 5. Automated CI/CD Toolchain
We maintain high code hygiene using automated GitHub Actions. The `ci.yml` pipeline spins up a live `pgvector/pgvector:pg16` service container, running strict `ruff` linting, `ty` static type validation, and unit test coverage reporting on every PR. When tagging new major/minor releases (`v*.*.*`), `release.yml` automatically compiles multi-architecture images (`linux/amd64`, `linux/arm64`) and pushes them simultaneously to both Docker Hub (`kemalbsoylu/localragvault`) and GitHub Container Registry (GHCR).

### 6. Next Architectural Improvements
* **Hybrid Search Integration:** Add PostgreSQL `tsvector` full-text search alongside `pgvector` dense search, combining results via Reciprocal Rank Fusion (RRF) to improve retrieval for exact keywords, model numbers, product codes, and acronyms.
* **Asynchronous Connection Pooling:** Transition the psycopg database layer to `psycopg_pool.AsyncConnectionPool` to maximize throughput during concurrent multi-turn chat sessions.
* **Token Streaming:** Implement Server-Sent Events (SSE) in FastAPI to stream LLM generation tokens to the frontend in real time, reducing perceived latency.

---

## Environment Variables Reference

The application uses `.env` for local native runs and `.env.docker` when orchestrated via Docker Compose:

| Variable                       | Default Value            | Description                                                     |
|:-------------------------------|:-------------------------|:----------------------------------------------------------------|
| `DB_NAME`                      | `localragvault`          | PostgreSQL database name.                                       |
| `DB_USER`                      | `postgres`               | Database authentication username.                               |
| `DB_PASSWORD`                  | *(empty / secure pass)*  | Database authentication password.                               |
| `DB_HOST`                      | `localhost` (or `db`)    | Database host address (`db` when inside container network).     |
| `DB_PORT`                      | `5432`                   | Database port.                                                  |
| `OLLAMA_BASE_URL`              | `http://localhost:11434` | Endpoint URI (`http://host.docker.internal:11434` in Docker).   |
| `API_BASE_URL`                 | `http://127.0.0.1:8000`  | Internal backend URI for UI (`http://api:8000` in Docker).      |
| `ALLOW_CLOUD_MODELS`           | `False`                  | Set to `True` to allow cloud models (proxies prompts remotely). |
| `MAX_FILE_SIZE_MB`             | `25`                     | Maximum upload file size in megabytes.                          |
| `DEFAULT_EMBEDDING_MODEL`      | `embeddinggemma:latest`  | Default vector embedding model pulled on startup.               |
| `DEFAULT_GENERATION_MODEL`     | `gemma4:latest`          | Default generation LLM pulled on startup.                       |
| `DEFAULT_CHUNK_SIZE`           | `500`                    | Character length per document block.                            |
| `DEFAULT_CHUNK_OVERLAP`        | `100`                    | Overlapping character count between adjacent chunks.            |
| `DEFAULT_TOP_K`                | `5`                      | Default number of vector chunks retrieved per search.           |
| `DEFAULT_SIMILARITY_THRESHOLD` | `0.15`                   | Minimum cosine similarity required to include a chunk.          |
| `DEFAULT_CHAT_HISTORY_LIMIT`   | `10`                     | Number of historical chat turns injected into context.          |

---

## API Reference (OpenAPI Specification)

The REST API is fully documented and interactive via FastAPI's native Swagger UI at:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** (or `http://localhost:8000/docs` when running in Docker).

### Core Endpoint Mapping

| Method   | Endpoint                   | Request Schema            | Response Schema              | Description                                                          |
|:---------|:---------------------------|:--------------------------|:-----------------------------|:---------------------------------------------------------------------|
| `GET`    | `/models/`                 | *None*                    | `ModelListResponse`          | Lists downloaded models currently available in the Ollama daemon.    |
| `GET`    | `/workspaces/`             | *None*                    | `List[WorkspaceResponse]`    | Lists all active document workspaces and locked hyperparameters.     |
| `POST`   | `/workspaces/`             | `WorkspaceCreate`         | `WorkspaceResponse`          | Creates a workspace and probes target embeddings for dimension lock. |
| `PATCH`  | `/workspaces/{id}`         | `WorkspaceUpdate`         | `WorkspaceResponse`          | Modifies workspace hyperparameters (chunk size, overlap, persona).   |
| `DELETE` | `/workspaces/{id}`         | *None*                    | `dict`                       | Permanently deletes a workspace, database records, and disk files.   |
| `POST`   | `/upload/`                 | `Multipart Form`          | `IngestionResponse`          | Ingests a single document (`.txt`, `.md`, `.pdf`, `.docx`, etc.).    |
| `POST`   | `/upload/batch/`           | `Multipart Form (List)`   | `BatchIngestionResponse`     | Batch ingests multiple files with automatic clean upserts.           |
| `GET`    | `/inventory/{id}`          | Query (`limit`, `offset`) | `WorkspaceInventoryResponse` | Returns a paginated list of physical files inside a workspace.       |
| `DELETE` | `/documents/{id}/{file}`   | *None*                    | `dict`                       | Purges a specific document's vector chunks and physical file.        |
| `GET`    | `/workspaces/{id}/threads` | Query (`limit`, `offset`) | `ThreadListResponse`         | Returns paginated conversation threads for a workspace.              |
| `GET`    | `/threads/{id}/messages`   | Query (`limit`, `offset`) | `ThreadHistoryResponse`      | Returns chronological message history for a thread.                  |
| `PATCH`  | `/threads/{id}`            | `ThreadUpdate`            | `ThreadResponse`             | Renames a conversation thread title.                                 |
| `DELETE` | `/threads/{id}`            | *None*                    | `dict`                       | Deletes a thread and its message history.                            |
| `POST`   | `/search/`                 | `SearchQuery`             | `VectorSearchResponse`       | Executes semantic vector similarity search against workspace chunks. |
| `POST`   | `/ask/`                    | `SearchQuery`             | `RAGQueryResponse`           | Executes the full multi-turn RAG pipeline with cited sources.        |

---

## Code Quality & Verification Pipeline

Before committing any changes or submitting a pull request, ensure your local code passes our strict quality verification suite:

### 1. Linting & Formatting (`ruff`)
```bash
# Auto-fix safe lint violations
uv run ruff check --fix

# Auto-format all Python code to 100-character line limits
uv run ruff format
```

### 2. Static Type Checking (`ty`)
```bash
uv run ty check
```

### 3. Automated Test Suite (`pytest`)
We separate fast mocked unit tests from live LLM integration tests:

```bash
# Run unit tests with mocked LLM/embeddings (instant CI feedback, isolated test DB)
uv run pytest -m "not integration" --cov=core --cov-report=term-missing

# Run the live integration suite against local Ollama models
uv run pytest -m integration
```

### 4. Full Quality Check
Run the entire verification command before pushing your branch:
```bash
uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
```

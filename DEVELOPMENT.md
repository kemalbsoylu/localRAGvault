# Architecture & Development Guide

This document covers the technical architecture, maintainability trade-offs, engineering standards, and API reference for `localRAGvault`. It is designed for developers, maintainers, and systems engineers.

---

## Tech Stack & Engine Specifications

* **Text Generation Models:** Open-weight models via Ollama (Default: `gemma4:latest`)
* **Vector Embedding Models:** Open-weight models via Ollama (Default: `embeddinggemma:latest`)
* **Vector Database:** PostgreSQL 16.14 + `pgvector` (via `psycopg` 3 binary driver)
* **Backend Framework:** Python 3.12 (managed by `uv`) + FastAPI + Pydantic v2
* **Frontend Interface:** Streamlit
* **Code Quality & Verification:** Ruff, Ty, Pytest, Pytest-cov

---

## Project Structure

```text
localRAGvault/
├── core/
│   ├── api.py             # FastAPI application, lifecycle management, and REST endpoints
│   ├── config.py          # Centralized environment configurations and hyperparameter defaults
│   ├── database.py        # PostgreSQL/pgvector connection pooling, schema init, and atomic queries
│   ├── extractors.py      # Format-specific text extractors (PDF, DOCX, CSV, JSON, Markdown)
│   ├── logging_config.py  # Rotating file and terminal telemetry logging setup
│   ├── schemas.py         # Strict Pydantic data validation and request/response models
│   └── utils.py           # Ollama client wrappers, text chunking physics, and LLM execution
├── ui/
│   └── app.py             # Streamlit frontend featuring multi-turn chat, vault search, and settings
├── tests/
│   ├── conftest.py          # Pytest fixtures, mock embeddings, and isolated DB provisioning
│   ├── test_workspaces.py   # Workspace CRUD, hyperparameter patching, and dimension safeguards
│   ├── test_documents.py    # Document ingestion, clean upserts, batch upload telemetry, and vector search
│   ├── test_chat_threads.py # Multi-turn RAG execution, message history, and thread renaming validation
│   ├── test_extractors.py   # Format-specific extraction engines (PDF, DOCX, CSV, JSON, Markdown)
│   └── test_utils.py        # Chunking physics, overlap boundary errors, and model tag normalization
├── uploads/               # Mirrored physical storage for ingested files (isolated by workspace ID)
├── logs/                  # System diagnostic logs and rotating error telemetry
├── pyproject.toml         # Project metadata, dependencies, Ruff linting rules, and Pytest markers
├── .env.example           # Environment configuration template
└── README.md              # User quickstart and feature overview
```

---

## Maintainability & Architecture Decisions

This project is engineered to be readable, defensible, and easily maintainable by future contributors. Below are the intentional design trade-offs made during development:

### 1. Strict Domain & Service Boundaries
Rather than writing monolithic route handlers, the application is separated into isolated domain modules:
* `core/extractors.py`: Pure functions dedicated solely to format-specific file parsing and table structural preservation.
* `core/database.py`: Handles connection pooling, vector registration, and atomic SQL execution without leaking database logic to the API.
* `core/utils.py`: Encapsulates Ollama daemon communication, vector embedding batching, and text chunking physics.
* `core/schemas.py`: Single source of truth for Pydantic data contracts.

### 2. Defensive Type Safety & Pydantic Normalization
All API boundaries are strictly governed by Pydantic v2 schemas. Custom field validators (such as enforcing `:latest` tags on models via `normalize_tag`) and model validators (ensuring `chunk_overlap` is strictly less than `chunk_size`) reject malformed payloads at the API edge before they ever touch the database or LLM engine. Python 3.12+ type annotations are used universally and verified statically via `ty check`.

### 3. Vector Pollution Safeguards (Database Resilience)
A common failure mode in RAG systems occurs when users switch embedding models (e.g., from a 384-dimensional model to a 768-dimensional model) within the same index, causing database crashes. `localRAGvault` solves this by probing the LLM for dimension size upon workspace creation and permanently locking the PostgreSQL table column to that dimension. Subsequent upload attempts with mismatched models are rejected with a clear 400 error.

### 4. Deterministic Toolchain & Dependency Hygiene
Used `uv` as the primary package manager and environment resolver. This guarantees deterministic builds across different OS environments via `uv.lock`, dramatically speeds up CI pipeline installations, and eliminates virtual environment discrepancies. Code formatting and linting are strictly standardized using `ruff` with safe auto-fixing.

### 5. Next Architectural Improvements
* **Hybrid Search Integration:** Add PostgreSQL `tsvector` full-text search alongside `pgvector` dense search, combining results via Reciprocal Rank Fusion (RRF) to improve retrieval for exact keywords, model numbers, product codes, and acronyms.
* **Asynchronous Connection Pooling:** Transition the psycopg database layer to `psycopg_pool.AsyncConnectionPool` to maximize throughput during concurrent multi-turn chat sessions.
* **Token Streaming:** Implement Server-Sent Events (SSE) in FastAPI to stream LLM generation tokens to the frontend in real time, reducing perceived latency.

---

## Environment Variables Reference

The application is configured via a `.env` file in the project root. Refer to `.env.example` for the starter template:

| Variable                       | Default Value            | Description                                                     |
|:-------------------------------|:-------------------------|:----------------------------------------------------------------|
| `DB_NAME`                      | `localragvault`          | PostgreSQL database name.                                       |
| `DB_USER`                      | `postgres`               | Database authentication username.                               |
| `DB_PASSWORD`                  | *(empty)*                | Database authentication password.                               |
| `DB_HOST`                      | `localhost`              | Database host address.                                          |
| `DB_PORT`                      | `5432`                   | Database port.                                                  |
| `OLLAMA_BASE_URL`              | `http://localhost:11434` | Endpoint URI for the local Ollama daemon.                       |
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

## API Reference

Once the backend is running, explore and test endpoints interactively via FastAPI's Swagger UI at:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### Endpoint Summary

| Method   | Endpoint                   | Description                                                                    |
|:---------|:---------------------------|:-------------------------------------------------------------------------------|
| `GET`    | `/models/`                 | Lists all downloaded models currently available in the Ollama daemon.          |
| `GET`    | `/workspaces/`             | Lists all active document workspaces and their locked hyperparameters.         |
| `POST`   | `/workspaces/`             | Creates a new workspace and probes the target model for vector dimensions.     |
| `PATCH`  | `/workspaces/{id}`         | Modifies workspace hyperparameters (chunk size, overlap, top-K, persona).      |
| `DELETE` | `/workspaces/{id}`         | Permanently deletes a workspace, database records, and disk files.             |
| `POST`   | `/upload/`                 | Ingests a single document (`.txt`, `.md`, `.pdf`, `.docx`, `.csv`, `.json`).   |
| `POST`   | `/upload/batch/`           | Batch ingests multiple files with automatic clean upserts and error telemetry. |
| `GET`    | `/inventory/{id}`          | Returns a paginated list of indexed documents inside a workspace.              |
| `DELETE` | `/documents/{id}/{file}`   | Purges a specific document's vector chunks and physical file.                  |
| `GET`    | `/workspaces/{id}/threads` | Returns paginated conversation threads for a workspace.                        |
| `GET`    | `/threads/{id}/messages`   | Returns paginated chronological message history for a thread.                  |
| `POST`   | `/search/`                 | Executes semantic vector similarity search against workspace chunks.           |
| `POST`   | `/ask/`                    | Executes the full multi-turn RAG pipeline, synthesizing context and history.   |

---

## Code Quality & Verification Pipeline

Before committing any changes, ensure your code passes the complete verification suite:

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
Separated fast mocked unit tests from live LLM integration tests:

```bash
# Run unit tests with mocked LLM/embeddings (instant feedback, isolated DB)
uv run pytest -m "not integration" --cov=core --cov-report=term-missing

# Run the live integration suite against local Ollama models
uv run pytest -m integration
```

### 4. Full Quality Check
Run the entire pipeline in a single command before submitting a pull request:
```bash
uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
```

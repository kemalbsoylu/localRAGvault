# localRAGvault

A fully local, privacy-first Retrieval-Augmented Generation (RAG) pipeline. 
This tool allows you to securely ingest documents and query them locally using open-weight models without relying on external APIs.

## Features
* **100% Local Execution:** Zero data leaves your machine. Powered by Ollama.
* **Isolated Workspaces:** Group your documents into distinct workspaces. Each workspace permanently locks to a specific embedding model and vector dimension upon creation.
* **Vector Dimension Safeguards:** Strict backend validation prevents "Vector Pollution" by rejecting model mismatches, ensuring your `pgvector` database never crashes from dimension collisions.
* **Physical File Storage:** Maintains a mirrored physical copy of your ingested documents on disk, logically separated by workspace ID.
* **Dynamic Model Steering:** Switch text generation models on the fly directly from the UI, while keeping your isolated vector search strictly locked to the workspace's embedding model.
* **Zero-Touch Startup:** Automatically probes the local environment and pulls missing default Ollama models during the API boot sequence.
* **Robust Backend Architecture:** Built with FastAPI, strictly normalized Pydantic schemas, centralized configuration, and rotating file telemetry logs.
* **Advanced Vector DB:** Fast, reliable, and scalable vector similarity search powered by PostgreSQL and `pgvector`.

## Tech Stack
* **Generation Models:** Supports all open-weight models via Ollama (Default: `gemma3`)
* **Embedding Models:** Supports all open-weight models via Ollama (Default: `embeddinggemma`)
* **Vector Database:** PostgreSQL 16 + `pgvector`
* **Backend:** Python (managed by `uv`) + FastAPI + Pydantic
* **Frontend:** Streamlit (Prototype) -> React/Vite (Stable target)

## Prerequisites

Before running this project, ensure your local environment meets the following requirements:

1.  **Ollama** (v0.21.0+)
    * Default models to pull: `ollama pull gemma3` & `ollama pull embeddinggemma`
2.  **PostgreSQL** (v16+)
    * Must have the `pgvector` extension installed.
3.  **uv** (v0.11.6+)
    * Used for Python dependency management.

> **Caution Regarding Ollama Cloud Models:**
> 
> This application is built local-first to guarantee strict document privacy. While Ollama allows using cloud hosted
> open models (e.g., aliases ending in `:cloud`), selecting a cloud model will proxy your prompt and document context to remote servers. 
> 
> If you explicitly choose to use Ollama Cloud models, you must first authenticate your local daemon by running `ollama signin` in your terminal:
> ```bash
> ollama signin
> ollama pull gemma4:cloud
> ```
> For guaranteed zero-data-leak privacy, stick to fully local downloaded models for both embedding and generation.

## Installation & Setup

1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/kemalbsoylu/localRAGvault.git
   cd localRAGvault
   ```
   
2. Sync the Python environment using `uv`:
   ```bash
   uv sync
   ```

3. Configure your environment variables.
   Create a `.env` file in the root directory and add your Postgres credentials:
   ```env
   DB_NAME=localragvault
   DB_USER=kemal
   DB_PASSWORD=your_secure_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

## Running the Application

This project currently uses a split architecture for development. You will need to run the backend and the frontend in separate terminal windows.

**1. Start the FastAPI Backend**
```bash
uv run uvicorn core.api:app --reload --reload-dir core
```
*The API will be available at `http://127.0.0.1:8000`. The normalized database schema and default models initialize automatically on startup.*

**2. Start the Streamlit Frontend**

Open a new terminal window and run:
```bash
uv run streamlit run ui/app.py
```
*The UI will automatically open in your browser at `http://localhost:8501`*

## API Documentation

FastAPI automatically generates interactive Swagger UI documentation for all endpoints and Pydantic schemas. 

Once the backend is running, you can explore and test the API directly in your browser by navigating to:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

## Project Structure
* `core/`: FastAPI application, Pydantic schemas, DB connections, configs, and utilities.
* `ui/`: Streamlit frontend prototype featuring workspace management.
* `tests/`: Pytest suite covering endpoints, isolated data contracts, and LLM behavior.
* `uploads/`: Local physical storage for ingested documents, organized internally by workspace IDs.
* `logs/`: System diagnostic and telemetry logs (auto-generated).

## Code Quality & Testing

### 1. Code Quality & Type Checking
Run linting, formatting, and static type checking:

```bash
# Check for lint issues and automatically fix safe violations
uv run ruff check --fix

# Auto-format all Python code
uv run ruff format

# Run static type checking
uv run ty check
```

### 2. Running Unit & Integration Tests

Run the standard test suite with coverage (uses mocks to bypass LLM generation for instant feedback and spins up an isolated `localragvault_test` database):
```bash
uv run pytest -m "not integration" --cov=core --cov-report=term-missing
```

Run the full integration suite (tests against your actual running local Ollama models):
```bash
uv run pytest -m integration
```

### 3. Full Quality Suite
To verify linting, formatting, type safety, and unit tests all at once before committing:
```bash
uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest
```

## License

[MIT License](LICENSE)

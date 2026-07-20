# localRAGvault

A fully local, privacy-first Retrieval-Augmented Generation (RAG) pipeline. 
This tool allows you to securely ingest documents and query them locally using open-weight models without relying on external APIs.

## Features
* **100% Local Execution:** Zero data leaves your machine. Powered by Ollama.
* **Dynamic Model Steering:** Switch between embedding and generation models on the fly directly from the UI.
* **Vector Separation:** Safely stores multiple embedding spaces in the same Postgres table without vector pollution.
* **Stateless Search UI:** A clean interface that mimics a private search engine, complete with expandable citations and UI-locked processing.
* **Robust Backend Architecture:** Built with FastAPI, Pydantic typing, centralized configuration, and rotating file logging.
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

3. Configure your environment variables:
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
uv run uvicorn core.api:app --reload
```
*The API will be available at `http://127.0.0.1:8000`. The database table initializes automatically on startup.*

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
* `ui/`: Streamlit frontend prototype.
* `tests/`: Pytest suite covering endpoints, data contracts, and LLM behavior.
* `uploads/`: Local storage for ingested documents.
* `logs/`: System diagnostic and telemetry logs (auto-generated).

## Testing

Run the standard test suite with coverage (uses mocks to bypass LLM generation for instant feedback):
```bash
uv run pytest -m "not integration" --cov=core --cov-report=term-missing
```

Run the full integration suite (tests against your actual running local Ollama models):
```bash
uv run pytest -m integration
```

## License

[MIT License](LICENSE)

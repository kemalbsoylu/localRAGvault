# localRAGvault

A fully local, privacy-first Retrieval-Augmented Generation (RAG) pipeline. 
This tool allows you to ingest documents and query them locally without relying on external APIs.

## Tech Stack
* **Generation Model:** `gemma4` (via Ollama)
* **Embedding Model:** `embeddinggemma` (via Ollama)
* **Vector Database:** PostgreSQL 16 + `pgvector`
* **Backend:** Python (managed by `uv`) + FastAPI
* **Frontend:** Streamlit (Prototype) -> React/Vite (Stable)

## Prerequisites

Before running this project, ensure your local environment meets the following requirements:

1.  **Ollama** (v0.21.0+)
    * Models required: `ollama pull gemma4` & `ollama pull embeddinggemma`
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
*The API will be available at `http://127.0.0.1:8000`*

**2. Start the Streamlit Frontend**
Open a new terminal window and run:
```bash
uv run streamlit run ui/app.py
```
*The UI will automatically open in your browser at `http://localhost:8501`*

## Project Structure
* `core/`: Contains the FastAPI application, database connections, and RAG utility functions.
* `ui/`: Contains the Streamlit frontend prototype.
* `uploads/`: Local storage for ingested documents.
* `tests/`: Test scripts for verifying API endpoints.

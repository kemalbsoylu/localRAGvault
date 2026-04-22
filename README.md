# localRAGvault

A fully local, privacy-first Retrieval-Augmented Generation (RAG) pipeline. 
This tool allows you to ingest documents and query them locally without relying on external APIs.

## Tech Stack
* **Generation Model:** `gemma4` (via Ollama)
* **Embedding Model:** `embeddinggemma` / `nomic-embed-text` (via Ollama)
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
   git clone <your-repo-url>
   cd localRAGvault
   ```
   
2. Sync the Python environment using `uv`:
   ```bash
   uv sync
   ```

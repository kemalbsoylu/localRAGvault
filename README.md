# localRAGvault

Your privacy-first, fully local **Retrieval-Augmented Generation (RAG)** assistant. 

**localRAGvault** allows you to securely chat with your private documents, financial reports, research papers, and spreadsheets using open-weight AI models. Everything runs 100% locally on your machine. No cloud subscriptions, no external APIs, and zero data leaks to third-party servers.

---

## Why localRAGvault?

* 🔒 **100% Privacy Guaranteed:** All text extraction, vector embeddings, and AI generation happen locally on your hardware using Ollama. Your sensitive documents never leave your computer.
* 🗄️ **Isolated Document Vaults:** Organize different projects into distinct workspaces. Each workspace acts as an independent memory vault with its own customized instructions and retrieval settings. The system includes automatic dimension safeguards that prevent "Vector Pollution".
* 📄 **Multi-Format Support:** Easily upload `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, and `.json` files. The built-in engine automatically organizes complex layouts, including Word tables and spreadsheet rows, so the AI understands your data accurately.
* 📦 **Bulk Uploads:** Upload multiple files simultaneously or drag-and-drop entire folders. The system replaces outdated files without creating duplicate memory, and provides clear summary reports tracking your successful uploads, updates, and failures.
* ⚙️ **Customizable AI Physics:** Tailor how the assistant reads and retrieves your data. Easily adjust search depth, conversational memory limits, and custom AI personalities for each workspace directly from the dashboard.
* 💬 **Multi-Turn Chat Memory:** Have natural, continuous conversations. The assistant remembers what you discussed earlier in the chat thread and cites the exact document paragraphs it used to answer your questions.
* ⚡ **Zero-Touch Setup:** The app automatically checks your environment on startup, generates the required database tables, and pulls any missing AI models in the background.

---

## Quickstart Guide

### Prerequisites
To run localRAGvault, you will need three lightweight tools installed on your system:
1. **[Ollama](https://ollama.com/download)** (v0.32.0 or newer) – Runs the local AI models (defaults are `gemma4` and `embeddinggemma`).
2. **[PostgreSQL](https://www.postgresql.org/download/)** (v16.14) with the `pgvector` extension enabled – Stores your document memory.
3. **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (v0.11.32 or newer) – Python package manager.

### Installation & Launch

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kemalbsoylu/localRAGvault.git
   cd localRAGvault
   ```

2. **Create the database and enable pgvector:**
   Run the following commands in your terminal to create the empty database and enable the vector engine (replace `postgres` with your database username if different):
   ```bash
   psql -U postgres -c "CREATE DATABASE localragvault;"
   psql -U postgres -d localragvault -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

3. **Configure your environment:**
   Copy the example environment file and update it with your PostgreSQL credentials:
   ```bash
   cp .env.example .env
   ```
   *(Note: Once the empty database is created and your `.env` is set, the app will automatically build all required internal tables and schemas on your very first launch!)*

4. **Install dependencies:**
   ```bash
   uv sync
   ```

5. **Start the Application:**
   This app uses a split architecture. Open two terminal windows:

   * **Terminal 1 (Start the Backend API):**
     ```bash
     uv run uvicorn core.api:app --reload --reload-dir core
     ```
     *The backend server will start at `http://127.0.0.1:8000` and initialize your database tables automatically.*

   * **Terminal 2 (Start the Web Interface):**
     ```bash
     uv run streamlit run ui/app.py
     ```
     *Your browser will automatically open the interactive dashboard at `http://localhost:8501`.*

---

### Caution Regarding Ollama Cloud Models

While localRAGvault is engineered local-first for strict data privacy, Ollama allows connecting to cloud-hosted open models (e.g., tags ending in `:cloud`). 

> ⚠️ **Warning:** Using cloud models will proxy your prompts and document excerpts to remote servers. For guaranteed privacy, restrict your workspaces exclusively to locally downloaded models.

If you explicitly choose to use Ollama Cloud models, authenticate your local daemon first:
```bash
ollama signin
ollama pull gemma4:cloud
```
Then update your `.env` file to enable cloud access:
```env
ALLOW_CLOUD_MODELS=True
```

---

## For Developers & Contributors

Interested in the technical architecture, maintainability trade-offs, API endpoints, or contributing to the codebase?
* Read **[Architecture & Development Guide](DEVELOPMENT.md)**
* Read **[Contribution Guidelines](CONTRIBUTING.md)**

---

## License

This project is open-source and available under the [MIT License](LICENSE).

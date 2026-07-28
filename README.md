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

Use **Docker Desktop** for an isolated, one-click environment. No need to manually install Python, databases, or configure complicated background services. Docker handles everything cleanly.

### 1. Prerequisites

Before launching the app, ensure you have these two applications installed and running on your computer:
1. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)** – Runs the application containers securely.
2. **[Ollama](https://ollama.com/download)** (v0.32.0 or newer) – Powers the local AI models on your native operating system. 

*(Note: You do not need to manually download any AI models! localRAGvault will automatically pull the required default models in the background on your very first launch.)*

---

### 2. Allow Docker to Connect to Ollama (Mandatory One-Time Step)

Because Ollama runs natively on your operating system while localRAGvault runs inside Docker Desktop, you must configure Ollama to accept incoming connections from your containers.

* **macOS:** Open your terminal application and run:
  ```bash
  launchctl setenv OLLAMA_HOST "0.0.0.0"
  ```
  *(Restart the Ollama application from your menu bar after running this command).*


* **Windows:** 
  1. Quit the Ollama application from your bottom-right taskbar tray.
  2. Open Control Panel > System and Security > System > Advanced system settings > Environment Variables.
  3. Under **User variables**, click **New**, set Variable name to `OLLAMA_HOST` and Variable value to `0.0.0.0`.
  4. Relaunch Ollama.


* **Linux:** Edit the systemd service file by running `sudo systemctl edit ollama.service` in your terminal and add:
  ```ini
  [Service]
  Environment="OLLAMA_HOST=0.0.0.0"
  ```
  Then reload and restart the service: `sudo systemctl daemon-reload && sudo systemctl restart ollama`.

---

### 3. Launch the Application

Choose whichever launch method works best for you:

#### Method A: Quick Launch via Docker Desktop (No Git Required)
If you do not have Git installed or prefer not to clone source code repositories, you can launch directly using verified Docker Hub images (`kemalbsoylu/localragvault`):

1. Download single configuration file: **[docker-compose.yml](https://raw.githubusercontent.com/kemalbsoylu/localRAGvault/main/docker-compose.yml)** *(Right-click the link and select "Save As..." to save it into a new, empty folder on your computer).*
2. Open your terminal (or PowerShell/Command Prompt on Windows), navigate into that folder, and start the app:
   ```bash
   docker compose up -d
   ```
3. Open your web browser and go to **[http://localhost:8501](http://localhost:8501)**. Your secure RAG vault is ready to use!

#### Method B: Clone & Launch (Using Git)
If you prefer standard cloning or want to inspect the codebase locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kemalbsoylu/localRAGvault.git
   cd localRAGvault
   ```
2. **Start the containers:**
   ```bash
   docker compose up -d
   ```
3. Open your web browser and go to **[http://localhost:8501](http://localhost:8501)**.

---

### Caution Regarding Ollama Cloud Models

While localRAGvault is engineered local-first for strict data privacy, Ollama allows connecting to cloud-hosted open models (e.g., tags ending in `:cloud`). 

> ⚠️ **Warning:** Using cloud models will proxy your prompts and document excerpts to remote servers. For guaranteed privacy, restrict your workspaces exclusively to locally downloaded models.

If you explicitly choose to use Ollama Cloud models, authenticate your local daemon first:
```bash
ollama signin
ollama pull gemma4:cloud
```
Then update your environment configuration to allow cloud access:
```env
ALLOW_CLOUD_MODELS=True
```

---

## For Developers & Contributors

Are you a developer wanting to run the application natively without Docker (using `uv` and PostgreSQL directly), explore API endpoints, or contribute to the codebase?
* Read **[Architecture & Development Guide](DEVELOPMENT.md)** *(Includes step-by-step native local setup instructions)*
* Read **[Contribution Guidelines](CONTRIBUTING.md)**

---

## License

This project is open-source and available under the [MIT License](LICENSE).

import os
from pathlib import Path

from dotenv import load_dotenv

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")


# Database Configuration
DB_NAME = os.getenv("DB_NAME", "localragvault")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")


# Model Governance & Configuration
ALLOW_CLOUD_MODELS = os.getenv("ALLOW_CLOUD_MODELS", "False").lower() in ("true", "1", "yes")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "embeddinggemma:latest")
DEFAULT_GENERATION_MODEL = os.getenv("DEFAULT_GENERATION_MODEL", "gemma3:latest")


# RAG Hyperparameters (Chunking & Retrieval)
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.15"))


# Conversation Memory Hyperparameters
DEFAULT_CHAT_HISTORY_LIMIT = int(os.getenv("DEFAULT_CHAT_HISTORY_LIMIT", "6"))


# File Storage Configuration
if os.getenv("DB_NAME") == "localragvault_test":
    UPLOAD_DIR = BASE_DIR / "uploads_test"
else:
    UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)


# Logging Setup
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

if os.getenv("DB_NAME") == "localragvault_test":
    LOG_FILE = LOG_DIR / "test_localragvault.log"
else:
    LOG_FILE = LOG_DIR / "localragvault.log"

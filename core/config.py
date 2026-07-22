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


# Model Configuration
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"
DEFAULT_GENERATION_MODEL = "gemma3:latest"


# RAG Hyperparameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


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

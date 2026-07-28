import os
import shutil
from unittest.mock import MagicMock, patch

import psycopg
import pytest

# MANDATORY: Override database name globally BEFORE importing project modules
os.environ["DB_NAME"] = "localragvault_test"

from core.config import BASE_DIR, DB_HOST, DB_PASSWORD, DB_PORT, DB_USER
from core.database import get_db_connection, init_db


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Runs ONCE per test session.
    Automatically provisions the test database and vector extension if missing.
    """
    try:
        # 1. Connect to default postgres db to create the isolated test database
        with psycopg.connect(
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            autocommit=True,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = 'localragvault_test'")
                if not cur.fetchone():
                    print("\nProvisioning isolated test database: localragvault_test...")
                    cur.execute("CREATE DATABASE localragvault_test")

        # 2. Connect to the test database to verify/install the pgvector extension
        with psycopg.connect(
            dbname="localragvault_test",
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            autocommit=True,
        ) as conn_test:
            with conn_test.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                if not cur.fetchone():
                    try:
                        cur.execute("CREATE EXTENSION vector")
                    except psycopg.errors.InsufficientPrivilege:
                        pytest.exit(
                            "\n❌ Missing superuser permissions to install pgvector.\n"
                            "Run this command in your terminal once to fix it:\n"
                            'sudo -u postgres psql -d localragvault_test -c "CREATE EXTENSION vector;"'
                        )

        # 3. Initialize application schema
        init_db()

    except Exception as e:
        pytest.exit(f"Critical failure provisioning the test database: {e}")


@pytest.fixture(autouse=True)
def clean_database():
    """
    Runs BEFORE AND AFTER EVERY test.
    Ensures tests do not pollute each other's vector space or chat history.
    """
    yield
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE TABLE workspaces, documents, threads, messages RESTART IDENTITY CASCADE;"
            )


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_uploads():
    """Cleans up the isolated test upload directory after the test session ends."""
    yield
    test_upload_dir = BASE_DIR / "uploads_test"
    if test_upload_dir.exists():
        shutil.rmtree(test_upload_dir, ignore_errors=True)


# =====================================================================
# --- UNIVERSAL OLLAMA SDK & HTTP MOCKING FOR CI/CD ---
# =====================================================================


@pytest.fixture(autouse=True)
def mock_ollama_for_unit_tests(request):
    """
    Runs automatically for all tests NOT marked 'integration'.
    Intercepts both module-level functions and pre-instantiated official 'ollama' SDK
    class methods so unit tests run instantly in CI without needing real AI models.
    """
    if "integration" in request.keywords:
        yield
        return

    class MockModel:
        def __init__(self, name):
            self.name = name
            self.model = name

        def get(self, key, default=None):
            return getattr(self, key, default)

        def __getitem__(self, key):
            return getattr(self, key)

    class MockListResponse:
        def __init__(self, models):
            self.models = models

        def get(self, key, default=None):
            return getattr(self, key, default)

        def __getitem__(self, key):
            return getattr(self, key)

    mock_list_res = MockListResponse(
        [
            MockModel("embeddinggemma:latest"),
            MockModel("gemma4:latest"),
            MockModel("differentmodel:latest"),
        ]
    )

    mock_embed_res = {"embedding": [0.1] * 768, "embeddings": [[0.1] * 768]}
    mock_show_res = {"modelfile": "# Modelfile\nPARAMETER ..."}
    mock_chat_res = {
        "response": "Mocked AI response.",
        "message": {"role": "assistant", "content": "Mocked AI response."},
    }

    with (
        patch("requests.get") as mock_get,
        patch("requests.post") as mock_post,
        patch("requests.delete") as mock_delete,
        patch("ollama.list", return_value=mock_list_res),
        patch("ollama.show", return_value=mock_show_res),
        patch("ollama.embeddings", return_value=mock_embed_res),
        patch("ollama.embed", return_value=mock_embed_res),
        patch("ollama.chat", return_value=mock_chat_res),
        patch("ollama.generate", return_value=mock_chat_res),
        patch("ollama.Client.list", return_value=mock_list_res),
        patch("ollama.Client.show", return_value=mock_show_res),
        patch("ollama.Client.embeddings", return_value=mock_embed_res),
        patch("ollama.Client.embed", return_value=mock_embed_res),
        patch("ollama.Client.chat", return_value=mock_chat_res),
        patch("ollama.Client.generate", return_value=mock_chat_res),
    ):

        def side_effect_get(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "/api/tags" in str(url):
                mock_resp.json.return_value = {
                    "models": [
                        {
                            "name": "embeddinggemma:latest",
                            "model": "embeddinggemma:latest",
                        },
                        {"name": "gemma4:latest", "model": "gemma4:latest"},
                        {
                            "name": "differentmodel:latest",
                            "model": "differentmodel:latest",
                        },
                    ]
                }
            return mock_resp

        def side_effect_post(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "/api/show" in str(url):
                mock_resp.json.return_value = mock_show_res
            elif "/api/embeddings" in str(url) or "/api/embed" in str(url):
                mock_resp.json.return_value = mock_embed_res
            elif "/api/generate" in str(url) or "/api/chat" in str(url):
                mock_resp.json.return_value = mock_chat_res
            return mock_resp

        mock_get.side_effect = side_effect_get
        mock_post.side_effect = side_effect_post
        mock_delete.return_value = MagicMock(status_code=200)

        yield

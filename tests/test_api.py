import pytest

from fastapi.testclient import TestClient
from unittest.mock import patch
from core.api import app


client = TestClient(app)


def test_health_check():
    """Test that the API is up and running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "localRAGvault API is running securely.",
    }


def test_upload_document():
    """Test successful upload and chunking of a valid text file."""
    file_content = b"This is a test document for localRAGvault testing. It verifies the chunking and embedding pipeline."
    files = {"file": ("test_doc.txt", file_content, "text/plain")}

    response = client.post("/upload/", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_doc.txt"
    assert data["chunks_saved"] > 0


def test_upload_invalid_file_type():
    """Test that uploading unsupported extensions returns a 400 error."""
    file_content = b"Fake PDF content"
    files = {"file": ("test_doc.pdf", file_content, "application/pdf")}

    response = client.post("/upload/", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .txt and .md files are supported for now."


def test_search_documents():
    """Test the vector search functionality."""
    data = {"query": "What database does localRAGvault use?", "top_k": 2}

    response = client.post("/search/", json=data)

    assert response.status_code == 200
    result = response.json()
    assert "query" in result
    assert "results" in result
    assert isinstance(result["results"], list)


@patch("core.api.generate_answer")
def test_ask_question(mock_generate):
    """Test the full RAG loop. We mock generate_answer."""
    # Force the mock to return a fake answer whenever it is called
    mock_generate.return_value = "This is a mocked LLM answer for testing."

    data = {"query": "What is localRAGvault?", "top_k": 2}

    response = client.post("/ask/", json=data)

    assert response.status_code == 200
    result = response.json()

    assert result["answer"] == "This is a mocked LLM answer for testing."
    assert "sources" in result
    assert isinstance(result["sources"], list)


@pytest.mark.integration
def test_ask_question_real_model():
    """
    Test the full RAG loop USING THE REAL LOCAL LLM.
    This will take a few seconds to run and tests actual generation.
    """
    # 1. First, upload a document so the database has context
    file_content = b"localRAGvault is a privacy-first, fully local RAG architecture. It uses Ollama to serve generation and embedding models directly on the host machine. By utilizing PostgreSQL and pgvector, it stores document embeddings securely without relying on external APIs or cloud services."
    files = {"file": ("integration_doc.txt", file_content, "text/plain")}
    client.post("/upload/", files=files)

    # 2. Ask a question about that specific document
    data = {
        "query": "What is localRAGvault's privacy approach?",
        "top_k": 1,
    }

    # 3. Hit the endpoint without a mock
    response = client.post("/ask/", json=data)

    assert response.status_code == 200
    result = response.json()

    # 4. Verify the LLM actually generated a response
    assert "local" in result["answer"]
    assert len(result["sources"]) > 0
    assert result["sources"][0]["filename"] == "integration_doc.txt"

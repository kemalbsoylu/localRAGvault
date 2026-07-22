from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.api import app
from core.schemas import LLMInternalResponse

client = TestClient(app)


def test_health_check() -> None:
    """Test that the API is up, running, and returns secure telemetry parameters."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "localRAGvault API is running securely.",
    }


def test_list_models_endpoint() -> None:
    """Verify models endpoint response structure matches schema contracts."""
    response = client.get("/models/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models" in data
    assert isinstance(data["models"], list)


def test_upload_document_success() -> None:
    """Test successful upload, parsing, chunking, and db serialization of valid text."""
    file_content = (
        b"This is a test document for localRAGvault tracking. It verifies text parsing loops."
    )
    files = {"file": ("test_doc.txt", file_content, "text/plain")}

    response = client.post("/upload/", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_doc.txt"
    assert data["chunks_saved"] > 0


def test_upload_invalid_file_type() -> None:
    """Test that uploading unsupported extensions yields a strict 400 rejection error."""
    file_content = b"Fake PDF processing binary stream layout details"
    files = {"file": ("test_doc.pdf", file_content, "application/pdf")}

    response = client.post("/upload/", files=files)

    assert response.status_code == 400
    assert response.json()["detail"] == "Only .txt and .md files are supported for now."


def test_upload_corrupted_encoding_binary() -> None:
    """Test that uploading non-UTF-8 binary data throws a handled 400 exception."""
    # Sending raw high-bit binary bytes that cannot be decoded as clean UTF-8
    corrupted_bytes = b"\x80\x81\x82\xff"
    files = {"file": ("corrupted.txt", corrupted_bytes, "text/plain")}

    response = client.post("/upload/", files=files)
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_search_documents_response_contract() -> None:
    """Test the vector search functionality and response format validity."""
    data = {"query": "What database does localRAGvault use?", "top_k": 2}

    response = client.post("/search/", json=data)

    assert response.status_code == 200
    result = response.json()
    assert result["query"] == "What database does localRAGvault use?"
    assert isinstance(result["results"], list)


@patch("core.api.generate_answer")
def test_ask_question_with_valid_context(mock_generate) -> None:
    """Test full pipeline loop under mocked conditions returning successful answers."""
    # 1. Upload dummy data to populate the isolated vector DB first
    file_content = b"localRAGvault is a privacy-first, fully local RAG architecture."
    client.post("/upload/", files={"file": ("test_doc.txt", file_content, "text/plain")})

    # 2. Mock the LLM generation
    mock_generate.return_value = LLMInternalResponse(
        text="This is a mocked LLM answer for testing.", is_valid=True
    )

    # 3. Test the /ask/ endpoint
    data = {"query": "What is localRAGvault?", "top_k": 2}
    response = client.post("/ask/", json=data)

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "This is a mocked LLM answer for testing."
    assert len(result["sources"]) > 0


@patch("core.api.generate_answer")
def test_ask_question_hides_sources_on_invalid_context(mock_generate) -> None:
    """Verify that sources are wiped from the output package if the LLM cannot answer."""
    # 1. Upload dummy data so the database actually returns context chunks
    file_content = b"Some unrelated context about apples and oranges."
    client.post("/upload/", files={"file": ("unrelated.txt", file_content, "text/plain")})

    # 2. Mock the LLM returning the failure string
    mock_generate.return_value = LLMInternalResponse(
        text="I cannot answer this based on the provided documents.", is_valid=False
    )

    # 3. Ask a question
    data = {"query": "Random query that doesn't exist", "top_k": 1}
    response = client.post("/ask/", json=data)

    assert response.status_code == 200
    result = response.json()
    assert result["sources"] == []  # Source array must be cleared out dynamically


@pytest.mark.integration
def test_ask_question_real_model() -> None:
    """Test the integration loop directly against working local Ollama model spaces."""
    # 1. First, upload a document so the database has context
    file_content = b"localRAGvault is a privacy-first, fully local RAG architecture. It uses Ollama to serve generation."
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

    # 4. Verify the LLM actually generated a response using the document
    assert (
        "local" in result["answer"].lower()
        or "privacy" in result["answer"].lower()
        or "ollama" in result["answer"].lower()
    )
    assert len(result["sources"]) > 0
    assert result["sources"][0]["filename"] == "integration_doc.txt"


def test_workspace_inventory_endpoint() -> None:
    """Verify that uploading a file populates the workspace inventory ledger."""
    file_content = b"Inventory tracking test payload bytes."
    client.post("/upload/", files={"file": ("inventory_test.txt", file_content, "text/plain")})

    response = client.get("/inventory/default")
    assert response.status_code == 200
    data = response.json()

    assert data["workspace_id"] == "default"
    assert len(data["documents"]) > 0
    assert data["documents"][0]["filename"] == "inventory_test.txt"
    assert "inventory_test.txt" in data["documents"][0]["file_path"]

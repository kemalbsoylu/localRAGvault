import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.api import app
from core.config import DEFAULT_EMBEDDING_MODEL
from core.schemas import LLMInternalResponse

client = TestClient(app)


def _create_test_workspace(name: str) -> str:
    """Helper to provision an isolated workspace for a test and return its ID."""
    response = client.post(
        "/workspaces/", json={"name": name, "embedding_model": DEFAULT_EMBEDDING_MODEL}
    )
    if response.status_code != 200:
        pytest.fail(f"Failed to create test workspace: {response.text}")
    return response.json()["id"]


# =====================================================================
# --- 1. HEALTH & CATALOG TESTS ---
# =====================================================================


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


# =====================================================================
# --- 2. WORKSPACE LIFECYCLE & DELETION TESTS ---
# =====================================================================


def test_create_workspace_endpoint() -> None:
    """Test successful creation, dimension locking, and schema validation of a workspace."""
    response = client.post(
        "/workspaces/", json={"name": "Alpha Workspace", "embedding_model": DEFAULT_EMBEDDING_MODEL}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alpha Workspace"
    assert "id" in data
    assert data["dimension"] > 0


def test_delete_workspace_endpoint() -> None:
    """Verify deleting a workspace wipes database records and blocks future searches."""
    ws_id = _create_test_workspace("To Be Deleted WS")

    # Upload a document to verify cascade deletion
    client.post(
        "/upload/",
        files={"file": ("dummy.txt", b"temporary content", "text/plain")},
        data={"workspace_id": ws_id},
    )

    # Delete workspace
    del_res = client.delete(f"/workspaces/{ws_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    # Verify workspace is gone from inventory check
    inv_res = client.get(f"/inventory/{ws_id}")
    assert inv_res.status_code == 404


# =====================================================================
# --- 3. DOCUMENT INGESTION, MULTI-FORMAT & UPSERT TESTS ---
# =====================================================================


def test_upload_document_txt_success() -> None:
    """Test successful text upload, chunking, and db serialization."""
    ws_id = _create_test_workspace("Txt Upload WS")

    file_content = (
        b"This is a test document for localRAGvault tracking. It verifies text parsing loops."
    )
    files = {"file": ("test_doc.txt", file_content, "text/plain")}
    response = client.post("/upload/", files=files, data={"workspace_id": ws_id})

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["workspace_id"] == ws_id
    assert res_data["filename"] == "test_doc.txt"
    assert res_data["chunks_saved"] > 0
    assert res_data["is_upsert"] is False


def test_upload_document_csv_success() -> None:
    """Verify structured CSV data is parsed into natural language sentences and indexed."""
    ws_id = _create_test_workspace("CSV Upload WS")

    csv_bytes = b"id,name,role\n101,Alice,Engineer\n102,Bob,Manager"
    files = {"file": ("staff.csv", csv_bytes, "text/csv")}
    response = client.post("/upload/", files=files, data={"workspace_id": ws_id})

    assert response.status_code == 200
    assert response.json()["chunks_saved"] > 0


def test_upload_document_json_success() -> None:
    """Verify JSON payloads are flattened into searchable text blocks."""
    ws_id = _create_test_workspace("JSON Upload WS")

    json_payload = [{"project": "localRAGvault", "status": "active", "version": "0.1.0"}]
    json_bytes = json.dumps(json_payload).encode("utf-8")
    files = {"file": ("config.json", json_bytes, "application/json")}
    response = client.post("/upload/", files=files, data={"workspace_id": ws_id})

    assert response.status_code == 200
    assert response.json()["chunks_saved"] > 0


def test_upload_clean_upsert() -> None:
    """Test that re-uploading an existing file name purges old chunks before saving new ones."""
    ws_id = _create_test_workspace("Upsert WS")

    # 1. Initial upload
    files = {"file": ("versioned.txt", b"Version 1 content string.", "text/plain")}
    res1 = client.post("/upload/", files=files, data={"workspace_id": ws_id})
    assert res1.status_code == 200
    assert res1.json()["is_upsert"] is False
    initial_chunks = res1.json()["chunks_saved"]

    # 2. Re-upload same filename with new content
    files_updated = {
        "file": ("versioned.txt", b"Version 2 updated expanded content string.", "text/plain")
    }
    res2 = client.post("/upload/", files=files_updated, data={"workspace_id": ws_id})

    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["is_upsert"] is True
    assert data2["chunks_deleted"] == initial_chunks
    assert data2["chunks_saved"] > 0


def test_upload_invalid_file_type() -> None:
    """Test that uploading unsupported extensions yields a strict 400 rejection error."""
    ws_id = _create_test_workspace("Invalid File Test WS")

    file_content = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00"
    files = {"file": ("malicious.exe", file_content, "application/octet-stream")}
    response = client.post("/upload/", files=files, data={"workspace_id": ws_id})

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_corrupted_encoding_binary() -> None:
    """Test that uploading malformed JSON data throws a handled 400 exception."""
    ws_id = _create_test_workspace("Corrupted Encoding Test WS")

    malformed_json_bytes = b"{ invalid json syntax payload }"
    files = {"file": ("malformed.json", malformed_json_bytes, "application/json")}
    response = client.post("/upload/", files=files, data={"workspace_id": ws_id})

    assert response.status_code == 400


# =====================================================================
# --- 4. DOCUMENT DELETION & INVENTORY TESTS ---
# =====================================================================


def test_workspace_inventory_and_deletion() -> None:
    """Verify inventory tracking and single document deletion loops."""
    ws_id = _create_test_workspace("Inventory Tracking WS")

    # Upload file
    client.post(
        "/upload/",
        files={"file": ("to_delete.txt", b"Delete me soon.", "text/plain")},
        data={"workspace_id": ws_id},
    )

    # Verify present in inventory
    inv_res = client.get(f"/inventory/{ws_id}")
    assert inv_res.status_code == 200
    assert len(inv_res.json()["documents"]) == 1

    # Delete document
    del_res = client.delete(f"/documents/{ws_id}/to_delete.txt")
    assert del_res.status_code == 200
    assert del_res.json()["chunks_deleted"] > 0

    # Verify inventory is empty
    inv_res_after = client.get(f"/inventory/{ws_id}")
    assert len(inv_res_after.json()["documents"]) == 0


# =====================================================================
# --- 5. VECTOR SEARCH & CHUNK METADATA TESTS ---
# =====================================================================


def test_search_documents_response_contract() -> None:
    """Test vector search functionality, verifying chunk_index is returned."""
    ws_id = _create_test_workspace("Search Contract Test WS")

    # Ingest dummy document
    client.post(
        "/upload/",
        files={
            "file": (
                "doc_metadata.txt",
                b"PostgreSQL is our core relational database engine.",
                "text/plain",
            )
        },
        data={"workspace_id": ws_id},
    )

    data = {"workspace_id": ws_id, "query": "What database does localRAGvault use?", "top_k": 2}
    response = client.post("/search/", json=data)

    assert response.status_code == 200
    result = response.json()
    assert result["workspace_id"] == ws_id
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0

    # Verify metadata contracts
    first_hit = result["results"][0]
    assert "chunk_index" in first_hit
    assert isinstance(first_hit["chunk_index"], int)
    assert first_hit["chunk_index"] >= 1


# =====================================================================
# --- 6. RAG PIPELINE & THREAD MANAGEMENT TESTS ---
# =====================================================================


@patch("core.api.generate_answer")
def test_ask_question_with_valid_context(mock_generate) -> None:
    """Test full pipeline loop under mocked conditions returning successful answers."""
    ws_id = _create_test_workspace("Mock RAG Pipeline WS")

    file_content = b"localRAGvault is a privacy-first, fully local RAG architecture."
    client.post(
        "/upload/",
        files={"file": ("test_doc.txt", file_content, "text/plain")},
        data={"workspace_id": ws_id},
    )

    mock_generate.return_value = LLMInternalResponse(
        text="This is a mocked LLM answer for testing.", is_valid=True
    )

    data = {"workspace_id": ws_id, "query": "What is localRAGvault?", "top_k": 2}
    response = client.post("/ask/", json=data)

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "This is a mocked LLM answer for testing."
    assert len(result["sources"]) > 0
    assert result["sources"][0]["chunk_index"] >= 1
    assert "thread_id" in result


@patch("core.api.generate_answer")
def test_ask_question_hides_sources_on_invalid_context(mock_generate) -> None:
    """Verify that sources are wiped from the output package if the LLM cannot answer."""
    ws_id = _create_test_workspace("Hide Sources Test WS")

    file_content = b"Some unrelated context about apples and oranges."
    client.post(
        "/upload/",
        files={"file": ("unrelated.txt", file_content, "text/plain")},
        data={"workspace_id": ws_id},
    )

    mock_generate.return_value = LLMInternalResponse(
        text="I cannot answer this based on the provided documents.", is_valid=False
    )

    data = {"workspace_id": ws_id, "query": "Random query that doesn't exist", "top_k": 1}
    response = client.post("/ask/", json=data)

    assert response.status_code == 200
    result = response.json()
    assert result["sources"] == []


@patch("core.api.generate_answer")
def test_delete_thread_endpoint(mock_generate) -> None:
    """Verify thread deletion removes the conversation history."""
    ws_id = _create_test_workspace("Thread Deletion WS")

    mock_generate.return_value = LLMInternalResponse(text="Mock reply.", is_valid=True)

    # 1. Ask a question to generate a thread
    ask_res = client.post("/ask/", json={"workspace_id": ws_id, "query": "Hello?", "top_k": 1})
    thread_id = ask_res.json()["thread_id"]

    # 2. Verify thread exists in list
    threads_res = client.get(f"/workspaces/{ws_id}/threads")
    assert len(threads_res.json()["threads"]) == 1

    # 3. Delete the thread
    del_res = client.delete(f"/threads/{thread_id}")
    assert del_res.status_code == 200

    # 4. Verify thread list is now empty
    threads_after = client.get(f"/workspaces/{ws_id}/threads")
    assert len(threads_after.json()["threads"]) == 0


@pytest.mark.integration
def test_ask_question_real_model() -> None:
    """Test the integration loop directly against working local Ollama model spaces."""
    ws_id = _create_test_workspace("Integration Test WS")

    file_content = b"localRAGvault is a privacy-first, fully local RAG architecture. It uses Ollama to serve generation."
    files = {"file": ("integration_doc.txt", file_content, "text/plain")}
    client.post("/upload/", files=files, data={"workspace_id": ws_id})

    data = {
        "workspace_id": ws_id,
        "query": "What is localRAGvault's privacy approach?",
        "top_k": 1,
    }

    response = client.post("/ask/", json=data)

    assert response.status_code == 200
    result = response.json()

    assert (
        "local" in result["answer"].lower()
        or "privacy" in result["answer"].lower()
        or "ollama" in result["answer"].lower()
    )
    assert len(result["sources"]) > 0
    assert result["sources"][0]["filename"] == "integration_doc.txt"
    assert result["sources"][0]["chunk_index"] >= 1

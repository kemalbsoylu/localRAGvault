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

    client.post(
        "/upload/",
        files={"file": ("dummy.txt", b"temporary content", "text/plain")},
        data={"workspace_id": ws_id},
    )

    del_res = client.delete(f"/workspaces/{ws_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    inv_res = client.get(f"/inventory/{ws_id}")
    assert inv_res.status_code == 404


# =====================================================================
# --- 3. WORKSPACE SETTINGS PATCH & VALIDATION TESTS ---
# =====================================================================


def test_patch_workspace_settings_success() -> None:
    """Test dynamically updating chunking physics, retrieval depth, and system instructions."""
    ws_id = _create_test_workspace("Settings Patch WS")
    updates = {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "top_k": 8,
        "similarity_threshold": 0.25,
        "chat_history_limit": 10,
        "system_prompt": "Always respond using concise technical bullet points.",
    }
    res = client.patch(f"/workspaces/{ws_id}", json=updates)
    assert res.status_code == 200
    data = res.json()
    assert data["chunk_size"] == 800
    assert data["chunk_overlap"] == 150
    assert data["top_k"] == 8
    assert round(data["similarity_threshold"], 2) == 0.25
    assert data["chat_history_limit"] == 10
    assert data["system_prompt"] == "Always respond using concise technical bullet points."


def test_patch_workspace_overlap_validation() -> None:
    """Verify setting chunk_overlap >= chunk_size triggers a strict HTTP 400/422 rejection."""
    ws_id = _create_test_workspace("Overlap Validation WS")
    res = client.patch(f"/workspaces/{ws_id}", json={"chunk_size": 500, "chunk_overlap": 500})
    assert res.status_code in (400, 422)


def test_create_workspace_overlap_validation() -> None:
    """Verify Pydantic schema validator blocks invalid overlap bounds during workspace creation."""
    payload = {
        "name": "Invalid Overlap WS",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": 400,
        "chunk_overlap": 450,
    }
    res = client.post("/workspaces/", json=payload)
    assert res.status_code == 422


# =====================================================================
# --- 4. DOCUMENT INGESTION, MULTI-FORMAT & UPSERT TESTS ---
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

    files = {"file": ("versioned.txt", b"Version 1 content string.", "text/plain")}
    res1 = client.post("/upload/", files=files, data={"workspace_id": ws_id})
    assert res1.status_code == 200
    assert res1.json()["is_upsert"] is False
    initial_chunks = res1.json()["chunks_saved"]

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
# --- 5. DOCUMENT DELETION & INVENTORY TESTS ---
# =====================================================================


def test_workspace_inventory_and_deletion() -> None:
    """Verify inventory tracking and single document deletion loops."""
    ws_id = _create_test_workspace("Inventory Tracking WS")

    client.post(
        "/upload/",
        files={"file": ("to_delete.txt", b"Delete me soon.", "text/plain")},
        data={"workspace_id": ws_id},
    )

    inv_res = client.get(f"/inventory/{ws_id}")
    assert inv_res.status_code == 200
    assert len(inv_res.json()["documents"]) == 1

    del_res = client.delete(f"/documents/{ws_id}/to_delete.txt")
    assert del_res.status_code == 200
    assert del_res.json()["chunks_deleted"] > 0

    inv_res_after = client.get(f"/inventory/{ws_id}")
    assert len(inv_res_after.json()["documents"]) == 0


# =====================================================================
# --- 6. VECTOR SEARCH & CHUNK METADATA TESTS ---
# =====================================================================


def test_search_documents_response_contract() -> None:
    """Test vector search functionality, verifying chunk_index is returned."""
    ws_id = _create_test_workspace("Search Contract Test WS")

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

    first_hit = result["results"][0]
    assert "chunk_index" in first_hit
    assert isinstance(first_hit["chunk_index"], int)
    assert first_hit["chunk_index"] >= 1


# =====================================================================
# --- 7. RAG PIPELINE & THREAD MANAGEMENT TESTS ---
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

    ask_res = client.post("/ask/", json={"workspace_id": ws_id, "query": "Hello?", "top_k": 1})
    thread_id = ask_res.json()["thread_id"]

    threads_res = client.get(f"/workspaces/{ws_id}/threads")
    assert len(threads_res.json()["threads"]) == 1

    del_res = client.delete(f"/threads/{thread_id}")
    assert del_res.status_code == 200

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


# =====================================================================
# --- 8. BATCH INGESTION & MULTI-STREAM TESTS ---
# =====================================================================


def test_upload_batch_success_and_upsert() -> None:
    """Verify multi-file batch ingestion correctly tracks new saves vs clean upserts."""
    ws_id = _create_test_workspace("Batch Success WS")

    files_batch = [
        ("files", ("doc_alpha.txt", b"Alpha document content.", "text/plain")),
        ("files", ("doc_beta.csv", b"id,val\n1,100\n2,200", "text/csv")),
    ]
    res1 = client.post(
        "/upload/batch/",
        files=files_batch,
        data={"workspace_id": ws_id, "embedding_model": DEFAULT_EMBEDDING_MODEL},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["summary"]["total_files"] == 2
    assert data1["summary"]["successful"] == 2
    assert data1["summary"]["upserts"] == 0
    assert data1["summary"]["failed"] == 0

    files_reupload = [
        ("files", ("doc_alpha.txt", b"Alpha document updated content.", "text/plain")),
    ]
    res2 = client.post(
        "/upload/batch/",
        files=files_reupload,
        data={"workspace_id": ws_id, "embedding_model": DEFAULT_EMBEDDING_MODEL},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["summary"]["successful"] == 0
    assert data2["summary"]["upserts"] == 1
    assert data2["results"][0]["status"] == "upserted"


def test_upload_batch_mixed_validity_and_unnamed_streams() -> None:
    """Verify batch ingestion isolates failures without halting valid streams in the array."""
    ws_id = _create_test_workspace("Batch Mixed Validity WS")

    files_mixed = [
        ("files", ("valid_doc.txt", b"Valid text stream.", "text/plain")),
        ("files", ("malicious.exe", b"binary executable payload", "application/octet-stream")),
        ("files", ("   ", b"unnamed stream bytes", "text/plain")),
    ]
    res = client.post(
        "/upload/batch/",
        files=files_mixed,
        data={"workspace_id": ws_id, "embedding_model": DEFAULT_EMBEDDING_MODEL},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["total_files"] == 3
    assert data["summary"]["successful"] == 1
    assert data["summary"]["failed"] == 2

    failed_reasons = [r["error_message"] for r in data["results"] if r["status"] == "failed"]
    assert any("Unsupported file type" in msg for msg in failed_reasons if msg)
    assert any("Missing or empty filename" in msg for msg in failed_reasons if msg)


# =====================================================================
# --- 9. CONVERSATION HISTORY & THREAD LISTING TESTS ---
# =====================================================================


@patch("core.api.generate_answer")
def test_list_threads_and_get_message_history(mock_generate) -> None:
    """Verify thread list cards and chronological message turn retrieval."""
    ws_id = _create_test_workspace("History Tracking WS")

    client.post(
        "/upload/",
        files={"file": ("dummy.txt", b"dummy context", "text/plain")},
        data={"workspace_id": ws_id},
    )

    mock_generate.return_value = LLMInternalResponse(
        text="Mocked historical response.", is_valid=True
    )

    ask1 = client.post(
        "/ask/", json={"workspace_id": ws_id, "query": "First question?", "top_k": 1}
    )
    thread_id = ask1.json()["thread_id"]
    client.post(
        "/ask/",
        json={
            "workspace_id": ws_id,
            "query": "Second follow-up?",
            "thread_id": thread_id,
            "top_k": 1,
        },
    )

    threads_res = client.get(f"/workspaces/{ws_id}/threads")
    assert threads_res.status_code == 200
    t_data = threads_res.json()
    assert t_data["workspace_id"] == ws_id
    assert len(t_data["threads"]) == 1
    assert t_data["threads"][0]["message_count"] == 4
    assert t_data["threads"][0]["last_query"] == "Second follow-up?"

    msgs_res = client.get(f"/threads/{thread_id}/messages")
    assert msgs_res.status_code == 200
    m_data = msgs_res.json()
    assert m_data["thread_id"] == thread_id
    assert len(m_data["messages"]) == 4
    assert m_data["messages"][0]["role"] == "user"
    assert m_data["messages"][0]["content"] == "First question?"
    assert m_data["messages"][1]["role"] == "assistant"
    assert m_data["messages"][3]["content"] == "Mocked historical response."


# =====================================================================
# --- 10. MODEL MISMATCH & DEFENSIVE BOUNDARY TESTS ---
# =====================================================================


def test_model_mismatch_safeguards() -> None:
    """Verify strict 400 rejection when request models violate locked workspace dimensions."""
    ws_id = _create_test_workspace("Locked Model WS")
    different_model = "differentmodel:latest"

    up_res = client.post(
        "/upload/",
        files={"file": ("test.txt", b"content", "text/plain")},
        data={"workspace_id": ws_id, "embedding_model": different_model},
    )
    assert up_res.status_code == 400
    assert "permanently locked" in up_res.json()["detail"]

    search_res = client.post(
        "/search/",
        json={"workspace_id": ws_id, "query": "test", "embedding_model": different_model},
    )
    assert search_res.status_code == 400
    assert "locked model" in search_res.json()["detail"]

    ask_res = client.post(
        "/ask/",
        json={"workspace_id": ws_id, "query": "test", "embedding_model": different_model},
    )
    assert ask_res.status_code == 400
    assert "must match workspace" in ask_res.json()["detail"]


def test_not_found_safeguards() -> None:
    """Verify proper 404 HTTP exceptions for nonexistent resources across all read/write routes."""
    fake_id = "nonexistent-uuid-9999"

    assert client.get(f"/inventory/{fake_id}").status_code == 404
    assert client.get(f"/workspaces/{fake_id}/threads").status_code == 404
    assert client.get(f"/threads/{fake_id}/messages").status_code == 404
    assert client.delete(f"/workspaces/{fake_id}").status_code == 404
    assert client.delete(f"/documents/{fake_id}/some_file.txt").status_code == 404
    assert client.delete(f"/threads/{fake_id}").status_code == 404

    up_res = client.post(
        "/upload/",
        files={"file": ("test.txt", b"data", "text/plain")},
        data={"workspace_id": fake_id},
    )
    assert up_res.status_code == 404

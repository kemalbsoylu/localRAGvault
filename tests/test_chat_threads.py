from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.api import app
from core.config import DEFAULT_EMBEDDING_MODEL
from core.schemas import LLMInternalResponse

client = TestClient(app)


def _create_test_workspace(name: str) -> str:
    response = client.post(
        "/workspaces/", json={"name": name, "embedding_model": DEFAULT_EMBEDDING_MODEL}
    )
    return response.json()["id"]


@patch("core.api.generate_answer")
def test_ask_question_with_valid_context(mock_generate) -> None:
    ws_id = _create_test_workspace("Mock RAG Pipeline WS")
    client.post(
        "/upload/",
        files={
            "file": ("test_doc.txt", b"localRAGvault is a local RAG architecture.", "text/plain")
        },
        data={"workspace_id": ws_id},
    )

    mock_generate.return_value = LLMInternalResponse(text="Mocked LLM answer.", is_valid=True)
    response = client.post(
        "/ask/", json={"workspace_id": ws_id, "query": "What is it?", "top_k": 2}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "Mocked LLM answer."
    assert "thread_id" in result


@patch("core.api.generate_answer")
def test_ask_question_hides_sources_on_invalid_context(mock_generate) -> None:
    ws_id = _create_test_workspace("Hide Sources Test WS")
    client.post(
        "/upload/",
        files={"file": ("unrelated.txt", b"Some unrelated context.", "text/plain")},
        data={"workspace_id": ws_id},
    )

    mock_generate.return_value = LLMInternalResponse(
        text="I cannot answer this based on the provided documents.", is_valid=False
    )
    response = client.post(
        "/ask/", json={"workspace_id": ws_id, "query": "Random query", "top_k": 1}
    )

    assert response.status_code == 200
    assert response.json()["sources"] == []


# =====================================================================
# --- EXHAUSTIVE THREAD RENAMING TESTS ---
# =====================================================================


@patch("core.api.generate_answer")
def test_rename_thread_success(mock_generate) -> None:
    """Verify renaming a thread successfully updates its title and returns a valid contract."""
    ws_id = _create_test_workspace("Rename Success WS")
    mock_generate.return_value = LLMInternalResponse(text="Reply.", is_valid=True)

    ask_res = client.post("/ask/", json={"workspace_id": ws_id, "query": "Hello?", "top_k": 1})
    thread_id = ask_res.json()["thread_id"]

    patch_res = client.patch(
        f"/threads/{thread_id}", json={"title": "Important Financial Discussion"}
    )
    assert patch_res.status_code == 200

    data = patch_res.json()
    assert data["id"] == thread_id
    assert data["title"] == "Important Financial Discussion"

    # Verify persistence via thread listing
    threads_res = client.get(f"/workspaces/{ws_id}/threads")
    assert threads_res.json()["threads"][0]["title"] == "Important Financial Discussion"


@patch("core.api.generate_answer")
def test_rename_thread_whitespace_or_empty_rejected(mock_generate) -> None:
    """Verify sending empty strings or whitespace-only titles is rejected by Pydantic validation."""
    ws_id = _create_test_workspace("Rename Whitespace WS")
    mock_generate.return_value = LLMInternalResponse(text="Reply.", is_valid=True)
    thread_id = client.post(
        "/ask/", json={"workspace_id": ws_id, "query": "Hello?", "top_k": 1}
    ).json()["thread_id"]

    res_empty = client.patch(f"/threads/{thread_id}", json={"title": ""})
    assert res_empty.status_code == 422

    res_spaces = client.patch(f"/threads/{thread_id}", json={"title": "     "})
    assert res_spaces.status_code == 422


@patch("core.api.generate_answer")
def test_rename_thread_too_long_rejected(mock_generate) -> None:
    """Verify sending a title over 100 characters triggers a schema validation rejection."""
    ws_id = _create_test_workspace("Rename Length WS")
    mock_generate.return_value = LLMInternalResponse(text="Reply.", is_valid=True)
    thread_id = client.post(
        "/ask/", json={"workspace_id": ws_id, "query": "Hello?", "top_k": 1}
    ).json()["thread_id"]

    long_title = "A" * 101
    res_long = client.patch(f"/threads/{thread_id}", json={"title": long_title})
    assert res_long.status_code == 422


def test_rename_thread_not_found() -> None:
    """Verify attempting to rename a non-existent thread ID returns a clean 404 error."""
    res = client.patch("/threads/nonexistent-uuid-9999", json={"title": "New Title"})
    assert res.status_code == 404


# =====================================================================
# --- DELETION & INTEGRATION TESTS ---
# =====================================================================


@patch("core.api.generate_answer")
def test_delete_thread_endpoint(mock_generate) -> None:
    ws_id = _create_test_workspace("Thread Deletion WS")
    mock_generate.return_value = LLMInternalResponse(text="Mock reply.", is_valid=True)
    thread_id = client.post(
        "/ask/", json={"workspace_id": ws_id, "query": "Hello?", "top_k": 1}
    ).json()["thread_id"]

    assert client.delete(f"/threads/{thread_id}").status_code == 200
    assert len(client.get(f"/workspaces/{ws_id}/threads").json()["threads"]) == 0


@pytest.mark.integration
def test_ask_question_real_model() -> None:
    ws_id = _create_test_workspace("Integration Test WS")
    client.post(
        "/upload/",
        files={
            "file": (
                "doc.txt",
                b"localRAGvault uses Ollama to serve generation locally.",
                "text/plain",
            )
        },
        data={"workspace_id": ws_id},
    )

    response = client.post(
        "/ask/", json={"workspace_id": ws_id, "query": "What serves generation?", "top_k": 1}
    )
    assert response.status_code == 200
    assert len(response.json()["sources"]) > 0

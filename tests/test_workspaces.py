import pytest
from fastapi.testclient import TestClient

from core.api import app
from core.config import DEFAULT_EMBEDDING_MODEL

client = TestClient(app)


def _create_test_workspace(name: str) -> str:
    response = client.post(
        "/workspaces/", json={"name": name, "embedding_model": DEFAULT_EMBEDDING_MODEL}
    )
    if response.status_code != 200:
        pytest.fail(f"Failed to create test workspace: {response.text}")
    return response.json()["id"]


def test_health_check() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "localRAGvault API is running securely.",
    }


def test_list_models_endpoint() -> None:
    response = client.get("/models/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert isinstance(data.get("models"), list)


def test_create_workspace_endpoint() -> None:
    response = client.post(
        "/workspaces/", json={"name": "Alpha Workspace", "embedding_model": DEFAULT_EMBEDDING_MODEL}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Alpha Workspace"
    assert "id" in data
    assert data["dimension"] > 0


def test_delete_workspace_endpoint() -> None:
    ws_id = _create_test_workspace("To Be Deleted WS")
    del_res = client.delete(f"/workspaces/{ws_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

    inv_res = client.get(f"/inventory/{ws_id}")
    assert inv_res.status_code == 404


def test_patch_workspace_settings_success() -> None:
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
    ws_id = _create_test_workspace("Overlap Validation WS")
    res = client.patch(f"/workspaces/{ws_id}", json={"chunk_size": 500, "chunk_overlap": 500})
    assert res.status_code in (400, 422)


def test_create_workspace_overlap_validation() -> None:
    payload = {
        "name": "Invalid Overlap WS",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "chunk_size": 400,
        "chunk_overlap": 450,
    }
    res = client.post("/workspaces/", json=payload)
    assert res.status_code == 422


def test_model_mismatch_safeguards() -> None:
    ws_id = _create_test_workspace("Locked Model WS")
    different_model = "differentmodel:latest"

    up_res = client.post(
        "/upload/",
        files={"file": ("test.txt", b"content", "text/plain")},
        data={"workspace_id": ws_id, "embedding_model": different_model},
    )
    assert up_res.status_code == 400
    assert "permanently locked" in up_res.json()["detail"]

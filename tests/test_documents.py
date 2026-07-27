import json

from fastapi.testclient import TestClient

from core.api import app
from core.config import DEFAULT_EMBEDDING_MODEL

client = TestClient(app)


def _create_test_workspace(name: str) -> str:
    response = client.post(
        "/workspaces/", json={"name": name, "embedding_model": DEFAULT_EMBEDDING_MODEL}
    )
    return response.json()["id"]


def test_upload_document_txt_success() -> None:
    ws_id = _create_test_workspace("Txt Upload WS")
    files = {
        "file": ("test_doc.txt", b"This is a test document tracking parsing loops.", "text/plain")
    }
    response = client.post("/upload/", files=files, data={"workspace_id": ws_id})

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["chunks_saved"] > 0
    assert res_data["is_upsert"] is False


def test_upload_document_csv_and_json_success() -> None:
    ws_id = _create_test_workspace("Structured Upload WS")

    csv_files = {"file": ("staff.csv", b"id,name\n101,Alice\n102,Bob", "text/csv")}
    assert client.post("/upload/", files=csv_files, data={"workspace_id": ws_id}).status_code == 200

    json_payload = [{"project": "localRAGvault", "version": "0.1.0"}]
    json_files = {
        "file": ("config.json", json.dumps(json_payload).encode("utf-8"), "application/json")
    }
    assert (
        client.post("/upload/", files=json_files, data={"workspace_id": ws_id}).status_code == 200
    )


def test_upload_clean_upsert() -> None:
    ws_id = _create_test_workspace("Upsert WS")
    files = {"file": ("versioned.txt", b"Version 1 content string.", "text/plain")}

    res1 = client.post("/upload/", files=files, data={"workspace_id": ws_id})
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
    ws_id = _create_test_workspace("Invalid File Test WS")
    files = {"file": ("malicious.exe", b"binary executable payload", "application/octet-stream")}
    response = client.post("/upload/", files=files, data={"workspace_id": ws_id})

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_workspace_inventory_and_deletion() -> None:
    ws_id = _create_test_workspace("Inventory Tracking WS")
    client.post(
        "/upload/",
        files={"file": ("to_delete.txt", b"Delete me soon.", "text/plain")},
        data={"workspace_id": ws_id},
    )

    assert len(client.get(f"/inventory/{ws_id}").json()["documents"]) == 1

    del_res = client.delete(f"/documents/{ws_id}/to_delete.txt")
    assert del_res.status_code == 200
    assert del_res.json()["chunks_deleted"] > 0
    assert len(client.get(f"/inventory/{ws_id}").json()["documents"]) == 0


def test_upload_batch_success_and_upsert() -> None:
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
    assert res1.json()["summary"]["successful"] == 2

    files_reupload = [
        ("files", ("doc_alpha.txt", b"Alpha document updated content.", "text/plain"))
    ]
    res2 = client.post(
        "/upload/batch/",
        files=files_reupload,
        data={"workspace_id": ws_id, "embedding_model": DEFAULT_EMBEDDING_MODEL},
    )
    assert res2.json()["summary"]["upserts"] == 1


def test_upload_batch_mixed_validity_and_unnamed_streams() -> None:
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
    assert data["summary"]["successful"] == 1
    assert data["summary"]["failed"] == 2


def test_search_documents_response_contract() -> None:
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

    response = client.post(
        "/search/", json={"workspace_id": ws_id, "query": "database engine", "top_k": 2}
    )
    assert response.status_code == 200
    result = response.json()
    assert len(result["results"]) > 0
    assert "chunk_index" in result["results"][0]

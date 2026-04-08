import io
from unittest.mock import patch

import pytest
pytestmark = pytest.mark.skip(reason="disabled in CI")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.mark.skip(reason="disabled as it needs genai api key")
def test_upload_files_success():
    mock_result = ["file1_embedded", "file2_embedded"]

    with patch(
            "app.routers.routes_ingest.ingest_and_store_embedding",
            return_value=mock_result,
    ):
        files = [
            ("files", ("a.txt", io.BytesIO(b"hello"), "text/plain")),
            ("files", ("b.txt", io.BytesIO(b"world"), "text/plain")),
        ]

        resp = client.post("/api/ingest/upload", files=files)

    assert resp.status_code == 200
    assert resp.json() == mock_result


@pytest.mark.skip(reason="disabled as it needs genai api key")
def test_upload_files_failure():
    with patch(
            "app.routers.routes_ingest.ingest_and_store_embedding",
            side_effect=Exception("boom"),
    ):
        files = [("files", ("a.txt", io.BytesIO(b"x"), "text/plain"))]

        resp = client.post("/api/ingest/upload", files=files)

    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


@pytest.mark.skip(reason="disabled as it needs genai api key")
def test_delete_collection_success():
    with patch(
            "app.routers.routes_ingest.delete_store_embedding", return_value=True
    ):
        resp = client.post("/api/ingest/delete?collection_name=testcol")

    assert resp.status_code == 200
    assert resp.json() == "deleted collection testcol"


@pytest.mark.skip(reason="disabled as it needs genai api key")
def test_delete_collection_failure():
    with patch(
            "app.routers.routes_ingest.delete_store_embedding",
            side_effect=Exception("delete error"),
    ):
        resp = client.post("/api/ingest/delete?collection_name=abc")

    assert resp.status_code == 500
    assert "delete error" in resp.json()["detail"]


@pytest.mark.skip(reason="disabled as it needs genai api key")
def test_list_collections_success():
    mock_cols = ["col1", "col2"]

    with patch(
            "app.routers.routes_ingest.list_collections_chroma",
            return_value=mock_cols,
    ):
        resp = client.get("/api/ingest/collections")

    assert resp.status_code == 200
    assert resp.json() == mock_cols

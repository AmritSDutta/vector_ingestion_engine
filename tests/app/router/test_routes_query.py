import pytest
pytestmark = pytest.mark.skip(reason="disabled in CI")

from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
import importlib

query_mod = importlib.import_module("app.routers.routes_query")

client = TestClient(app)


@pytest.mark.skip(reason="disabled as it needs genai api key")
def test_search_docs_success():
    mock_answer = {"answer": "ok"}

    with patch.object(query_mod, "sanitize_passage", return_value=None), \
         patch.object(query_mod, "query_handler", return_value=mock_answer, autospec=True):

        resp = client.post(
            "/api/query/analyse",
            json={"q": "hello", "top_k": 2},
        )

    assert resp.status_code == 200
    assert resp.json() == mock_answer


@pytest.mark.skip(reason="disabled as it needs genai api key")
def test_search_docs_validation_error():
    """Missing required field q → 422."""
    resp = client.post("/api/query/analyse", json={"top_k": 5})
    assert resp.status_code == 422

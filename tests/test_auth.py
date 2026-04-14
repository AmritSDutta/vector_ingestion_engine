import os
from unittest.mock import patch, AsyncMock

# Set test environment variables before importing app to avoid collection errors
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_TOKEN"] = "test-token"
os.environ["QDRANT_HOST"] = "http://localhost:6333"
os.environ["QDRANT_API_KEY"] = "test-key"

from fastapi.testclient import TestClient
from app.main import app
from app.config.config import get_settings

client = TestClient(app)

def test_health_endpoint_is_public():


    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"health": "Server in fine health"}


def test_api_route_requires_auth():
    # Attempting to access an /api route should fail
    response = client.post("/api/query/analyse", json={"q": "test", "top_k": 3})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Basic"


def test_api_route_with_invalid_credentials():
    response = client.post(
        "/api/query/analyse",
        json={"q": "test", "top_k": 3},
        auth=("wrong", "credentials")
    )
    assert response.status_code == 401


@patch("app.routers.routes_query.query_handler", new_callable=AsyncMock)
@patch("app.routers.routes_query.sanitize_passage", new_callable=AsyncMock)
def test_api_route_with_valid_credentials(mock_sanitize, mock_query):
    mock_query.return_value = {"results": []}
    settings = get_settings()

    response = client.post(
        "/api/query/analyse",
        json={"q": "test", "top_k": 3},
        auth=(settings.API_USERNAME, settings.API_PASSWORD)
    )

    assert response.status_code == 200
    mock_query.assert_called_once()

@patch("app.routers.routes_query.query_handler", new_callable=AsyncMock)
@patch("app.routers.routes_query.sanitize_passage", new_callable=AsyncMock)
def test_api_route_auth_disabled(mock_sanitize, mock_query):
    mock_query.return_value = {"results": []}
    settings = get_settings()
    
    # Manually disable auth
    original_val = settings.IS_AUTH_ENABLED
    settings.IS_AUTH_ENABLED = False
    
    try:
        response = client.post(
            "/api/query/analyse", 
            json={"q": "test", "top_k": 3}
            # No auth provided
        )
        assert response.status_code == 200
    finally:
        # Restore settings
        settings.IS_AUTH_ENABLED = original_val


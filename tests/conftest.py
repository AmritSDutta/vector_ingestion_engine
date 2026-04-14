import os
from unittest.mock import MagicMock

os.environ.setdefault("QDRANT_HOST", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-api-key")
os.environ.setdefault("MILVUS_URI", "http://localhost:19530")
os.environ.setdefault("MILVUS_TOKEN", "test-token")

# Mock fastapi-limiter to avoid 429s in tests
import fastapi_limiter.depends
fastapi_limiter.depends.RateLimiter = MagicMock(return_value=lambda: None)



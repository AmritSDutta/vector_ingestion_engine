import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from .request_validator import sanitize_passage
from ..services.query_service import query_handler, hybrid_query_handler

logger = logging.getLogger(__name__)
query_router = APIRouter(prefix="/query", tags=["query"])


class QueryIn(BaseModel):
    q: str = Field(max_length=500, min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

    @field_validator('q')
    @classmethod
    def validate_query_content(cls, v) -> str:
        # Strip whitespace
        v = v.strip()
        if not v:
            raise ValueError('Query cannot be empty or whitespace')
        return v


@query_router.post("/analyse", status_code=200, response_model=dict[str, list[Any]])
async def search_docs(query: QueryIn) -> dict[str, list[Any]]:
    """
    Retrieve items by category with an optional limit.
    """
    logging.info(f'received req {query.q}')
    sanitize_passage(query.q)
    ans: dict[str, list[Any]] = await query_handler(query.q, query.top_k)
    return ans


@query_router.post("/hybrid_analyse", status_code=200, response_model=dict[str, list[Any]])
async def hybrid_search_docs(query: QueryIn) -> dict[str, list[Any]]:
    """
    Retrieve items by category with an optional limit.
    """
    logging.info(f'received req: {query.q}')
    sanitize_passage(query.q)
    ans: dict[str, list[Any]] = await hybrid_query_handler(query.q, query.top_k)
    return ans

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from .request_validator import sanitize_passage
from ..services.query_service import query_handler, hybrid_query_handler

logger = logging.getLogger(__name__)
query_router = APIRouter(prefix="/query", tags=["query"])


class QueryIn(BaseModel):
    q: str
    top_k: int = 3


@query_router.post("/analyse", status_code=200, response_model=dict[str, list[Any]])
async def search_docs(query: QueryIn) -> dict[str, list[Any]]:
    """
    Retrieve items by category with an optional limit.
    """
    logging.info(f'received req: {query}')
    sanitize_passage(query.q)
    ans: dict[str, list[Any]] = await query_handler(query.q, query.top_k)
    return ans


@query_router.post("/hybrid_analyse", status_code=200, response_model=dict[str, list[Any]])
async def hybrid_search_docs(query: QueryIn) -> dict[str, list[Any]]:
    """
    Retrieve items by category with an optional limit.
    """
    logging.info(f'received req: {query}')
    sanitize_passage(query.q)
    ans: dict[str, list[Any]] = await hybrid_query_handler(query.q, query.top_k)
    return ans

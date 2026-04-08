import logging

from fastapi import APIRouter
from pydantic import BaseModel

from .request_validator import sanitize_passage
from ..services.query_service import query_handler

query_router = APIRouter(prefix="/query", tags=["query"])


class QueryIn(BaseModel):
    q: str
    top_k: int = 3


@query_router.post("/analyse", status_code=200, response_model=dict[str, str])
async def search_docs(query: QueryIn) -> dict[str, str]:
    """
    Retrieve items by category with an optional limit.
    """
    logging.info(f'received req: {query}')
    sanitize_passage(query.q)
    ans: dict[str, str] = await query_handler(query.q, query.top_k)
    return ans

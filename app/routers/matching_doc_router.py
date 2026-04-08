import logging
from typing import List

from fastapi import APIRouter

logger = logging.getLogger(__name__)
doc_router = APIRouter(prefix="/docs", tags=["docs"])


@doc_router.get("/search", status_code=200, response_model=List[str])
async def search_docs() -> List[str]:
    """
    Retrieve items by category with an optional limit.
    """
    logger.info(f'received req: query')
    docs: List[str] = ['working']
    return docs

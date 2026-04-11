import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from pyrate_limiter import Limiter, Rate, Duration

from app.services.ingest_service import ingest_and_store_embedding, _get_vector_store
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)
ingest_router = APIRouter(prefix="/ingest", tags=["ingestion"])


@ingest_router.post("/upload",
                    dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(1, Duration.MINUTE * 15))))])
async def upload_files(background_tasks: BackgroundTasks) -> dict[str, str]:
    """
    Accept multiple files, store them in a per-request temporary directory,
    run ingestion, and automatically clean up.
    """
    try:
        background_tasks.add_task(ingest_and_store_embedding)
        return {"message": "uploading started"}

    except Exception as exc:
        logging.error('file upload errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))


@ingest_router.post("/delete", response_model=str)
async def delete_collections(collection_name: Optional[str] = None) -> str:
    """
    delete vector store
    """
    try:
        vstore = _get_vector_store()
        response: Optional[str] = await vstore.delete_collection(collection_name)
        return f'deleted collection {response if response else 'None'}'

    except Exception as exc:
        logging.error('chroma collection delete errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))


@ingest_router.get("/collections", response_model=list[str],
                   dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(1, Duration.MINUTE * 5))))])
async def list_collections() -> list[str]:
    """
    list collections from vector store
    """
    try:
        vstore = _get_vector_store()
        return await vstore.list_collection()

    except Exception as exc:
        logging.error('collection list errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))

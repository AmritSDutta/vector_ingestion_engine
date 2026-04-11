import logging
from typing import Optional, Any

from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel
from pyrate_limiter import Limiter, Rate, Duration

from app.celery_worker import celery_app
from app.celery_task import ingest_task_wrapper
from app.services.ingest_service import _get_vector_store

logger = logging.getLogger(__name__)
ingest_router = APIRouter(prefix="/ingest", tags=["ingestion"])


class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None


@ingest_router.post("/upload",
                    dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(1, Duration.MINUTE * 15))))])
async def upload_files() -> dict[str, str]:
    """
    Accept multiple files, store them in a per-request temporary directory,
    run ingestion, and automatically clean up.
    """
    try:
        logger.info("Received request to upload files and start ingestion.")
        task = ingest_task_wrapper.delay()
        logger.info(f"Ingestion task triggered successfully with task_id: {task.id}")
        return {"message": "uploading started", "task_id": task.id}

    except Exception as exc:
        logger.error(f"Error triggering file upload and ingestion: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@ingest_router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    from celery.result import AsyncResult
    logger.info(f"Checking status for task_id: {task_id}")
    res = AsyncResult(task_id, app=celery_app)
    status_info = {
        "task_id": task_id,
        "status": res.status,
        "result": res.result if res.ready() else res.info
    }
    logger.info(f"Task status for {task_id}: {res.status}")
    return status_info


@ingest_router.post("/delete", response_model=str)
async def delete_collections(collection_name: Optional[str] = None) -> str:
    """
    delete vector store
    """
    try:
        vstore = _get_vector_store()
        response: Optional[str] = await vstore.delete_collection(collection_name)
        return f'deleted collection {response if response else "None"}'

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

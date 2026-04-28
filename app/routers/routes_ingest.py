import asyncio
import logging
from typing import Optional, Any

from celery import chain, group
from fastapi import APIRouter, HTTPException, Depends
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel
from pyrate_limiter import Limiter, Rate, Duration

from app.celery_task import (
    ingest_task_wrapper,
    prepare_data_task,
    ingest_single_db_task
)
from app.celery_worker import celery_app
from app.services.vector_store.VectorStoreFactory import get_vector_store, DatabaseType

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

        # Trigger the background poller
        asyncio.create_task(_poll_task_status(task.id))

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
        vstore = get_vector_store()
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
        vstore = get_vector_store()
        return await vstore.list_collection()

    except Exception as exc:
        logging.error('collection list errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))


async def _poll_task_status(task_id: str):
    """
    Background task to poll Celery for status and log it to the application console.
    """
    from celery.result import AsyncResult
    logger.info(f"Started background polling for Task ID: {task_id}")

    while True:
        res = AsyncResult(task_id, app=celery_app)
        status = res.status
        info = res.info if not res.ready() else res.result

        # Extract step information if available in meta
        step = info.get('step', 'N/A') if isinstance(info, dict) else 'N/A'
        logger.info(f"[Task Status reporter]: Task {task_id} | Status: {status} | Step: {step}")
        if res.ready():
            logger.info(f"[Task Status reporter]: Task {task_id} has finished with status: {status}")
            break
        await asyncio.sleep(5)


@ingest_router.post("/upload_to_all_database", response_model=dict,
                    dependencies=[Depends(RateLimiter(limiter=Limiter(Rate(1, Duration.MINUTE * 30))))])
async def upload_data_to_all() -> dict[str, str]:
    """
    Accept multiple files, store them in a per-request temporary directory,
    run ingestion, and automatically clean up.
    """
    try:
        logger.info("Received request to upload files and start ingestion to all DB.")

        # Build the Celery Canvas workflow:
        # 1. Prepare data (PII redaction + Embedding)
        # 2. Parallel ingestion into all supported databases
        workflow = chain(
            prepare_data_task.s(),
            group(ingest_single_db_task.s(db.name) for db in DatabaseType)
        )
        task = workflow.delay()

        logger.info(f"Ingestion workflow triggered successfully with task_id: {task.id}")

        # Trigger the background poller
        asyncio.create_task(_poll_task_status(task.id))

        return {"message": "uploading started to all DB", "task_id": task.id}

    except Exception as exc:
        logger.error(f"Error triggering file upload and ingestion: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

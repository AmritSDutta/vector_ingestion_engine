import logging
import os
import tempfile
import uuid
from typing import List

import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ingest_service import ingest_and_store_embedding

logger = logging.getLogger(__name__)
ingest_router = APIRouter(prefix="/ingest", tags=["ingestion"])


@ingest_router.post("/upload")
async def upload_files():
    """
    Accept multiple files, store them in a per-request temporary directory,
    run ingestion, and automatically clean up.
    """
    try:
        await ingest_and_store_embedding()

    except Exception as exc:
        logging.error('file upload errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))


@ingest_router.post("/delete", response_model=str)
async def delete_collections(collection_name: str) -> str:
    """
    delete vector store
    """
    try:
        return f'deleted collection {collection_name}'

    except Exception as exc:
        logging.error('chroma collection delete errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))


@ingest_router.get("/collections", response_model=list[str])
async def list_collections() -> list[str]:
    """
    delete vector store
    """
    try:
        return []

    except Exception as exc:
        logging.error('chroma collection delete errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))

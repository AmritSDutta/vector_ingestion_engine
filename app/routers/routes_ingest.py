import logging
import os
import tempfile
import uuid
from typing import List

import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ingest_service import ingest_and_store_embedding, delete_store_embedding, list_collections_chroma

logger = logging.getLogger(__name__)
ingest_router = APIRouter(prefix="/ingest", tags=["ingestion"])


@ingest_router.post("/upload", response_model=List[str])
async def upload_files(files: List[UploadFile] = File(...)) -> List[str]:
    """
    Accept multiple files, store them in a per-request temporary directory,
    run ingestion, and automatically clean up.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths: List[str] = []
            saved_files: List[UploadFile] = []

            for f in files:
                dest = os.path.join(tmpdir, f"{uuid.uuid4().hex}_{f.filename}")
                logging.info(f'stored in temp: {dest}')
                async with aiofiles.open(dest, "wb") as out:
                    while chunk := await f.read(64 * 1024):
                        await out.write(chunk)
                saved_paths.append(dest)

            # If sync and CPU-bound, optionally wrap in executor.
            result = ingest_and_store_embedding(saved_paths, f.filename)
            return result

    except Exception as exc:
        logging.error('file upload errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))


@ingest_router.post("/delete", response_model=str)
async def delete_collections(collection_name: str) -> str:
    """
    delete vector store
    """
    try:
        result = delete_store_embedding(collection_name)
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
        return list_collections_chroma()

    except Exception as exc:
        logging.error('chroma collection delete errors %s', exc)
        raise HTTPException(status_code=500, detail=str(exc))

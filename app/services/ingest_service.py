import logging
from typing import Any, Callable

import pandas as pd

from app.config.config import get_settings
from app.services.embedding.EmbeddingFactory import get_embedding_service
from app.services.vector_store.VectorStoreFactory import get_vector_store

logger = logging.getLogger(__name__)


async def _get_custom_embedding(texts: list[str]) -> list[Any]:
    """Generate a Gemini embedding for a given text."""
    import asyncio
    embedding_service = get_embedding_service()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embedding_service.embed_batch, texts)


def _get_vector_store():
    """Generate a Gemini embedding for a given text."""
    return get_vector_store()


async def ingest_and_store_embedding(progress_callback: Callable[[str], None] = None) -> dict:
    logging.info(f'uploaded files stored in : {get_settings().data_file_path}')
    if progress_callback:
        progress_callback('Loading Data')

    pd_data = pd.read_json(get_settings().data_file_path)
    data = pd_data.iloc[0:3].copy()
    logging.info(f"Processing rows {pd_data.columns} into DB")
    logging.info(f"Total rows selected from file: {len(data)}")

    texts_to_embed: list[str] = data["overall"].tolist()
    if progress_callback:
        progress_callback('Embedding started')

    embeddings: list[Any] = await _get_custom_embedding(texts_to_embed)
    data["embeddings"] = embeddings

    if progress_callback:
        progress_callback('Saving to Vector Store')

    vstore = _get_vector_store()
    await vstore.create()
    await vstore.save(data)
    logging.info(f'Successfully indexes built and stored in vector store, rows: {len(data)}')
    return {"status": "success", "rows": len(data)}

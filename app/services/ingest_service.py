import logging
from typing import Any, Callable
import pandas as pd
from app.config.config import get_settings
from app.services.embedding.EmbeddingFactory import get_embedding_service
from app.services.utils.text_cleaner import clean_text
from app.services.vector_store.VectorStoreFactory import get_vector_store, DatabaseType

logger = logging.getLogger(__name__)


async def _get_custom_embedding(texts: list[str]) -> list[Any]:
    """Generate a Gemini embedding for a given text."""
    import asyncio
    embedding_service = get_embedding_service()
    return await embedding_service.embed_batch(texts)


async def ingest_and_store_embedding(progress_callback: Callable[[str], None] = None) -> dict:
    logging.info(f'uploaded files stored in : {get_settings().data_file_path}')
    if progress_callback:
        progress_callback('Loading Data')

    pd_data = pd.read_json(get_settings().data_file_path)
    data = pd_data.iloc[:-1].copy()
    logging.info(f"Total rows selected from file: {len(data)}")

    texts_to_embed: list[str] = data["overall"].tolist()
    if progress_callback:
        progress_callback('Embedding started')

    logging.info("Starting embedding task")
    embeddings: list[Any] = await _get_custom_embedding(texts_to_embed)
    data["embeddings"] = embeddings

    if progress_callback:
        progress_callback('Saving to Vector Store')

    vstore = get_vector_store()
    await vstore.create()
    await vstore.save(data)
    logging.info(f'Successfully indexes built and stored in vector store, rows: {len(data)}')
    return {"status": "success", "rows": len(data)}


async def ingest_and_store_to_all_database(progress_callback: Callable[[str], None] = None) -> dict:
    logging.info(f'uploaded files stored in : {get_settings().data_file_path}')
    if progress_callback:
        progress_callback('Loading Data')

    from app.services.utils.pii_redaction import PII_Redactor
    pii_redactor = PII_Redactor()
    pd_data = pd.read_json(get_settings().data_file_path)
    data = pd_data.iloc[:-1].copy()
    logging.info(f"Total rows selected from file: {len(data)}")

    redacted_texts = await pii_redactor.do_pii_redaction_text(data["overall"].tolist())
    
    texts_to_embed = [clean_text(text) for text in redacted_texts]
    if progress_callback:
        progress_callback('Embedding started')

    logging.info("Starting embedding task")
    embeddings: list[Any] = await _get_custom_embedding(texts_to_embed)
    data["embeddings"] = embeddings

    for db in DatabaseType:
        logging.info(f"Trying: {db}, Name: {db.name}, Value: {db.value}")
        vstore = get_vector_store(db)
        await vstore.create()
        await vstore.save(data)
        logging.info(f"Completed saving: {db}, Name: {db.name}, Value: {db.value}")
        if progress_callback:
            progress_callback(f'Saved to Vector Store: {db.value}')

    logging.info(f'Successfully indexes built and stored in vector store, rows: {len(data)}')
    return {"status": "success", "rows": len(data)}

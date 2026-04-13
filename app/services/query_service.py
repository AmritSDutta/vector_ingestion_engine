import logging
from typing import Any

from app.services.embedding.EmbeddingFactory import get_embedding_service
from app.services.vector_store.VectorStoreFactory import get_vector_store
from app.services.utils.time_helper import time_coro

logger = logging.getLogger(__name__)


async def _get_custom_embedding(texts: list[str]):
    """Generate a custom embedding for a given text."""
    embedding_service = get_embedding_service()
    return await embedding_service.embed_batch(texts)


async def query_handler(user_query: str, top_k: int = 3) -> dict[str, list[Any]]:
    logging.info(f'received req: q_len={len(user_query)}')

    texts_to_embed: list[str] = [user_query]
    embeddings = await _get_custom_embedding(texts_to_embed)

    vstore = get_vector_store()
    response: dict[str, list[Any]] = await time_coro('simple_vector_search',
                                                     vstore.query(embeddings[0], n_results=top_k, query=user_query))
    logging.info(f'response keys={list(response.keys())}')
    return response


async def hybrid_query_handler(user_query: str, top_k: int = 3) -> dict[str, list[Any]]:
    logging.info(f'received req: q_len={len(user_query)}')

    texts_to_embed: list[str] = [user_query]
    embeddings = await time_coro('query_embedding', _get_custom_embedding(texts_to_embed))

    vstore = get_vector_store()
    response: dict[str, list[Any]] = await time_coro('hybrid_search',
                                                     vstore.hybrid_search(embeddings[0], n_results=top_k,
                                                                          query=user_query))
    logging.info(f'response keys={list(response.keys())}')
    return response

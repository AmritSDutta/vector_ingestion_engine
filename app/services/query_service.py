import logging
from typing import Any

from app.services.embedding.EmbeddingFactory import get_embedding_service
from app.services.vector_store.VectoreStoreFaactory import get_vector_store

logger = logging.getLogger(__name__)


def _get_custom_embedding(texts: list[str]):
    """Generate a custom embedding for a given text."""
    embedding_service = get_embedding_service()
    return embedding_service.embed_batch(texts)


async def query_handler(user_query: str, top_k: int = 20) -> dict[str, list[Any]]:
    logging.info(f'Query : {user_query}')

    texts_to_embed: list[str] = [user_query]
    embeddings = _get_custom_embedding(texts_to_embed)

    vstore = get_vector_store()
    response: dict[str, list[Any]] = await vstore.query(embeddings[0], n_results=top_k, query=user_query)
    logging.info(f'response: {response}')
    return response


async def hybrid_query_handler(user_query: str, top_k: int = 20) -> dict[str, list[Any]]:
    logging.info(f'Query : {user_query}')

    texts_to_embed: list[str] = [user_query]
    embeddings = _get_custom_embedding(texts_to_embed)

    vstore = get_vector_store()
    response: dict[str, list[Any]] = await vstore.hybrid_search(embeddings[0], n_results=top_k, query=user_query)
    logging.info(f'response: {response}')
    return response

import logging

from app.config.config import Settings
from app.services.vector_store.qdrant_vector_store import QdrantStore

_qdrantStore: QdrantStore | None = None


def get_vector_store():
    global _qdrantStore
    settings = Settings()
    logging.info(f'collection name to be used: {settings.collection_name}')

    if settings.VECTOR_STORE == "qdrant":
        if _qdrantStore is None:
            _qdrantStore = QdrantStore(settings.collection_name)
        return _qdrantStore  # persist in cloud

    raise RuntimeError("unsupported vector store")

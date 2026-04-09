import logging

from app.config.config import Settings
from app.services.vector_store.milvus_vector_store import MilvusStore
from app.services.vector_store.qdrant_vector_store import QdrantStore

logger = logging.getLogger(__name__)
_qdrantStore: QdrantStore | None = None
_milvusStore: MilvusStore | None = None


def get_vector_store():
    global _qdrantStore, _milvusStore
    settings = Settings()
    logging.info(f'collection name to be used: {settings.COLLECTION_NAME},'
                 f' and vector store will be used: {settings.VECTOR_STORE}')

    if settings.VECTOR_STORE == "qdrant":
        if _qdrantStore is None:
            _qdrantStore = QdrantStore()
        return _qdrantStore  # persist in cloud
    elif settings.VECTOR_STORE == "milvus":
        if _milvusStore is None:
            _milvusStore = MilvusStore()
        return _milvusStore

    raise RuntimeError("unsupported vector store")

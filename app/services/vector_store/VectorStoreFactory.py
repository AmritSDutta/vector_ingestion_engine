import logging
from enum import Enum
from typing import Optional

from app.config.config import Settings
from app.services.vector_store.milvus_vector_store import MilvusStore
from app.services.vector_store.qdrant_vector_store import QdrantStore
from app.services.vector_store.postgres_vector_store import PGVectorStore

logger = logging.getLogger(__name__)
_qdrantStore: QdrantStore | None = None
_milvusStore: MilvusStore | None = None
_pgStore: PGVectorStore | None = None


class DatabaseType(Enum):
    QDRANT = "qdrant"
    MILVUS = "milvus"
    POSTGRES = "postgres"


def get_vector_store(db_type: Optional[DatabaseType] = None):
    global _qdrantStore, _milvusStore, _pgStore
    settings = Settings()
    logging.info(f'collection name to be used: {settings.COLLECTION_NAME},'
                 f' and vector store will be used: {db_type if db_type else settings.VECTOR_STORE}')

    effective_db_type = db_type if db_type else DatabaseType(settings.VECTOR_STORE)

    if effective_db_type == DatabaseType.QDRANT:
        if _qdrantStore is None:
            _qdrantStore = QdrantStore()
        return _qdrantStore
    elif effective_db_type == DatabaseType.MILVUS:
        if _milvusStore is None:
            _milvusStore = MilvusStore()
        return _milvusStore
    elif effective_db_type == DatabaseType.POSTGRES:
        if _pgStore is None:
            _pgStore = PGVectorStore()
        return _pgStore

    raise RuntimeError("unsupported vector store")

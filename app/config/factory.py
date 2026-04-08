import logging
from typing import Optional

from app.config.config import Settings
from app.rag.embedder import GoogleEmbedder
from app.rag.qdrant_vector_store import QdrantStore
from app.rag.vector_store import ChromaStore

_qdrantStore: QdrantStore | None = None
_chromaStore: ChromaStore | None = None


def get_embedding_service():
    settings = Settings()
    if settings.EMBEDDER == "google":
        logging.info(f'embedding model to be used: {settings.EMBEDDING_MODEL}, DIMENSION: {settings.EMBEDDING_DIM}')
        return GoogleEmbedder()

    raise RuntimeError("unsupported embedder")


def get_vector_store(collection_name: Optional[str] = "insight_scope"):
    global _qdrantStore
    global _chromaStore
    settings = Settings()
    logging.info(f'collection name to be used: {collection_name}')
    if settings.VECTOR_STORE == "chroma":
        if _chromaStore is None:
            _chromaStore = ChromaStore(collection_name)
        return _chromaStore  # persist_dir=Settings.CHROMA_DIR
    elif settings.VECTOR_STORE == "qdrant":
        if _qdrantStore is None:
            _qdrantStore = QdrantStore(collection_name)
        return _qdrantStore  # persist in cloud
    raise RuntimeError("unsupported vector store")

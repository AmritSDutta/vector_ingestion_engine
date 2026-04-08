import logging
from typing import Dict, Sequence, Optional

import chromadb
from chromadb.config import Settings as Chroma_Settings

from app.config.config import Settings


class VectorStore:
    def save(self, ids: Sequence[str], docs: Sequence[str], metas: Sequence[Dict],
             embeddings: Sequence[Sequence[float]]):
        raise NotImplementedError

    def create(self, collection_name: Optional[str] = "insight_scope"):
        raise NotImplementedError

    def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        raise NotImplementedError

    def delete_collection(self, name: str): ...


class ChromaStore(VectorStore):
    def __init__(self, collection_name: Optional[str] = "insight_scope"):
        settings = Settings()
        client_settings = Chroma_Settings(chroma_db_impl="duckdb+parquet", persist_directory=settings.CHROMA_DIR)
        # self.client = chromadb.Client()
        self.client = chromadb.PersistentClient()
        self.collection_name = collection_name
        self.col = None

    def create(self, collection_name_forced: Optional[str] = None):
        try:
            effective_collection_name = collection_name_forced if collection_name_forced else self.collection_name
            self.col = self.client.get_or_create_collection(
                effective_collection_name
            )
            logging.info(f'creating effective collection: {effective_collection_name}')
        except Exception as e:
            logging.error('ChromaStore initialization error', e)
            raise e

    def save(self, ids: Sequence[str], docs: Sequence[str], metas: Sequence[Dict],
             embeddings: Sequence[Sequence[float]]):
        self.col.add(ids=list(ids), documents=list(docs), metadatas=list(metas), embeddings=list(embeddings))

    def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        return self.col.query(query_embeddings=[list(query_embedding)], n_results=n_results,
                              include=["documents", "metadatas", "distances"])

    def delete_collection(self, name: str):
        logging.warning(f'deleting collection: {name}')
        self.client.delete_collection(name)

    def list_collection(self):
        logging.info('listing collection')
        # [collection(name="my_collection", metadata={})]
        chroma_collections_list = [collection.name for collection in self.client.list_collections()]
        logging.info(f'found {len(chroma_collections_list)} collections')

        return chroma_collections_list

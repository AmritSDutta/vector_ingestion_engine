from typing import Optional, Sequence, Dict, Any

from pandas import DataFrame

from app.services.vector_store.vector_store import VectorStore


class PGVectorStore(VectorStore):
    def save(self, data: DataFrame):
        raise NotImplementedError

    def create(self, collection_name: Optional[str] = "insight_scope"):
        raise NotImplementedError

    def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        raise NotImplementedError

    def delete_collection(self, name: str):
        raise NotImplementedError

    def list_collection(self) -> list[str]:
        raise NotImplementedError

    def hybrid_search(self, query_embedding: Sequence[float],
                      n_results: int = 3, query: str = '') -> dict[str, list[Any]]:
        raise NotImplementedError
from typing import Dict, Sequence, Optional

from pandas import DataFrame


class VectorStore:
    def save(self, data: DataFrame):
        raise NotImplementedError

    def create(self, collection_name: Optional[str] = "insight_scope"):
        raise NotImplementedError

    def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        raise NotImplementedError

    def delete_collection(self, name: str): ...

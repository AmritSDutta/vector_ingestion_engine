from typing import Dict, Sequence, Optional


class VectorStore:
    def save(self, ids: Sequence[str], docs: Sequence[str], metas: Sequence[Dict],
             embeddings: Sequence[Sequence[float]]):
        raise NotImplementedError

    def create(self, collection_name: Optional[str] = "insight_scope"):
        raise NotImplementedError

    def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        raise NotImplementedError

    def delete_collection(self, name: str): ...

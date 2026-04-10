from typing import Dict, Sequence, Optional, Any

from fastembed.rerank.cross_encoder import TextCrossEncoder
from pandas import DataFrame


def get_reranker_model() -> TextCrossEncoder:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')


def validate_collection_name(collection_name: str):
    import re
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', collection_name):
        raise ValueError(
            f"Invalid collection name '{collection_name}'. "
            "Must start with letter and contain only letters, numbers, and underscores."
        )


class VectorStore:
    async def save(self, data: DataFrame):
        raise NotImplementedError

    async def create(self):
        raise NotImplementedError

    async def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        raise NotImplementedError

    async def delete_collection(self):
        raise NotImplementedError

    async def list_collection(self) -> list[str]:
        raise NotImplementedError

    async def hybrid_search(self, query_embedding: Sequence[float],
                            n_results: int = 3, query: str = '') -> dict[str, list[Any]]:
        raise NotImplementedError

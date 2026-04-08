import logging
from typing import Sequence, Dict, Optional

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import models, CollectionsResponse

from app.config.config import Settings
from app.rag.vector_store import VectorStore


class QdrantStore(VectorStore):

    def __init__(self, collection_name: Optional[str] = "insight_scope"):
        self.collection_name = collection_name
        settings = Settings()
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        self.reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')

    def create(self, collection_name_overridden: Optional[str] = None):
        settings = Settings()
        effective_collection_name = collection_name_overridden if collection_name_overridden else self.collection_name

        try:
            if self.qdrant_client.collection_exists(effective_collection_name):
                logging.info(f'existing collection: {effective_collection_name}')
                return

            logging.info(f'creating collection: {effective_collection_name}')
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                    on_disk=True
                )
            )
            logging.info(f'created qdrant collection: {effective_collection_name}')

        except Exception as e:
            logging.error('qdrant initialization error', e)

    def save(self, ids: Sequence[str], docs: Sequence[str], metas: Sequence[Dict],
             embeddings: Sequence[Sequence[float]]):
        try:
            payload = [
                {**meta, "doc": doc}
                for doc, meta in zip(docs, metas)
            ]
            self.qdrant_client.upload_collection(
                self.collection_name,
                vectors=[list(row) for row in embeddings],
                payload=payload,
            )
            logging.info(f'uploaded data to  collection: {self.collection_name}')
        except Exception as e:
            logging.error('qdrant persistence error', e)

    def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        # 1. dense ranking
        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=list(query_embedding),
            limit=n_results).points
        # Extract descriptions for reranking
        descriptions = [hit.payload["doc"] for hit in hits]

        # 2. Rerank using cross-encoder / LLM reranker
        rerank_scores = self.reranker.rerank(query, descriptions)  # list[float]

        # 3. Combine dense score + rerank score
        combined = []
        for hit, rerank_score in zip(hits, rerank_scores):
            logging.info(f"""
                hit.score: {float(hit.score)},
                rerank_score: {float(rerank_score)},
                sneak peek {hit.payload["doc"][:25]} ....
                """
                         )

            combined_score = float(hit.score) + float(rerank_score)
            combined.append({
                "payload": hit.payload,
                "dense_score": float(hit.score),
                "rerank_score": float(rerank_score),
                "final_score": combined_score
            })

        # 4. Sort by final score descending
        combined.sort(key=lambda x: x["final_score"], reverse=True)
        return {"results": combined}

    def delete_collection(self, name: str):
        self.qdrant_client.delete_collection(self.collection_name)
        logging.info(f'deleted collection: {self.collection_name}')

    def list_collection(self) -> list[str]:
        logging.info('listing collection')
        response: CollectionsResponse = self.qdrant_client.get_collections()
        logging.info(f'returned collections: {response.collections}')
        existing_collections_list = [collection.name for collection in response.collections]
        logging.info(f'found {len(existing_collections_list)} collections')
        return existing_collections_list

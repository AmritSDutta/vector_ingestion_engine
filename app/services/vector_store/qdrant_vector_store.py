import logging
from typing import Sequence, Optional, Any

import qdrant_client
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
from pandas import DataFrame
from qdrant_client.http.models import models, CollectionsResponse, UpdateResult

from app.config.config import get_settings
from app.services.vector_store.vector_store import VectorStore, get_reranker_model, validate_collection_name

logger = logging.getLogger(__name__)


class QdrantStore(VectorStore):

    def __init__(self):
        settings = get_settings()
        self.collection_name = settings.COLLECTION_NAME
        validate_collection_name(self.collection_name)

        self.qdrant_client = qdrant_client.AsyncQdrantClient(
            url=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
        )
        self.bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25", threads=2)
        self.late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0", threads=4)
        self.reranker = get_reranker_model()

    async def create(self):
        settings = get_settings()
        effective_collection_name = self.collection_name
        validate_collection_name(effective_collection_name)

        try:
            if await self.qdrant_client.collection_exists(effective_collection_name):
                logging.info(f'existing collection: {effective_collection_name}')
                return

            logging.info(f'creating collection: {effective_collection_name}')
            await self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "genai": models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=models.Distance.COSINE,
                        quantization_config=models.ScalarQuantization(
                            scalar=models.ScalarQuantizationConfig(
                                type=models.ScalarType.INT8,
                                quantile=0.99,
                                always_ram=True,
                            ),
                        ),
                        on_disk=True
                    ),
                    "colbert": models.VectorParams(
                        size=self.late_interaction_embedding_model.embedding_size,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM,
                        ),
                        hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for reranking
                    ),
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            logging.info(f'created qdrant collection: {effective_collection_name}')

        except Exception as e:
            logging.error(f'qdrant initialization error {e}', exc_info=True)
            raise

    async def save(self, data: DataFrame):
        try:
            import numpy as np
            from qdrant_client.http import models

            # 1. Generate Embeddings
            # Ensure 'overall' text column exists and has no NaNs
            texts = data["overall"].fillna("").tolist()

            bm25_embeddings = list(self.bm25_embedding_model.embed(texts))
            late_interaction_embeddings = list(self.late_interaction_embedding_model.embed(texts))
            dense_embeddings = data["embeddings"].tolist()

            points = []
            for i, row in data.iterrows():
                # 2. Convert NumPy types to Python native for JSON serialization
                payload = {
                    "ResumeID": int(row["ResumeID"]) if isinstance(row["ResumeID"], (np.integer, int)) else str(
                        row["ResumeID"]),
                    "Name": str(row["Name"]),
                    "Category": str(row["Category"]),
                    "Education": str(row["Education"]),
                    "Skills": [s.strip() for s in row["Skills"].split(",")],
                    "Summary": str(row["Summary"]),
                    "doc": str(row["Summary"]),
                    "Phone": str(row["Phone"]),
                    "Location": str(row["Location"])
                }

                # 3. Construct PointStruct with named vectors
                points.append(models.PointStruct(
                    id=int(i),  # Or use row["ResumeID"] if it's a valid UUID/int
                    vector={
                        "genai": dense_embeddings[i],
                        "colbert": late_interaction_embeddings[i].tolist(),
                        "bm25": models.SparseVector(
                            indices=bm25_embeddings[i].indices.tolist(),
                            values=bm25_embeddings[i].values.tolist()
                        )
                    },
                    payload=payload
                )
                )

            # 4. Use upsert for better handling of mixed vector types in batches
            batch_size = get_settings().BATCH_SIZE
            total_points = len(points)
            for i in range(0, total_points, batch_size):
                batch_points = points[i:i + batch_size]
                result: UpdateResult = await self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
                logging.info(
                    f'Successfully uploaded batch {i // batch_size + 1}, {len(batch_points)} points. Status: {result.status}')
            logging.info(f'Successfully uploaded total {total_points} points to {self.collection_name}')

        except Exception as e:
            logging.error(f'Qdrant persistence error: {e}', exc_info=True)
            raise

    async def query(self, query_embedding: Sequence[float], n_results: int = 3,
                    query: str = '') -> dict[str, list[Any]]:
        # 1. dense ranking
        try:
            hits = await self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=list(query_embedding),
                using="genai",
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
            logging.info(f'{len(combined)} combined results')
            return {"results": combined}

        except Exception as e:
            logging.error(f'Qdrant persistence error: {e}', exc_info=True)
            raise

    async def delete_collection(self) -> Optional[str]:
        validate_collection_name(self.collection_name)
        if self.collection_name is None:
            logging.info(f"collection name is None ")
            return None
        try:
            await self.qdrant_client.delete_collection(self.collection_name)
            logging.info(f'deleted collection: {self.collection_name}')
            return self.collection_name
        except Exception as e:
            logging.error(f'Qdrant persistence error: {e}', exc_info=True)
            raise

    async def list_collection(self) -> list[str]:
        logging.info('listing collection')
        try:
            response: CollectionsResponse = await self.qdrant_client.get_collections()
            logging.info(f'returned collections: {response.collections}')
            existing_collections_list = [collection.name for collection in response.collections]
            logging.info(f'found {len(existing_collections_list)} collections')
            return existing_collections_list
        except Exception as e:
            logging.error(f'Qdrant persistence error: {e}', exc_info=True)
            raise

    async def hybrid_search(self, query_embedding: Sequence[float],
                            n_results: int = 3, query: str = '') -> dict[str, list[Any]]:
        from qdrant_client.http import models

        # 1. Generate query components locally using FastEmbed
        sparse_query = next(self.bm25_embedding_model.query_embed(query))
        colbert_query = next(self.late_interaction_embedding_model.query_embed(query))

        try:
            # 2. Single-call Multi-stage Search: (Dense + Sparse) -> RRF -> ColBERT Rerank
            response = await self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        prefetch=[
                            models.Prefetch(query=query_embedding, using="genai", limit=n_results * 10),
                            models.Prefetch(
                                query=models.SparseVector(
                                    indices=sparse_query.indices.tolist(),
                                    values=sparse_query.values.tolist()
                                ),
                                using="bm25",
                                limit=n_results * 10
                            ),
                        ],
                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                        limit=n_results * 10  # This pool is sent to ColBERT reranking
                    )
                ],
                query=colbert_query.tolist(),  # The "Reranker"
                using="colbert",
                limit=n_results
            )
            logging.info(f'query results: {len(response.points)}')
            return {"results": [p.payload for p in response.points]}
        except Exception as e:
            logging.error(f'Qdrant persistence error: {e}', exc_info=True)
            raise

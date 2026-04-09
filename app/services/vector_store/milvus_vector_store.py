import logging
from typing import Sequence, Dict, Optional, Any

from pandas import DataFrame
from pymilvus import MilvusClient, DataType, Function, FunctionType, MilvusException, AnnSearchRequest

from app.config.config import get_settings
from app.services.vector_store.vector_store import VectorStore, get_reranker_model

logger = logging.getLogger(__name__)


class MilvusStore(VectorStore):

    def __init__(self, collection_name: Optional[str] = "resume_details"):
        settings = get_settings()
        self.collection_name = collection_name if collection_name else settings.COLLECTION_NAME

        self.client = MilvusClient(
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN,
            timeout=30,
            secure=True
        )
        logging.info(f"Connected to DB: {settings.MILVUS_URI} successfully")

        # Check if the collection exists
        check_collection = self.client.has_collection(self.collection_name)

        if check_collection:
            logging.info(f"Existing collection {self.collection_name} confirmed")
        self.reranker = get_reranker_model()

    async def create(self, collection_name_overridden: Optional[str] = None):
        settings = get_settings()
        coll_name = self.collection_name

        logging.info(f"Creating Milvus collection: {coll_name}")
        if self.client.has_collection(collection_name=coll_name):
            logging.info(f"Collection {coll_name} already exists.")
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="ResumeID", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name="Name", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM)
        schema.add_field(field_name="Summary", datatype=DataType.VARCHAR, max_length=10000,
                         enable_analyzer=True)  # Text field
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        bm25_function = Function(
            name="summary_text_bm25_emb",  # Function name
            input_field_names=["Summary"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings, set to `BM25`
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="sparse",

            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            }

        )
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="AUTOINDEX",
            index_name="vector_index"
        )

        self.client.create_collection(
            collection_name=coll_name,
            schema=schema,
            index_params=index_params
        )
        logging.info(f"Created Milvus collection: {coll_name}")

    async def save(self, data: DataFrame):
        try:
            import numpy as np
            coll_name = self.collection_name

            # Prepare data list for insertion
            insert_data = []
            for _, row in data.iterrows():
                record = {
                    "ResumeID": row["ResumeID"],
                    "Summary": row["Summary"],
                    "vector": np.array(row["embeddings"]).tolist(),
                    "Name": row["Name"],
                    "Category": row["Category"],
                    "Education": row["Education"],
                    "Skills": row["Skills"],
                    "doc": row["Summary"]
                }
                insert_data.append(record)

            self.client.insert(collection_name=coll_name, data=insert_data)
            logging.info(f"Uploaded {len(insert_data)} records to Milvus collection: {coll_name}")

        except MilvusException as milvus_exception:
            logging.error(f"Milvus persistence error: {milvus_exception}")

    async def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        # 1. Search in Milvus
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[list(query_embedding)],
            limit=n_results,
            anns_field='vector',
            output_fields=["ResumeID", "Name", "Category", "Education", "Skills", "Summary", "doc"]
        )

        hits = search_res[0]
        descriptions = [hit["entity"]["doc"] for hit in hits]

        # 2. Rerank
        rerank_scores = self.reranker.rerank(query, descriptions)

        # 3. Combine
        combined = []
        for hit, rerank_score in zip(hits, rerank_scores):
            dense_score = hit["distance"]
            combined_score = float(dense_score) + float(rerank_score)
            combined.append({
                "payload": hit["entity"],
                "dense_score": float(dense_score),
                "rerank_score": float(rerank_score),
                "final_score": combined_score
            })

        combined.sort(key=lambda x: x["final_score"], reverse=True)
        return {"results": combined}

    async def delete_collection(self, name: Optional[str] = None) -> Optional[str]:
        if name is not None and name != self.collection_name:
            logging.info(f"collection name: {self.collection_name}, mismatched with : {name} ")
            return None

        check_collection = self.client.has_collection(self.collection_name)
        if check_collection:
            logging.info(f"dropping Existing collection {self.collection_name} confirmed")
            self.client.drop_collection(self.collection_name)
        logging.info(f"Deleted Milvus collection if existed: {self.collection_name}")
        return self.collection_name

    async def list_collection(self) -> list[str]:
        return self.client.list_collections()

    async def hybrid_search(self, query_embedding: Sequence[float],
                      n_results: int = 3, query: str = '') -> dict[str, list[Any]]:
        logging.info(f"Hybrid search with: {query}")
        # text semantic search (dense)
        search_param_1 = {
            "data": [query_embedding],
            "anns_field": "vector",
            "param": {"nprobe": 10},
            "limit": n_results
        }
        request_1 = AnnSearchRequest(**search_param_1)

        # full-text search (sparse)
        search_param_2 = {
            "data": [query],
            "anns_field": "sparse",
            "param": {"nprobe": 10},
            "limit": n_results
        }
        request_2 = AnnSearchRequest(**search_param_2)
        reqs = [request_1, request_2]

        ranker = Function(
            name="rrf",
            input_field_names=[],  # Must be an empty list
            function_type=FunctionType.RERANK,
            params={
                "reranker": "rrf",
                "k": 100  # Optional
            }
        )
        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=ranker,
            output_fields=["ResumeID", "Name", "Category", "Education", "Skills", "Summary", "doc"],
            limit=n_results
        )

        # Transformation Logic
        formatted_results = []
        for hit in res[0]:
            formatted_results.append({
                "payload": {
                    "Name": hit.entity.get("Name"),
                    "Summary": hit.entity.get("Summary"),
                    "ResumeID": hit.entity.get("ResumeID"),
                    "Category": hit.entity.get("Category"),
                    "Education": hit.entity.get("Education"),
                    "Skills": hit.entity.get("Skills"),
                    "doc": hit.entity.get("doc"),
                },
                "dense_score": 0.0,  # Placeholder: Milvus RRF combines scores
                "rerank_score": 0.0,  # Placeholder
                "final_score": hit.distance
            })

        return {"results": formatted_results}

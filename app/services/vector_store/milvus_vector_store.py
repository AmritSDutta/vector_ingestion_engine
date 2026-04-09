import logging
from typing import Sequence, Dict, Optional

from fastembed.rerank.cross_encoder import TextCrossEncoder
from pandas import DataFrame
from pymilvus import MilvusClient, DataType

from app.config.config import get_settings
from app.services.vector_store.vector_store import VectorStore


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
        print(f"Connected to DB: {settings.MILVUS_URI} successfully")

        # Check if the collection exists
        check_collection = self.client.has_collection(self.collection_name)

        if check_collection:
            print(f"Existing collection {self.collection_name} confirmed")
        self.reranker = TextCrossEncoder(model_name='jinaai/jina-reranker-v2-base-multilingual')

    def create(self, collection_name_overridden: Optional[str] = None):
        settings = get_settings()
        coll_name = self.collection_name

        logging.info(f"Creating Milvus collection: {coll_name}")
        if self.client.has_collection(collection_name=coll_name):
            logging.info(f"Collection {coll_name} already exists.")
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(field_name="ResumeID", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="AUTOINDEX",
            index_name="vector_index"
        )
        logging.info(f"Creating Milvus collection: {coll_name}")
        self.client.create_collection(
            collection_name=coll_name,
            schema=schema,
            index_params=index_params
        )
        logging.info(f"Created Milvus collection: {coll_name}")

    def save(self, data: DataFrame):
        try:
            import numpy as np
            coll_name = self.collection_name

            # Prepare data list for insertion
            insert_data = []
            for _, row in data.iterrows():
                record = {
                    "ResumeID": row["ResumeID"],
                    "vector": np.array(row["embeddings"]).tolist(),
                    "Name": row["Name"],
                    "Category": row["Category"],
                    "Education": row["Education"],
                    "Skills": row["Skills"],
                    "Summary": row["Summary"],
                    "doc": row["Summary"]
                }
                insert_data.append(record)

            self.client.insert(collection_name=coll_name, data=insert_data)
            logging.info(f"Uploaded {len(insert_data)} records to Milvus collection: {coll_name}")
        except Exception as e:
            logging.error(f"Milvus persistence error: {e}")

    def query(self, query_embedding: Sequence[float], n_results: int = 3, query: str = '') -> Dict:
        # 1. Search in Milvus
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[list(query_embedding)],
            limit=n_results,
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

    def delete_collection(self, name: Optional[str] = None)-> Optional[str]:
        if name is not None and name != self.collection_name:
            logging.info(f"collection name: {self.collection_name}, mismatched with : {name} ")
            return None

        check_collection = self.client.has_collection(self.collection_name)
        if check_collection:
            logging.info(f"dropping Existing collection {self.collection_name} confirmed")
            self.client.drop_collection(self.collection_name)
        logging.info(f"Deleted Milvus collection if existed: {self.collection_name}")
        return self.collection_name

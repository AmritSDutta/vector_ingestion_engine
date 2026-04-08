import logging
import numpy as np
import pandas as pd

from app.config.config import get_settings
from app.services.embedding.EmbeddingFactory import get_embedding_service
from app.services.vector_store.VectoreStoreFaactory import get_vector_store


def _get_custom_embedding(texts: list[str]):
    """Generate a Gemini embedding for a given text."""
    embedding_service = get_embedding_service()
    return embedding_service.embed_batch(texts)


def _get_vector_Store():
    """Generate a Gemini embedding for a given text."""
    return get_vector_store()


async def ingest_and_store_embedding():
    logging.info(f'uploaded files stored in : {get_settings().data_file_path}')
    pd_data = pd.read_json(get_settings().data_file_path, lines=True)
    data = pd_data.iloc[:-1].copy()
    logging.info(f"Processing rows {pd_data.columns} into DB")
    logging.info(f"Total rows selected from file: {len(data)}")

    texts_to_embed: list[str] = data["overall"].tolist()
    embeddings = _get_custom_embedding(texts_to_embed)
    data["embeddings"] = embeddings

    data_list = [(
            row["ResumeID"], row["Name"], row["Category"], row["Education"],
            row["Skills"], row["Summary"], np.array(row["embeddings"])
        )
        for _, row in data.iterrows()
    ]
    vstore = _get_vector_Store()
    vstore.create()
    vstore.save(data)
    logging.info(f'indexes built and stored in vector store')

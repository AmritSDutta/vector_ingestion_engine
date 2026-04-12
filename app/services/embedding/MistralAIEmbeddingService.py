import logging
import os
from typing import Sequence

from mistralai.client import Mistral

from app.config.config import get_settings
from app.services.embedding.base import EmbeddingService

logger = logging.getLogger(__name__)


class MistralAIEmbeddingService(EmbeddingService):
    def __init__(self, api_key: str = None):
        settings = get_settings()
        self.client = Mistral(api_key=api_key) if api_key else Mistral(os.getenv('MISTRAL_API_KEY'))
        self.model: str = settings.EMBEDDING_MODEL
        self.dimension: int = settings.EMBEDDING_DIM
        self.type: str = "semantic_similarity"

    async def embed(self, texts: str, task_type: str = None,
                    output_dimensionality: int = None) -> list[float] | None:
        """mistral ai embedding for text collections."""
        resp = await self.client.embeddings.create_async(
            model=self.model,
            inputs=[texts],
        )
        logging.debug(len(resp.data[0].embedding))
        return resp.data[0].embedding

    async def embed_batch(
            self,
            texts: Sequence[str],
            batch_size: int = 50,
            output_dimensionality: int = 1024,
    ):
        """Batch embedding for large text collections."""

        texts = list(texts)
        if not texts:
            return []

        final_embeddings = []

        chunks = [texts[x: x + batch_size] for x in range(0, len(texts), batch_size)]
        embeddings_response = [
            await self.client.embeddings.create_async(model=self.model, inputs=c) for c in chunks
        ]
        final_embeddings.extend([d.embedding for e in embeddings_response for d in e.data])

        return final_embeddings

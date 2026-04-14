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
            batch_size: int = 5,
            output_dimensionality: int = 1024,
    ):
        """Batch embedding for large text collections."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        max_tokens = 8000

        texts = list(texts)
        if not texts:
            return []

        # Truncate each text to max_tokens
        truncated_texts = []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                truncated_text = tokenizer.decode(truncated_tokens)
                truncated_texts.append(truncated_text)
            else:
                truncated_texts.append(text)

        final_embeddings = []

        chunks = [truncated_texts[x: x + batch_size] for x in range(0, len(truncated_texts), batch_size)]
        embeddings_response = [
            await self.client.embeddings.create_async(model=self.model, inputs=c) for c in chunks
        ]
        final_embeddings.extend([d.embedding for e in embeddings_response for d in e.data])

        return final_embeddings
